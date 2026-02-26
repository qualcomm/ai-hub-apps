# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import queue
import subprocess
import time
import warnings
from typing import Any

import cv2
import gi
import numpy as np
import utils.bbox_processing as BBOX
import utils.constants as C
import utils.image_processing as IMG
import utils.model_io_processing as IO
import utils.webui as ui
from ai_edge_litert.interpreter import Delegate, Interpreter
from utils.draw import draw_predictions
from utils.input_processing import get_gstreamer_input_pipeline

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

outq: queue.Queue[np.ndarray[Any, np.dtype[np.uint8]]] = queue.Queue(maxsize=4)


def dequantize(values, zero_points, scales):
    if zero_points.size == 0 or scales.size == 0:
        return values.astype(np.float32)

    return ((values - np.int32(zero_points)) * np.float64(scales)).astype(np.float32)


def quantize(values, zero_points, scales):
    v = np.asarray(values, dtype=np.float32)
    z = np.asarray(zero_points, dtype=np.int32)
    s = np.asarray(scales, dtype=np.float64)

    q_float = np.rint(v / s) + z

    info = np.iinfo(np.uint8)
    q_clipped = np.clip(q_float, info.min, info.max)

    return q_clipped.astype(np.uint8, copy=False)


def on_new_sample(sink):
    sample = sink.emit("pull-sample")
    buf = sample.get_buffer()
    caps = sample.get_caps().get_structure(0)
    w, h = caps.get_value("width"), caps.get_value("height")

    # Map buffer memory as read-only
    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        return Gst.FlowReturn.OK
    try:
        rowstride = mapinfo.size // h
        arr = np.frombuffer(mapinfo.data, dtype=np.uint8, count=h * rowstride)
        arr = arr.reshape(h, rowstride)[:, : w * 3].copy()
        arr = arr.reshape((h, w, 3))
    finally:
        buf.unmap(mapinfo)

    try:
        outq.put_nowait(arr)
    except queue.Full:
        pass

    return Gst.FlowReturn.OK


def main(args):
    if args.list_devices:
        subprocess.call(["v4l2-ctl", "--list-devices"])
        return

    Gst.init(None)

    if args.video_gstreamer_source:
        video_source = args.video_gstreamer_source
    else:
        video_source = f"v4l2src name=camsrc device={args.video_device}"
    pipeline = Gst.parse_launch(
        get_gstreamer_input_pipeline(
            video_source, (args.video_source_width, args.video_source_height)
        )
    )
    appsink = pipeline.get_by_name("appsink")
    if not appsink:
        raise RuntimeError("Could not find appsink element named 'sink'")

    appsink.set_property("emit-signals", True)
    appsink.connect("new-sample", on_new_sample)

    delegate_path = "libQnnTFLiteDelegate.so"

    hand_detector = Interpreter(
        "models/PalmDetector.tflite",
        experimental_delegates=[
            Delegate(
                delegate_path,
                {
                    "backend_type": "htp",
                    "htp_performance_mode": "2",
                    "log_level": "1",
                },
            )
        ],
    )
    landmark_detector = Interpreter(
        "models/HandLandmarkDetector.tflite",
        experimental_delegates=[
            Delegate(
                delegate_path,
                {
                    "backend_type": "htp",
                    "htp_performance_mode": "2",
                    "log_level": "1",
                },
            )
        ],
    )

    gesture_classifier = Interpreter("models/CannedGestureClassifier.tflite")

    hand_detector.allocate_tensors()
    landmark_detector.allocate_tensors()
    gesture_classifier.allocate_tensors()

    detector_input = hand_detector.get_input_details()
    detector_output = hand_detector.get_output_details()

    landmark_input = landmark_detector.get_input_details()
    landmark_output = landmark_detector.get_output_details()

    classifier_input = gesture_classifier.get_input_details()
    classifier_output = gesture_classifier.get_output_details()

    print(
        "--------------------------- Gstreamer ----------------------------", flush=True
    )
    pipeline.set_state(Gst.State.PLAYING)
    start_time = time.perf_counter()
    frame_count = 0

    warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

    print(
        "--------------------------- Web server ----------------------------",
        flush=True,
    )
    try:
        ui.start_thread()
        while True:
            rgb_frame = outq.get(timeout=5)

            input_val, scale, pad = IMG.resize_pad(
                rgb_frame, (C.INPUT_WIDTH, C.INPUT_HEIGHT)
            )
            input_val = np.expand_dims(input_val, axis=0)

            hand_detector.set_tensor(detector_input[0]["index"], input_val)

            hand_detector.invoke()
            # Outputs:
            # - box_coords: <B, N, C>, where N == # of anchors & C == # of of coordinates
            #       Layout of C is (box_center_x, box_center_y, box_w, box_h, keypoint_0_x, keypoint_0_y, ..., keypoint_maxKey_x, keypoint_maxKey_y)
            # - box_scores: <B, N>, where N == # of anchors.
            box_coords = dequantize(
                hand_detector.get_tensor(detector_output[0]["index"]),
                zero_points=detector_output[0]["quantization_parameters"][
                    "zero_points"
                ],
                scales=detector_output[0]["quantization_parameters"]["scales"],
            )
            box_scores = dequantize(
                hand_detector.get_tensor(detector_output[1]["index"]),
                zero_points=detector_output[1]["quantization_parameters"][
                    "zero_points"
                ],
                scales=detector_output[1]["quantization_parameters"]["scales"],
            )
            box_coords = box_coords.reshape((*box_coords.shape[:-1], -1, 2))

            flattened_box_coords = box_coords.reshape(
                [*list(box_coords.shape)[:-2], -1]
            )

            # Run non maximum suppression on the output
            # batched_selected_coords = list[torcTensor(shape=[Num Boxes, 4])],
            # where 4 = (x0, y0, x1, y1)
            batched_selected_coords, *_ = BBOX.batched_nms(
                C.NMS_IOU_THRESHOLD,
                C.MIN_DETECTOR_BOX_SCORE,
                flattened_box_coords,
                box_scores,
            )

            selected_boxes = []
            selected_keypoints = []
            for i in range(len(batched_selected_coords)):
                selected_coords = batched_selected_coords[i]
                if len(selected_coords) != 0:
                    boxes_list = []
                    kps_list = []
                    for j in range(len(selected_coords)):
                        selected_coords_ = selected_coords[j : j + 1].reshape(
                            [*list(selected_coords[j : j + 1].shape)[:-1], -1, 2]
                        )

                        selected_coords_ = IMG.denormalize_coordinates(
                            selected_coords_, (1, 1), scale, pad
                        )

                        boxes_list.append(selected_coords_[:, :2])
                        kps_list.append(selected_coords_[:, 2:])

                    if boxes_list:
                        selected_boxes.append(np.concatenate(boxes_list, axis=0))
                        selected_keypoints.append(np.concatenate(kps_list, axis=0))

                    else:
                        selected_boxes.append(np.empty(0, dtype=np.float32))
                        selected_keypoints.append(np.empty(0, dtype=np.float32))
                else:
                    selected_boxes.append(np.empty(0, dtype=np.float32))
                    selected_keypoints.append(np.empty(0, dtype=np.float32))

            batched_roi_4corners = IO.compute_object_roi(
                selected_boxes, selected_keypoints
            )
            batched_roi_4corners = np.unstack(batched_roi_4corners[0])

            batched_selected_landmarks: list[np.ndarray] = []
            batched_is_right_hand: list[list[bool]] = []
            batched_gesture_labels: list[list[str] | None] = []

            for _, roi_4corners in enumerate(batched_roi_4corners):
                if roi_4corners.size == 0:
                    continue

                affines = BBOX.compute_box_affine_crop_resize_matrix(
                    roi_4corners[np.newaxis, :, :3], (224, 224)
                )
                # Create input images by applying the affine transforms.
                keypoint_net_inputs = IMG.apply_batched_affines_to_frame(
                    rgb_frame, affines, (224, 224)
                ).astype(np.uint8, copy=False)

                landmark_detector.set_tensor(
                    landmark_input[0]["index"], keypoint_net_inputs
                )

                # Compute landmarks.
                landmark_detector.invoke()

                landmarks = dequantize(
                    landmark_detector.get_tensor(landmark_output[0]["index"]),
                    zero_points=landmark_output[0]["quantization_parameters"][
                        "zero_points"
                    ],
                    scales=landmark_output[0]["quantization_parameters"]["scales"],
                ).reshape(1, 21, 3)

                ld_scores = dequantize(
                    landmark_detector.get_tensor(landmark_output[1]["index"]),
                    zero_points=landmark_output[1]["quantization_parameters"][
                        "zero_points"
                    ],
                    scales=landmark_output[1]["quantization_parameters"]["scales"],
                )
                lr = dequantize(
                    landmark_detector.get_tensor(landmark_output[2]["index"]),
                    zero_points=landmark_output[2]["quantization_parameters"][
                        "zero_points"
                    ],
                    scales=landmark_output[2]["quantization_parameters"]["scales"],
                )

                all_landmarks = []
                all_lr = []
                gesture_label = []
                for ld_batch_idx in range(landmarks.shape[0]):
                    # Exclude landmarks that don't meet the appropriate score threshold.
                    if ld_scores[ld_batch_idx] >= C.MIN_LANDMARK_SCORE:
                        # Apply the inverse of affine transform used above to the landmark coordinates.
                        # This will convert the coordinates to their locations in the original input image.
                        inverted_affine = cv2.invertAffineTransform(
                            affines[ld_batch_idx]
                        ).astype(np.float32)
                        landmarks[ld_batch_idx][
                            :, :2
                        ] = IMG.apply_affine_to_coordinates(
                            landmarks[ld_batch_idx][:, :2], inverted_affine
                        )

                        # Add the predicted landmarks to our list.
                        all_landmarks.append(landmarks[ld_batch_idx])
                        all_lr.append(np.round(lr[ld_batch_idx]).item() == 1)

                        hand = np.expand_dims(landmarks[ld_batch_idx], axis=0)
                        lr = np.expand_dims(lr[ld_batch_idx], axis=0)

                        x64_a = IO.preprocess_hand_x64(hand, lr, mirror=False)
                        x64_b = IO.preprocess_hand_x64(hand, lr, mirror=True)

                        x64_a = quantize(
                            x64_a,
                            zero_points=classifier_input[0]["quantization_parameters"][
                                "zero_points"
                            ],
                            scales=classifier_input[0]["quantization_parameters"][
                                "scales"
                            ],
                        )
                        x64_b = quantize(
                            x64_b,
                            zero_points=classifier_input[1]["quantization_parameters"][
                                "zero_points"
                            ],
                            scales=classifier_input[1]["quantization_parameters"][
                                "scales"
                            ],
                        )

                        gesture_classifier.set_tensor(
                            classifier_input[0]["index"], x64_a
                        )
                        gesture_classifier.set_tensor(
                            classifier_input[1]["index"], x64_b
                        )

                        gesture_classifier.invoke()

                        score = gesture_classifier.get_tensor(
                            classifier_output[0]["index"]
                        )

                        score = dequantize(
                            score,
                            zero_points=classifier_output[0]["quantization_parameters"][
                                "zero_points"
                            ],
                            scales=classifier_output[0]["quantization_parameters"][
                                "scales"
                            ],
                        )

                        gesture_id = np.argmax(score.flatten())
                        gesture_label.append(C.GESTURE_LABELS[gesture_id])

                # Add this batch of landmarks to the output list.
                batched_selected_landmarks.append(
                    np.stack(all_landmarks, axis=0)
                    if all_landmarks
                    else np.empty(0, dtype=np.float32)
                )
                batched_is_right_hand.append(all_lr)
                batched_gesture_labels.append(gesture_label)
            # Add None for these lists, since this batch has no predicted bounding boxes.
            batched_selected_landmarks.append(np.empty(0, dtype=np.float32))
            batched_is_right_hand.append([])
            batched_gesture_labels.append([])

            draw_predictions(
                [rgb_frame] * len(batched_selected_landmarks),
                batched_selected_landmarks,
                batched_is_right_hand,
                batched_gesture_labels,
                landmark_connections=C.HAND_LANDMARK_CONNECTIONS,
            )
            cur_time = time.perf_counter()
            frame_count += 1
            if cur_time - start_time > 1.0:
                start_time += 1.0
                print("FPS:", frame_count, flush=True)
                frame_count = 0

            ui.set_frame(rgb_frame[..., ::-1])

    except queue.Empty:
        print("Timed out waiting for input! Exiting...")
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand Gesture Recognition")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list-devices", action="store_true", help="List options for --video-device"
    )
    group.add_argument(
        "--video-device",
        type=str,
        help='GStreamer v4l2src video device (e.g. "/dev/video0")',
    )
    group.add_argument(
        "--video-gstreamer-source",
        type=str,
        help='GStreamer video source (e.g. "v4l2src device=/dev/video2" or "qtiqmmfsrc name=camsrc camera=0")',
    )
    parser.add_argument(
        "--video-source-width",
        type=int,
        required=False,
        default=1024,
        help="Video width (input), default 1024",
    )
    parser.add_argument(
        "--video-source-height",
        type=int,
        required=False,
        default=768,
        help="Video height (input), default 768",
    )

    args = parser.parse_args()
    main(args)
