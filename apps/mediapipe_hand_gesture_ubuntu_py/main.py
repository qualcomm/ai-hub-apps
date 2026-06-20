# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import contextlib
import queue
import subprocess
import warnings
from pathlib import Path
from typing import Any

import cv2
import gi
import numpy as np
import qai_hub_apps_utils.webui as ui
import utils.constants as C
from ai_edge_litert.interpreter import Delegate, Interpreter
from qai_hub_apps_utils.bbox_processing import (
    batched_nms,
    compute_box_affine_crop_resize_matrix,
)
from qai_hub_apps_utils.fps import FpsCounter
from qai_hub_apps_utils.image_processing import (
    apply_affine_to_coordinates,
    apply_batched_affines_to_frame,
    denormalize_coordinates,
    resize_pad,
)
from qai_hub_apps_utils.quantization import dequantize, quantize
from utils.draw import draw_predictions
from utils.input_processing import get_gstreamer_input_pipeline
from utils.model_io_processing import compute_object_roi, preprocess_hand_x64

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

outq: queue.Queue[np.ndarray[Any, np.dtype[np.uint8]]] = queue.Queue(maxsize=4)


def on_new_sample(sink: Any) -> Any:
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

    with contextlib.suppress(queue.Full):
        outq.put_nowait(arr)

    return Gst.FlowReturn.OK


def detect_hands(
    rgb_frame: np.ndarray,
    hand_detector: Interpreter,
    detector_input: list[dict[str, Any]],
    detector_output: list[dict[str, Any]],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run palm detection on rgb_frame.

    Parameters
    ----------
    rgb_frame
        Input RGB image as a numpy array.
    hand_detector
        TFLite interpreter for palm detection.
    detector_input
        Input tensor details from hand_detector.get_input_details().
    detector_output
        Output tensor details from hand_detector.get_output_details().

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        selected_boxes and selected_keypoints, one array per batch item.
    """
    input_val, scale, pad = resize_pad(rgb_frame, (C.INPUT_WIDTH, C.INPUT_HEIGHT))
    input_val = np.expand_dims(input_val, axis=0)

    hand_detector.set_tensor(detector_input[0]["index"], input_val)
    hand_detector.invoke()

    # Outputs:
    # - box_coords: <B, N, C>, where N == # of anchors & C == # of coordinates
    #       Layout of C is (box_center_x, box_center_y, box_w, box_h,
    #                        keypoint_0_x, keypoint_0_y, ..., keypoint_maxKey_x, keypoint_maxKey_y)
    # - box_scores: <B, N>, where N == # of anchors.
    box_coords = dequantize(
        hand_detector.get_tensor(detector_output[0]["index"]),
        zero_points=detector_output[0]["quantization_parameters"]["zero_points"],
        scales=detector_output[0]["quantization_parameters"]["scales"],
    )
    box_scores = dequantize(
        hand_detector.get_tensor(detector_output[1]["index"]),
        zero_points=detector_output[1]["quantization_parameters"]["zero_points"],
        scales=detector_output[1]["quantization_parameters"]["scales"],
    )
    box_coords = box_coords.reshape((*box_coords.shape[:-1], -1, 2))
    flattened_box_coords = box_coords.reshape([*list(box_coords.shape)[:-2], -1])

    batched_selected_coords, *_ = batched_nms(
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
                selected_coords_ = denormalize_coordinates(
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

    return selected_boxes, selected_keypoints


def detect_landmarks(
    rgb_frame: np.ndarray,
    roi_4corners: np.ndarray,
    landmark_detector: Interpreter,
    landmark_input: list[dict[str, Any]],
    landmark_output: list[dict[str, Any]],
) -> tuple[np.ndarray, list[bool]]:
    """Crop to ROI, run landmark detection, and map coords back to the original frame.

    Parameters
    ----------
    rgb_frame
        Input RGB image as a numpy array.
    roi_4corners
        Four corner coordinates of the region of interest, shape [4, 2].
    landmark_detector
        TFLite interpreter for landmark detection.
    landmark_input
        Input tensor details from landmark_detector.get_input_details().
    landmark_output
        Output tensor details from landmark_detector.get_output_details().

    Returns
    -------
    tuple[np.ndarray, list[bool]]
        Stacked (N, 21, 3) landmark array (or empty) and handedness flags per hand.
    """
    affines = compute_box_affine_crop_resize_matrix(
        roi_4corners[np.newaxis, :, :3], (224, 224)
    )
    keypoint_net_inputs = apply_batched_affines_to_frame(
        rgb_frame, affines, (224, 224)
    ).astype(np.uint8, copy=False)

    landmark_detector.set_tensor(landmark_input[0]["index"], keypoint_net_inputs)
    landmark_detector.invoke()

    landmarks = dequantize(
        landmark_detector.get_tensor(landmark_output[0]["index"]),
        zero_points=landmark_output[0]["quantization_parameters"]["zero_points"],
        scales=landmark_output[0]["quantization_parameters"]["scales"],
    ).reshape(1, 21, 3)

    ld_scores = dequantize(
        landmark_detector.get_tensor(landmark_output[1]["index"]),
        zero_points=landmark_output[1]["quantization_parameters"]["zero_points"],
        scales=landmark_output[1]["quantization_parameters"]["scales"],
    )
    lr = dequantize(
        landmark_detector.get_tensor(landmark_output[2]["index"]),
        zero_points=landmark_output[2]["quantization_parameters"]["zero_points"],
        scales=landmark_output[2]["quantization_parameters"]["scales"],
    )

    all_landmarks = []
    is_right_hand = []
    for ld_batch_idx in range(landmarks.shape[0]):
        if ld_scores[ld_batch_idx] >= C.MIN_LANDMARK_SCORE:
            inverted_affine = cv2.invertAffineTransform(affines[ld_batch_idx]).astype(
                np.float32
            )
            landmarks[ld_batch_idx][:, :2] = apply_affine_to_coordinates(
                landmarks[ld_batch_idx][:, :2], inverted_affine
            )
            all_landmarks.append(landmarks[ld_batch_idx])
            is_right_hand.append(np.round(lr[ld_batch_idx]).item() == 1)

    if all_landmarks:
        return np.stack(all_landmarks, axis=0), is_right_hand
    return np.empty(0, dtype=np.float32), is_right_hand


def classify_gesture(
    landmarks: np.ndarray,
    lr_hand: np.floating[Any],
    gesture_classifier: Interpreter,
    classifier_input: list[dict[str, Any]],
    classifier_output: list[dict[str, Any]],
) -> str:
    """Run gesture classification for a single detected hand.

    Parameters
    ----------
    landmarks
        Landmark array of shape (1, 21, 3) for one hand.
    lr_hand
        Scalar handedness value (0=left, 1=right).
    gesture_classifier
        TFLite interpreter for gesture classification.
    classifier_input
        Input tensor details from gesture_classifier.get_input_details().
    classifier_output
        Output tensor details from gesture_classifier.get_output_details().

    Returns
    -------
    str
        Gesture label string (e.g. "Thumb_Up").
    """
    hand = np.expand_dims(landmarks, axis=0)
    lr_expanded = np.expand_dims(lr_hand, axis=0)

    x64_a = preprocess_hand_x64(hand, lr_expanded, mirror=False)
    x64_b = preprocess_hand_x64(hand, lr_expanded, mirror=True)

    x64_a = quantize(
        x64_a,
        zero_points=classifier_input[0]["quantization_parameters"]["zero_points"],
        scales=classifier_input[0]["quantization_parameters"]["scales"],
    )
    x64_b = quantize(
        x64_b,
        zero_points=classifier_input[1]["quantization_parameters"]["zero_points"],
        scales=classifier_input[1]["quantization_parameters"]["scales"],
    )

    gesture_classifier.set_tensor(classifier_input[0]["index"], x64_a)
    gesture_classifier.set_tensor(classifier_input[1]["index"], x64_b)
    gesture_classifier.invoke()

    score = dequantize(
        gesture_classifier.get_tensor(classifier_output[0]["index"]),
        zero_points=classifier_output[0]["quantization_parameters"]["zero_points"],
        scales=classifier_output[0]["quantization_parameters"]["scales"],
    )

    gesture_id = np.argmax(score.flatten())
    return C.GESTURE_LABELS[gesture_id]


def run_inference(
    rgb_frame: np.ndarray,
    hand_detector: Interpreter,
    detector_input: list[dict[str, Any]],
    detector_output: list[dict[str, Any]],
    landmark_detector: Interpreter,
    landmark_input: list[dict[str, Any]],
    landmark_output: list[dict[str, Any]],
    gesture_classifier: Interpreter,
    classifier_input: list[dict[str, Any]],
    classifier_output: list[dict[str, Any]],
) -> tuple[list[np.ndarray], list[list[bool]], list[list[str]]]:
    """Run the full three-stage inference pipeline on a single RGB frame.

    Parameters
    ----------
    rgb_frame
        Input RGB image as a numpy array.
    hand_detector
        TFLite interpreter for palm detection.
    detector_input
        Input tensor details from hand_detector.get_input_details().
    detector_output
        Output tensor details from hand_detector.get_output_details().
    landmark_detector
        TFLite interpreter for landmark detection.
    landmark_input
        Input tensor details from landmark_detector.get_input_details().
    landmark_output
        Output tensor details from landmark_detector.get_output_details().
    gesture_classifier
        TFLite interpreter for gesture classification.
    classifier_input
        Input tensor details from gesture_classifier.get_input_details().
    classifier_output
        Output tensor details from gesture_classifier.get_output_details().

    Returns
    -------
    tuple[list[np.ndarray], list[list[bool]], list[list[str]]]
        batched_landmarks, batched_is_right_hand, batched_gesture_labels per batch item.
    """
    selected_boxes, selected_keypoints = detect_hands(
        rgb_frame, hand_detector, detector_input, detector_output
    )

    batched_roi_4corners = compute_object_roi(selected_boxes, selected_keypoints)
    batched_roi_4corners = np.unstack(batched_roi_4corners[0])

    batched_landmarks: list[np.ndarray] = []
    batched_is_right_hand: list[list[bool]] = []
    batched_gesture_labels: list[list[str]] = []

    for roi_4corners in batched_roi_4corners:
        if roi_4corners.size == 0:
            continue

        landmarks, is_right_hand = detect_landmarks(
            rgb_frame, roi_4corners, landmark_detector, landmark_input, landmark_output
        )

        gesture_labels = []
        # Re-run lr dequantization for per-hand classify — use stored lr from detect_landmarks
        # by iterating over the stacked landmarks directly.
        if landmarks.size > 0:
            # landmarks shape: (N, 21, 3); is_right_hand length: N
            # We need the raw lr values; detect_landmarks already stored bool flags.
            # Reconstruct scalar lr values from the bool flags for the classifier.
            for hand_idx in range(landmarks.shape[0]):
                lr_scalar = np.float32(1.0 if is_right_hand[hand_idx] else 0.0)
                label = classify_gesture(
                    landmarks[hand_idx],
                    lr_scalar,
                    gesture_classifier,
                    classifier_input,
                    classifier_output,
                )
                gesture_labels.append(label)

        batched_landmarks.append(landmarks)
        batched_is_right_hand.append(is_right_hand)
        batched_gesture_labels.append(gesture_labels)

    # Append empty sentinel for the trailing batch slot (mirrors original behaviour)
    batched_landmarks.append(np.empty(0, dtype=np.float32))
    batched_is_right_hand.append([])
    batched_gesture_labels.append([])

    return batched_landmarks, batched_is_right_hand, batched_gesture_labels


def main(args: argparse.Namespace) -> None:
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

    delegate_path = (
        args.qairt_path / "lib" / "aarch64-oe-linux-gcc11.2" / "libQnnTFLiteDelegate.so"
    )
    delegate = Delegate(
        delegate_path,
        {
            "backend_type": "htp",
            "htp_performance_mode": "2",
            "library_path": str(
                args.qairt_path / "lib" / "aarch64-oe-linux-gcc11.2" / "libQnnHtp.so"
            ),
            "skel_library_dir": str(
                args.qairt_path / "lib" / f"hexagon-{args.hexagon_version}" / "unsigned"
            ),
        },
    )

    hand_detector = Interpreter(
        "models/palm_detector.tflite", experimental_delegates=[delegate]
    )
    landmark_detector = Interpreter(
        "models/hand_landmark_detector.tflite", experimental_delegates=[delegate]
    )

    gesture_classifier = Interpreter("models/canned_gesture_classifier.tflite")

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
    fps_counter = FpsCounter()

    warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

    print(
        "--------------------------- Web server ----------------------------",
        flush=True,
    )
    try:
        ui.start_thread()
        while True:
            rgb_frame = outq.get(timeout=5)

            landmarks, is_right, gestures = run_inference(
                rgb_frame,
                hand_detector,
                detector_input,
                detector_output,
                landmark_detector,
                landmark_input,
                landmark_output,
                gesture_classifier,
                classifier_input,
                classifier_output,
            )

            draw_predictions(
                [rgb_frame] * len(landmarks),
                landmarks,
                is_right,
                gestures,
                landmark_connections=C.HAND_LANDMARK_CONNECTIONS,
            )

            fps_counter.tick()

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
    parser.add_argument(
        "--qairt-path",
        type=Path,
        required=True,
        help="Path to QAIRT SDK root",
    )
    parser.add_argument(
        "--hexagon-version",
        type=str,
        default="v73",
        help="Hexagon version of the device, e.g. v73, default v73",
    )

    args = parser.parse_args()
    main(args)
