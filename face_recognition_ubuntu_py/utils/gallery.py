# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np

import utils.constants as C
from utils.detection import crop_face, detect_faces, largest_box
from utils.input_processing import load_image_rgb
from utils.model_io_processing import build_prototype

# Image file extensions treated as enrollment photos.
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def _warn(message: str) -> None:
    """Emit a warning without interrupting enrollment.

    Each message embeds the offending image path, so it is unique and the
    default warning filter will not collapse per-image warnings into one.
    """
    warnings.warn(message, stacklevel=2)


def _iter_identity_images(gallery_dir: Path) -> Iterator[tuple[str, Path]]:
    """
    Yield (identity_name, image_path) pairs from a gallery directory.

    Two layouts are supported and may be mixed:
      - ``<gallery>/<Name>/*.jpg`` — the subdirectory name is the identity, and
        every image inside contributes to that identity.
      - ``<gallery>/<Name>.jpg`` — a top-level image whose filename stem is the
        identity.

    Hidden entries and non-image top-level files are skipped.

    Parameters
    ----------
    gallery_dir
        Root directory of the gallery.

    Yields
    ------
    tuple[str, Path]
        The identity name and the path to one of its enrollment images.
    """
    for entry in sorted(gallery_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            for image_path in sorted(entry.iterdir()):
                if (
                    image_path.is_file()
                    and not image_path.name.startswith(".")
                    and image_path.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    yield entry.name, image_path
        elif entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            yield entry.stem, entry


def build_gallery(
    gallery_dir: str,
    embed_fn: Callable[[np.ndarray], np.ndarray],
    *,
    cascade: cv2.CascadeClassifier,
    margin: float = C.BOX_MARGIN,
) -> dict[str, np.ndarray]:
    """
    Enroll identities from a gallery directory into prototype embeddings.

    For every enrollment image the largest detected face is cropped and embedded;
    if no face is detected the whole image is embedded (so tightly pre-cropped
    faces still enroll). All embeddings for one identity are combined into a
    single unit-length prototype via :func:`build_prototype`.

    Parameters
    ----------
    gallery_dir
        Root directory of the gallery (see :func:`_iter_identity_images` for the
        supported layouts).
    embed_fn
        Callable mapping an RGB face crop to its embedding of shape [D].
    cascade
        Loaded Haar cascade used to locate faces in enrollment images.
    margin
        Fraction by which each detected face box is expanded before cropping.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from identity name to its unit-length prototype embedding.
    """
    root = Path(gallery_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Gallery directory not found: {gallery_dir}")

    embeddings_by_name: dict[str, list[np.ndarray]] = defaultdict(list)
    for name, image_path in _iter_identity_images(root):
        try:
            rgb = load_image_rgb(str(image_path))
        except FileNotFoundError:
            _warn(f"could not read {image_path}; skipping")
            continue

        boxes = detect_faces(rgb, cascade)
        if boxes:
            if len(boxes) > 1:
                _warn(
                    f"{image_path} has {len(boxes)} faces; using the largest. "
                    "Enrollment images should contain a single face."
                )
            crop = crop_face(rgb, largest_box(boxes), margin)
        else:
            _warn(f"no face detected in {image_path}; embedding the whole image")
            crop = rgb

        embeddings_by_name[name].append(embed_fn(crop))

    gallery = {
        name: build_prototype(embeddings)
        for name, embeddings in embeddings_by_name.items()
        if embeddings
    }

    if not gallery:
        raise ValueError(
            f"No usable faces found in gallery '{gallery_dir}'. "
            "Add at least one identity image."
        )

    print(
        f"Enrolled {len(gallery)} identit"
        f"{'y' if len(gallery) == 1 else 'ies'}: {', '.join(sorted(gallery))}",
        flush=True,
    )
    return gallery
