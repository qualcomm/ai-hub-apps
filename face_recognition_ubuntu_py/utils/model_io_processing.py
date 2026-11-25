# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np

import utils.constants as C


def l2_normalize(embedding: np.ndarray) -> np.ndarray:
    """
    Scale an embedding to unit L2 length.

    The exported CavaFace model already emits L2-normalized embeddings; this is
    applied defensively so cosine similarity stays a plain dot product even if a
    variant of the model returns un-normalized features.

    Parameters
    ----------
    embedding
        Face embedding of shape [D], dtype float32.

    Returns
    -------
    np.ndarray
        Unit-length embedding of shape [D], dtype float32.
    """
    norm = np.linalg.norm(embedding) + 1e-9
    return (embedding / norm).astype(np.float32)


def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """
    Cosine similarity between two face embeddings.

    Parameters
    ----------
    embedding_a
        First face embedding of shape [D].
    embedding_b
        Second face embedding of shape [D].

    Returns
    -------
    float
        Cosine similarity in [-1, 1]; higher means more similar faces.
    """
    a = l2_normalize(embedding_a)
    b = l2_normalize(embedding_b)
    return float(np.dot(a, b))


def build_prototype(embeddings: list[np.ndarray]) -> np.ndarray:
    """
    Combine one or more face embeddings into a single unit-length prototype.

    The prototype is the mean of the L2-normalized embeddings, renormalized to
    unit length. Averaging several views of one identity yields a more robust
    template than any single embedding.

    Parameters
    ----------
    embeddings
        Non-empty list of face embeddings, each of shape [D].

    Returns
    -------
    np.ndarray
        Unit-length prototype embedding of shape [D], dtype float32.
    """
    if not embeddings:
        raise ValueError("Cannot build a prototype from zero embeddings")
    normalized = np.stack([l2_normalize(e) for e in embeddings], axis=0)
    return l2_normalize(normalized.mean(axis=0))


def identify(
    embedding: np.ndarray,
    gallery: dict[str, np.ndarray],
    threshold: float,
) -> tuple[str, float]:
    """
    Match a face embedding against a gallery of identity prototypes.

    Parameters
    ----------
    embedding
        Query face embedding of shape [D].
    gallery
        Mapping from identity name to that identity's prototype embedding of
        shape [D]. Must be non-empty.
    threshold
        Minimum cosine similarity for a match. If the best match scores below
        this, the face is reported as unknown.

    Returns
    -------
    tuple[str, float]
        The matched identity name (or ``constants.UNKNOWN_LABEL`` when the best
        score is below ``threshold``) and the best cosine similarity in [-1, 1].
    """
    if not gallery:
        raise ValueError("Cannot identify against an empty gallery")

    best_name = C.UNKNOWN_LABEL
    best_score = -1.0
    for name, prototype in gallery.items():
        score = cosine_similarity(embedding, prototype)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score < threshold:
        return C.UNKNOWN_LABEL, best_score
    return best_name, best_score
