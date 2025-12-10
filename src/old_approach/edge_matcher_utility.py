import numpy as np
import cv2
from typing import Tuple

def _resize_to_common(stripA: np.ndarray, stripB: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize two strip images to a common (h,w) = (min_h, min_w).
    Keeps aspect reasonably consistent and simplifies comparison.
    """
    if stripA is None or stripB is None or stripA.size == 0 or stripB.size == 0:
        return None, None

    hA, wA = stripA.shape[:2]
    hB, wB = stripB.shape[:2]

    target_h = max(1, min(hA, hB))
    target_w = max(1, min(wA, wB))

    if (hA, wA) != (target_h, target_w):
        stripA_resized = cv2.resize(stripA, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:
        stripA_resized = stripA

    if (hB, wB) != (target_h, target_w):
        stripB_resized = cv2.resize(stripB, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:
        stripB_resized = stripB

    return stripA_resized, stripB_resized


def strip_color_mse(stripA: np.ndarray, stripB: np.ndarray) -> float:
    """
    Color-based MSE on RGB, using only pixels where both strips are non-zero
    (i.e., both belong to actual piece, not padded background).
    """
    stripA, stripB = _resize_to_common(stripA, stripB)
    if stripA is None or stripB is None:
        return 1e9

    A = stripA.astype(np.float32)
    B = stripB.astype(np.float32)

    # valid pixels: at least one channel non-zero
    validA = np.any(A > 0, axis=2)
    validB = np.any(B > 0, axis=2)
    valid = validA & validB

    if not np.any(valid):
        return 1e9

    diff = A[valid] - B[valid]
    return float(np.mean(diff ** 2))


def strip_gradient_mse(stripA: np.ndarray, stripB: np.ndarray) -> float:
    """
    Gradient-based similarity: compare edge / texture structure.
    Uses Sobel magnitude on grayscale strips, again only on overlapping non-zero pixels.
    """
    stripA, stripB = _resize_to_common(stripA, stripB)
    if stripA is None or stripB is None:
        return 1e9

    # Convert to grayscale
    grayA = cv2.cvtColor(stripA, cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(stripB, cv2.COLOR_RGB2GRAY)

    # Sobel gradients
    gxA = cv2.Sobel(grayA, cv2.CV_32F, 1, 0, ksize=3)
    gyA = cv2.Sobel(grayA, cv2.CV_32F, 0, 1, ksize=3)
    magA = cv2.magnitude(gxA, gyA)

    gxB = cv2.Sobel(grayB, cv2.CV_32F, 1, 0, ksize=3)
    gyB = cv2.Sobel(grayB, cv2.CV_32F, 0, 1, ksize=3)
    magB = cv2.magnitude(gxB, gyB)

    # valid pixels (avoid background)
    validA = grayA > 0
    validB = grayB > 0
    valid = validA & validB

    if not np.any(valid):
        return 1e9

    diff = magA[valid] - magB[valid]
    return float(np.mean(diff ** 2))


def strip_combined_score(stripA: np.ndarray,
                         stripB: np.ndarray,
                         w_color: float = 1.0,
                         w_grad: float = 0.25) -> Tuple[float, float, float]:
    """
    Combine color MSE + gradient MSE into a single score.
    Lower score = better match.
    Returns (total_score, color_mse, grad_mse).
    """
    c_mse = strip_color_mse(stripA, stripB)
    g_mse = strip_gradient_mse(stripA, stripB)

    total = w_color * c_mse + w_grad * g_mse
    return total, c_mse, g_mse