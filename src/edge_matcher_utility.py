import math
from typing import List, Tuple

Pixel = Tuple[int, int, int]

def _edge_score(
    a: List[Pixel], b: List[Pixel], weights: Tuple[float, float, float]
) -> float:
    """Weighted score combining MSE, gradient difference, and histogram diff."""
    mse_w, grad_w, hist_w = weights
    a_adj, b_adj = _align_edges(a, b)
    mse_val = _mse(a_adj, b_adj)
    grad_val = _gradient_diff(a_adj, b_adj)
    hist_val = _hist_diff(a_adj, b_adj)
    return mse_w * mse_val + grad_w * grad_val + hist_w * hist_val


def _align_edges(a: List[Pixel], b: List[Pixel]) -> Tuple[List[Pixel], List[Pixel]]:
    """Align edge vectors by resampling the longer one down to the shorter."""
    if not a or not b:
        return a, b
    if len(a) == len(b):
        return a, b
    target_len = min(len(a), len(b))
    return _resample(a, target_len), _resample(b, target_len)


def _resample(edge: List[Pixel], target_len: int) -> List[Pixel]:
    if len(edge) == target_len or target_len == 0:
        return edge[:target_len]
    resampled: List[Pixel] = []
    for i in range(target_len):
        src_pos = i * (len(edge) - 1) / max(target_len - 1, 1)
        lower = int(math.floor(src_pos))
        upper = min(lower + 1, len(edge) - 1)
        t = src_pos - lower
        interp = tuple(
            int((1 - t) * edge[lower][c] + t * edge[upper][c]) for c in range(3)
        )
        resampled.append(interp)
    return resampled

def _mse(a: List[Pixel], b: List[Pixel]) -> float:
    if not a or not b:
        return float("inf")
    total = 0.0
    count = 0
    for pa, pb in zip(a, b):
        for ca, cb in zip(pa, pb):
            diff = ca - cb
            total += diff * diff
            count += 1
    return total / count if count else float("inf")

def _gradient_diff(a: List[Pixel], b: List[Pixel]) -> float:
    """Compare gradients along the edge."""
    if len(a) < 2 or len(b) < 2:
        return float("inf")
    total = 0.0
    count = 0
    for i in range(1, len(a)):
        ga = [a[i][c] - a[i - 1][c] for c in range(3)]
        gb = [b[i][c] - b[i - 1][c] for c in range(3)]
        for da, db in zip(ga, gb):
            diff = da - db
            total += diff * diff
            count += 1
    return total / count if count else float("inf")


def _hist_diff(a: List[Pixel], b: List[Pixel], bins: int = 16) -> float:
    """Simple per-channel histogram L1 difference."""
    if not a or not b:
        return float("inf")
    def hist(edge: List[Pixel]) -> List[List[int]]:
        h = [[0] * bins for _ in range(3)]
        for pix in edge:
            for c in range(3):
                idx = min(bins - 1, pix[c] * bins // 256)
                h[c][idx] += 1
        return h
    ha = hist(a)
    hb = hist(b)
    diff = 0
    for ca, cb in zip(ha, hb):
        for ba, bb in zip(ca, cb):
            diff += abs(ba - bb)
    # Normalize by total pixels to keep scale manageable.
    total = len(a)
    return diff / total if total else float("inf")