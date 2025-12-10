from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np
import cv2
import math

Pixel = Tuple[int, int, int]
Coord = Tuple[int, int]


@dataclass
class ConnectedComponent:
    """Puzzle piece extracted via connected components."""
    cells: List[Coord] = field(default_factory=list)

    # Filled after finalize()
    min_r: int = None
    max_r: int = None
    min_c: int = None
    max_c: int = None

    bounding_patch: np.ndarray = None
    bounding_mask: np.ndarray = None
    patch: np.ndarray = None
    mask: np.ndarray = None

    # -------------------------------
    def finalize(self, pixels, strip_size: int = 10):
        self._compute_bounding_box()
        self._extract_patch_and_mask(pixels)
        self._unrotate_patch()

    # -------------------------------
    def _compute_bounding_box(self):
        rows = [r for r, _ in self.cells]
        cols = [c for _, c in self.cells]

        self.min_r = min(rows)
        self.max_r = max(rows)
        self.min_c = min(cols)
        self.max_c = max(cols)

    @property
    def height(self):
        return self.max_r - self.min_r + 1

    @property
    def width(self):
        return self.max_c - self.min_c + 1

    # -------------------------------
    def _extract_patch_and_mask(self, pixels):
        h, w = self.height, self.width
        patch = np.zeros((h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)

        for r, c in self.cells:
            pr = r - self.min_r
            pc = c - self.min_c
            patch[pr, pc] = pixels[r][c]
            mask[pr, pc] = 255

        self.bounding_patch = patch
        self.bounding_mask = mask

    def _unrotate_patch(self):
        mask = self.bounding_mask
        patch = self.bounding_patch

        ys, xs = np.where(mask > 0)
        pts = np.column_stack([xs, ys]).astype(np.float32)

        if pts.shape[0] == 0:
            self.patch = patch.copy()
            self.mask = mask.copy()
            return

        # 1. Fit minimal bounding box
        (cx, cy), (size_w, size_h), angle = cv2.minAreaRect(pts)
        # angle is in range [-90, 0)

        # 2. Normalize rotation angle
        # -----------------------------------
        # If angle < -45, the rectangle is taller than wide → rotate 90°
        if angle < -45:
            angle_to_rotate = angle + 90
        else:
            angle_to_rotate = angle
        # -----------------------------------

        # 3. Build rotation matrix
        M = cv2.getRotationMatrix2D((cx, cy), angle_to_rotate, 1.0)

        H, W = mask.shape

        # 4. Rotate patch and mask
        rot_patch = cv2.warpAffine(patch, M, (W, H), flags=cv2.INTER_LINEAR)
        rot_mask = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST)

        # 5. Crop rotated patch
        ys2, xs2 = np.where(rot_mask > 0)
        miny, maxy = ys2.min(), ys2.max()
        minx, maxx = xs2.min(), xs2.max()

        self.patch = rot_patch[miny:maxy + 1, minx:maxx + 1]
        self.mask = rot_mask[miny:maxy + 1, minx:maxx + 1]
