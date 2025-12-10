import numpy as np
import cv2
import time


def animate_island_merging(mask, pixels, steps=50, delay=20):
    """
    One-window animation using OpenCV.
    delay = milliseconds per frame (default 20ms = 50 FPS)
    """

    H = len(mask)
    W = len(mask[0])

    pix = np.array(pixels, dtype=np.uint8)

    # --------------------------
    # Extract connected islands
    # --------------------------
    visited = [[False] * W for _ in range(H)]
    islands = []

    def bfs(sr, sc):
        q = [(sr, sc)]
        comp = []
        visited[sr][sc] = True

        for r, c in q:
            comp.append((r, c))
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if mask[nr][nc] and not visited[nr][nc]:
                        visited[nr][nc] = True
                        q.append((nr, nc))
        return comp

    for r in range(H):
        for c in range(W):
            if mask[r][c] and not visited[r][c]:
                islands.append(bfs(r, c))

    # --------------------------
    # Compute centroids
    # --------------------------
    centroids = []
    for isl in islands:
        cr = sum(r for r, c in isl) / len(isl)
        cc = sum(c for r, c in isl) / len(isl)
        centroids.append((cr, cc))

    # target = center
    target_r, target_c = H / 2, W / 2

    # --------------------------
    # Create persistent canvas
    # --------------------------
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    cv2.namedWindow("animation", cv2.WINDOW_NORMAL)

    # --------------------------
    # Animate
    # --------------------------
    for step in range(steps):
        t = step / (steps - 1)

        # Clear frame (white)
        frame = canvas.copy()

        for isl, (cr, cc) in zip(islands, centroids):

            new_cr = cr + (target_r - cr) * t
            new_cc = cc + (target_c - cc) * t

            dr = new_cr - cr
            dc = new_cc - cc

            for (r, c) in isl:
                nr = int(r + dr)
                nc = int(c + dc)

                if 0 <= nr < H and 0 <= nc < W:
                    frame[nr, nc] = pix[r, c]

        cv2.imshow("animation", frame)

        key = cv2.waitKey(delay)  # delay in ms
        if key == 27:  # ESC to stop
            break

    cv2.destroyAllWindows()