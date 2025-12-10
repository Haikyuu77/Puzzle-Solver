from collections import deque
from typing import Dict, List, Sequence, Tuple

from connected_component import ConnectedComponent, Coord

Pixel = Tuple[int, int, int]
DEFAULT_THRESHOLD = 10

def clamp(x):
    return max(0, min(255, int(x)))

class ImageAlgorithms:
    """Utility algorithms for background detection and component grouping."""

    @staticmethod
    def flood_fill(
        pixels: List[List[Pixel]], threshold: int = DEFAULT_THRESHOLD
    ) -> List[List[bool]]:
        """
        Mark canvas/background as True and puzzle pieces as False using flood fill.

        Assumes the canvas colour is represented by the border pixels. Any pixel
        within `threshold` distance (per channel) of that colour is considered
        background and will be filled from the borders inward.
        """
        if not pixels or not pixels[0]:
            return []

        height = len(pixels)
        width = len(pixels[0])

        # Estimate canvas colour using the border (assumed to be background).
        border_pixels = (
            pixels[0]
            + pixels[-1]
            + [row[0] for row in pixels]
            + [row[-1] for row in pixels]
        )
        background_colour = tuple(
            clamp(sum(channel) / len(border_pixels))
            for channel in zip(*border_pixels)
        )

        visited = [[False] * width for _ in range(height)]
        canvas_mask = [[False] * width for _ in range(height)]
        que = deque()
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def is_under_threshold(pixel: Pixel, reference: Pixel) -> bool:
            return all(abs(p - r) <= threshold for p, r in zip(pixel, reference))

        def try_enqueue(r: int, c: int) -> None:
            if visited[r][c]:
                return
            visited[r][c] = True
            if is_under_threshold(pixels[r][c], background_colour):
                canvas_mask[r][c] = True
                que.append((r, c))

        # Seed queue with background-looking border pixels.
        for x in range(width):
            try_enqueue(0, x)
            try_enqueue(height - 1, x)
        for y in range(height):
            try_enqueue(y, 0)
            try_enqueue(y, width - 1)

        # Standard BFS flood fill to mark connected background.
        while que:
            r, c = que.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    try_enqueue(nr, nc)

        return canvas_mask

    @staticmethod
    def find_components(mask: List[List[bool]], pixels: List[List[Pixel]]) -> List[ConnectedComponent]:
        """Find connected False regions (puzzle pieces) using BFS."""
        if not mask or not mask[0]:
            return []

        height = len(mask)
        width = len(mask[0])
        visited = [[False] * width for _ in range(height)]
        components = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for r in range(height):
            for c in range(width):
                if mask[r][c] or visited[r][c]:
                    continue

                queue = deque([(r, c)])
                visited[r][c] = True
                cells = []

                while queue:
                    cr, cc = queue.popleft()
                    cells.append((cr, cc))
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < height
                            and 0 <= nc < width
                            and not visited[nr][nc]
                            and not mask[nr][nc]
                        ):
                            visited[nr][nc] = True
                            queue.append((nr, nc))

                comp = ConnectedComponent(cells=cells)
                comp.finalize(pixels)
                components.append(comp)

        return components
