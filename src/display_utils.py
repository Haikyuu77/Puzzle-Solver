import numpy as np
from typing import List, Tuple

from PIL import Image
from PIL.ImageDraw import ImageDraw
from PIL.ImageFont import ImageFont

from connected_component import ConnectedComponent


Pixel = Tuple[int, int, int]


class DisplayUtils:

    # ============================================================
    # MOSAIC GRID BUILDER (patches or generic images)
    # ============================================================
    @staticmethod
    def _mosaic(images, max_columns, padding, background_color):
        """Tile images into a grid."""
        if not images:
            return Image.new("RGB", (1, 1), background_color)

        rows, row = [], []
        for img in images:
            row.append(img)
            if len(row) == max_columns:
                rows.append(row)
                row = []
        if row:
            rows.append(row)

        row_heights = [max(im.height for im in r) for r in rows]
        row_widths = [
            sum(im.width for im in r) + padding * (len(r) - 1)
            for r in rows
        ]

        W = max(row_widths)
        H = sum(row_heights) + padding * (len(rows) - 1)

        canvas = Image.new("RGB", (W, H), background_color)
        y = 0

        for r_imgs, h in zip(rows, row_heights):
            x = 0
            for img in r_imgs:
                canvas.paste(img, (x, y))
                x += img.width + padding
            y += h + padding

        return canvas

    # ============================================================
    # DISPLAY CROPPED PATCHES OF COMPONENTS
    # ============================================================
    @staticmethod
    def display_component_patches(
            components: List[ConnectedComponent],
            max_columns: int = 4,
            padding: int = 10,
            background_color: Pixel = (255, 255, 255)):

        patch_images = []

        for comp in components:
            if comp.patch is None:
                continue
            img = Image.fromarray(comp.patch.astype("uint8"), "RGB")
            patch_images.append(img)

        if not patch_images:
            print("No component patches found.")
            return None

        mosaic = DisplayUtils._mosaic(
            patch_images, max_columns, padding, background_color
        )

        mosaic.show()
        return mosaic

    # ============================================================
    # STRIP → IMAGE Converter
    # ============================================================
    @staticmethod
    def strip_to_image(strip: np.ndarray,
                       target_height=40,
                       background_color=(255, 255, 255)) -> Image.Image:

        if strip is None or strip.size == 0:
            return Image.new("RGB", (40, target_height), background_color)

        # Handle flattened Nx3 strips -> convert to vertical 2D form
        if len(strip.shape) == 2 and strip.shape[1] == 3:
            strip = strip.reshape(strip.shape[0], 1, 3)

        img = Image.fromarray(strip.astype("uint8"), "RGB")

        # Normalize height
        scale = target_height / img.height
        new_w = max(1, int(img.width * scale))
        img = img.resize((new_w, target_height))

        return img

    # ============================================================
    # DISPLAY 2×2 STRIP GRID FOR ONE COMPONENT
    # ============================================================
    @staticmethod
    def display_component_strips(component: ConnectedComponent,
                                 strip_size=6,
                                 background_color=(255, 255, 255)) -> Image.Image:

        sides = ["top", "bottom", "left", "right"]

        strips = {
            s: component.get_edge_strip(s, strip_size)
            for s in sides
        }

        imgs = {s: DisplayUtils.strip_to_image(strips[s]) for s in sides}

        # Uniform cell sizes
        cell_w = max(img.width for img in imgs.values()) + 80
        cell_h = max(img.height for img in imgs.values()) + 40

        grid_w = cell_w * 2
        grid_h = cell_h * 2

        canvas = Image.new("RGB", (grid_w, grid_h), background_color)
        draw = ImageDraw(canvas)

        # Load font safely
        try:
            font = ImageFont.truetype("Arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        layout = {
            "top": (0, 0),
            "bottom": (cell_w, 0),
            "left": (0, cell_h),
            "right": (cell_w, cell_h),
        }

        for side, (x, y) in layout.items():
            draw.text((x + 5, y + 5), side.upper(), fill=(0, 0, 0), font=font)

            strip_img = imgs[side]
            canvas.paste(strip_img, (x + 5, y + 30))

        return canvas

    # ============================================================
    # DISPLAY STRIPS FOR ALL COMPONENTS
    # ============================================================
    @staticmethod
    def display_all_strips(components: List[ConnectedComponent],
                           strip_size=6,
                           max_columns=3,
                           padding=20,
                           background_color=(255, 255, 255)):

        grids = [
            DisplayUtils.display_component_strips(comp, strip_size, background_color)
            for comp in components
        ]

        # Build mosaic of the 2×2 strip grids
        rows, row = [], []
        for g in grids:
            row.append(g)
            if len(row) == max_columns:
                rows.append(row)
                row = []
        if row:
            rows.append(row)

        row_heights = [max(g.height for g in r) for r in rows]
        row_widths = [
            sum(g.width for g in r) + padding * (len(r) - 1)
            for r in rows
        ]

        W = max(row_widths)
        H = sum(row_heights) + padding * (len(rows) - 1)

        canvas = Image.new("RGB", (W, H), background_color)

        y = 0
        for r_imgs, h in zip(rows, row_heights):
            x = 0
            for g in r_imgs:
                canvas.paste(g, (x, y))
                x += g.width + padding
            y += h + padding

        canvas.show()
        return canvas