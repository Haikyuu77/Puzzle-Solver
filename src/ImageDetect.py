from typing import List, Tuple

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "Pillow is required to read images. Install with `pip install Pillow`."
    ) from exc

Pixel = Tuple[int, int, int]

def image_to_2d_array(image_path: str) -> List[List[Pixel]]:
    """Load an image and return a 2D list of RGB pixel tuples."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")   # <-- IMPORTANT: forces 3-channel RGB
        width, height = img.size

        print("width:", width, "height:", height)

        pixels = list(img.getdata())  # now guaranteed to be (R,G,B)
        print("total flattened pixels:", len(pixels))

        return [
            pixels[row * width : (row + 1) * width]
            for row in range(height)
        ]
