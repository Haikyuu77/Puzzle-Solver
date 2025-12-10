import argparse
import os
import cv2

# Import the helper from the local module (lowercase filename).
from old_approach.ImageDetect import image_to_2d_array
from old_approach.algorithm_utils import ImageAlgorithms
from old_approach.display_utils import DisplayUtils
from old_approach.edge_matcher import StripMatcher
from test import solve_puzzle



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read an image file and display its pixels as a 2D array.",
    )
    parser.add_argument(
        "image_path",
        help="Path to the image file (e.g., PNG, JPG).",
    )
    parser.add_argument(
        "--save-components",
        dest="save_dir",
        help="Optional directory to save images of each detected component.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip the component viewer (enabled by default).",
    )
    parser.add_argument(
        "--save-connected",
        dest="save_connected",
        help="Optional file path to save a connected-chain visualization.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        raise SystemExit(f"Image not found: {args.image_path}")

    pixels_2d = image_to_2d_array(args.image_path)
    mask = ImageAlgorithms.flood_fill(pixels_2d)

    true_count = sum(value for row in mask for value in row)
    print(f"Canvas (True) pixels: {true_count}")

    components = ImageAlgorithms.find_components(mask, pixels_2d)

    grid, result_image, score = solve_puzzle(
        components,
        rows=4,
        cols=4,
        time_limit=60
    )

    # Save or display result
    cv2.imwrite('solved_puzzle.png', result_image)

    # The grid array tells you which piece goes where
    print(f"Piece arrangement:\n{grid}")


if __name__ == "__main__":
    main()
