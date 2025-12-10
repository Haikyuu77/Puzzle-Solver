from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np

from old_approach.edge_matcher_utility import strip_combined_score


@dataclass
class StripMatcher:
    """
    Strip-based matching over all components.

    For each component and each side ('top', 'bottom', 'left', 'right'),
    finds the best matching side of any other component using a combined
    (color + gradient) score on thick edge strips.
    """
    components: List
    strip_size: int = 6
    w_color: float = 1.0
    w_grad: float = 0.25

    def match_all(self) -> Dict[int, Dict[str, Dict[str, float | int | str]]]:
        """
        Returns:
          {
            comp_i: {
              'top': {
                  'j': best_match_index,
                  'side': best_match_side,
                  'score': total_score,
                  'color_mse': c_mse,
                  'grad_mse': g_mse,
              },
              'bottom': { ... },
              'left': { ... },
              'right': { ... },
            },
            ...
          }
        """

        sides = ["top", "bottom", "left", "right"]
        N = len(self.components)
        results: Dict[int, Dict[str, Dict[str, float | int | str]]] = {
            i: {} for i in range(N)
        }

        # Precompute strips for every component/side
        strip_map: Dict[int, Dict[str, np.ndarray]] = {
            i: {side: self.components[i].get_edge_strip(side, self.strip_size)
                for side in sides}
            for i in range(N)
        }

        # Full pairwise comparison
        for i in range(N):
            for sideA in sides:
                stripA = strip_map[i][sideA]

                best_j = None
                best_sideB = None
                best_score = float("inf")
                best_c_mse = float("inf")
                best_g_mse = float("inf")

                for j in range(N):
                    if j == i:
                        continue

                    for sideB in sides:
                        stripB = strip_map[j][sideB]

                        total, c_mse, g_mse = strip_combined_score(
                            stripA, stripB,
                            w_color=self.w_color,
                            w_grad=self.w_grad,
                        )

                        if total < best_score:
                            best_score = total
                            best_j = j
                            best_sideB = sideB
                            best_c_mse = c_mse
                            best_g_mse = g_mse

                results[i][sideA] = {
                    "j": best_j,
                    "side": best_sideB,
                    "score": best_score,
                    "color_mse": best_c_mse,
                    "grad_mse": best_g_mse,
                }

        return results

    # Pretty printer
    def print_matches(self, matches: Dict[int, Dict[str, Dict[str, float | int | str]]]):
        sides = ["top", "bottom", "left", "right"]
        for i in range(len(matches)):
            print(f"Component {i}:")
            for side in sides:
                m = matches[i][side]
                print(
                    f"  {side:<6} → component {m['j']}, side '{m['side']}', "
                    f"score={m['score']:.2f}, "
                    f"color={m['color_mse']:.2f}, grad={m['grad_mse']:.2f}"
                )
            print()