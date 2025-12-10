import numpy as np
import cv2
from typing import List, Tuple, Dict
from dataclasses import dataclass
import random
import time


@dataclass
class PuzzlePiece:
    """Represents a puzzle piece with its image data."""
    patch: np.ndarray
    mask: np.ndarray
    id: int


def rotate_piece(patch: np.ndarray, mask: np.ndarray, angle: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate piece by specified angle."""
    if angle == 0:
        return patch, mask
    k = angle // 90
    return np.rot90(patch, k), np.rot90(mask, k)


def extract_edge(patch: np.ndarray, mask: np.ndarray, edge: str, width: int = 10):
    """Extract edge strip from piece."""
    h, w = patch.shape[:2]
    if edge == 'top':
        return patch[:width, :].copy(), mask[:width, :].copy()
    elif edge == 'bottom':
        return patch[-width:, :].copy(), mask[-width:, :].copy()
    elif edge == 'left':
        return patch[:, :width].copy(), mask[:, :width].copy()
    elif edge == 'right':
        return patch[:, -width:].copy(), mask[:, -width:].copy()


def compute_edge_score(patch1: np.ndarray, mask1: np.ndarray,
                       patch2: np.ndarray, mask2: np.ndarray,
                       edge1: str, edge2: str) -> float:
    """Compute compatibility score between two edges.
    Lower scores are better (penalty-based system).
    """
    width = 15  # Increased strip width for better matching
    strip1, mask1_strip = extract_edge(patch1, mask1, edge1, width)
    strip2, mask2_strip = extract_edge(patch2, mask2, edge2, width)

    h1, w1 = strip1.shape[:2]
    h2, w2 = strip2.shape[:2]

    # Resize to same dimensions
    if edge1 in ['left', 'right']:
        target_h = max(h1, h2)  # Use max to preserve more info
        if target_h < 5:
            return float('inf')  # Impossible configuration
        strip1 = cv2.resize(strip1, (width, target_h), interpolation=cv2.INTER_LINEAR)
        strip2 = cv2.resize(strip2, (width, target_h), interpolation=cv2.INTER_LINEAR)
        mask1_strip = cv2.resize(mask1_strip, (width, target_h), interpolation=cv2.INTER_NEAREST)
        mask2_strip = cv2.resize(mask2_strip, (width, target_h), interpolation=cv2.INTER_NEAREST)
    else:
        target_w = max(w1, w2)  # Use max to preserve more info
        if target_w < 5:
            return float('inf')  # Impossible configuration
        strip1 = cv2.resize(strip1, (target_w, width), interpolation=cv2.INTER_LINEAR)
        strip2 = cv2.resize(strip2, (target_w, width), interpolation=cv2.INTER_LINEAR)
        mask1_strip = cv2.resize(mask1_strip, (target_w, width), interpolation=cv2.INTER_NEAREST)
        mask2_strip = cv2.resize(mask2_strip, (target_w, width), interpolation=cv2.INTER_NEAREST)

    # Get valid overlap pixels
    valid = (mask1_strip > 0) & (mask2_strip > 0)
    if not np.any(valid):
        return float('inf')  # Impossible configuration

    # Color difference (main metric)
    diff = strip1.astype(float) - strip2.astype(float)
    color_diff = np.sqrt(np.sum(diff ** 2, axis=2))
    color_score = np.percentile(color_diff[valid], 75)  # Use 75th percentile to ignore outliers

    # Also check edges more carefully
    if edge1 == 'right':
        edge1_pixels = strip1[:, -2:].reshape(-1, 3)
        edge2_pixels = strip2[:, :2].reshape(-1, 3)
    elif edge1 == 'left':
        edge1_pixels = strip1[:, :2].reshape(-1, 3)
        edge2_pixels = strip2[:, -2:].reshape(-1, 3)
    elif edge1 == 'bottom':
        edge1_pixels = strip1[-2:, :].reshape(-1, 3)
        edge2_pixels = strip2[:2, :].reshape(-1, 3)
    else:  # top
        edge1_pixels = strip1[:2, :].reshape(-1, 3)
        edge2_pixels = strip2[-2:, :].reshape(-1, 3)

    # Boundary compatibility
    boundary_diff = np.sqrt(np.sum((edge1_pixels.astype(float) - edge2_pixels.astype(float)) ** 2, axis=1))
    boundary_score = np.percentile(boundary_diff, 75)

    return color_score + 1.5 * boundary_score


def compute_global_coherence(pieces: List[PuzzlePiece], arrangement: List[Tuple[int, int]],
                             rows: int, cols: int) -> float:
    """
    Compute global image coherence score.
    Rewards arrangements that create visually coherent global patterns.
    """
    # Build the full image
    rotated_pieces = []
    max_h, max_w = 0, 0

    for piece_id, rotation in arrangement:
        patch, mask = rotate_piece(pieces[piece_id].patch, pieces[piece_id].mask, rotation)
        rotated_pieces.append(patch)
        max_h = max(max_h, patch.shape[0])
        max_w = max(max_w, patch.shape[1])

    # Create assembled image (low-res for speed)
    canvas_h = rows * max_h
    canvas_w = cols * max_w

    # Downsample for speed
    scale = min(1.0, 200.0 / max(canvas_h, canvas_w))
    canvas_h = int(canvas_h * scale)
    canvas_w = int(canvas_w * scale)

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)

    for idx, patch in enumerate(rotated_pieces):
        r = idx // cols
        c = idx % cols

        # Resize patch
        patch_h = int(max_h * scale)
        patch_w = int(max_w * scale)
        patch_resized = cv2.resize(patch, (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)

        y = r * patch_h
        x = c * patch_w

        canvas[y:y + patch_h, x:x + patch_w] = patch_resized

    # Analyze global properties
    scores = []

    # 1. Color gradient continuity (portraits should have smooth gradients)
    gray = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Penalize high discontinuities at piece boundaries
    discontinuity_score = 0.0
    for r in range(rows - 1):
        y = int((r + 1) * patch_h * scale)
        if y < canvas_h:
            boundary_grad = gradient_magnitude[max(0, y - 2):min(canvas_h, y + 2), :]
            discontinuity_score += np.mean(boundary_grad)

    for c in range(cols - 1):
        x = int((c + 1) * patch_w * scale)
        if x < canvas_w:
            boundary_grad = gradient_magnitude[:, max(0, x - 2):min(canvas_w, x + 2)]
            discontinuity_score += np.mean(boundary_grad)

    scores.append(discontinuity_score)

    # 2. Color distribution coherence
    # Convert to LAB for perceptual color
    lab = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_RGB2LAB)

    # Check if colors are distributed smoothly
    l_std = np.std(lab[:, :, 0])
    scores.append(-l_std * 0.1)  # Reward more uniform lightness

    # 3. Symmetry score (faces tend to have some symmetry)
    left_half = canvas[:, :canvas_w // 2]
    right_half = np.fliplr(canvas[:, canvas_w // 2:])
    min_w = min(left_half.shape[1], right_half.shape[1])
    if min_w > 0:
        symmetry_diff = np.mean(np.abs(left_half[:, :min_w] - right_half[:, :min_w]))
        scores.append(-symmetry_diff * 0.05)

    return sum(scores)


class PuzzleSolver:
    """Advanced puzzle solver with global coherence."""

    def __init__(self, pieces: List[PuzzlePiece], rows: int, cols: int):
        self.pieces = pieces
        self.rows = rows
        self.cols = cols
        self.n_pieces = len(pieces)
        self.rotations = [0, 90, 180, 270]
        self.compatibility_cache = {}
        self._precompute_compatibilities()

    def _precompute_compatibilities(self):
        """Precompute edge compatibility scores."""
        print("Precomputing edge compatibilities...")
        start_time = time.time()

        for i in range(self.n_pieces):
            for rot_i in self.rotations:
                patch_i, mask_i = rotate_piece(self.pieces[i].patch, self.pieces[i].mask, rot_i)

                for j in range(self.n_pieces):
                    if i == j:
                        continue

                    for rot_j in self.rotations:
                        patch_j, mask_j = rotate_piece(self.pieces[j].patch, self.pieces[j].mask, rot_j)

                        h_score = compute_edge_score(patch_i, mask_i, patch_j, mask_j, 'right', 'left')
                        self.compatibility_cache[(i, rot_i, j, rot_j, 'H')] = h_score

                        v_score = compute_edge_score(patch_i, mask_i, patch_j, mask_j, 'bottom', 'top')
                        self.compatibility_cache[(i, rot_i, j, rot_j, 'V')] = v_score

        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.1f} seconds\n")

    def evaluate_arrangement(self, arrangement: List[Tuple[int, int]],
                             use_global: bool = True, global_weight: float = 0.1) -> float:
        """Evaluate with both local edge matching and global coherence.
        
        Lower scores are better (penalty-based).
        Emphasizes local edge matching heavily.
        """
        # Local edge scores (primary objective)
        local_score = 0.0
        edge_count = 0

        for idx in range(len(arrangement)):
            r = idx // self.cols
            c = idx % self.cols
            piece_id, rotation = arrangement[idx]

            # Check right neighbor
            if c < self.cols - 1:
                right_idx = idx + 1
                right_id, right_rot = arrangement[right_idx]
                key = (piece_id, rotation, right_id, right_rot, 'H')
                if key in self.compatibility_cache:
                    score = self.compatibility_cache[key]
                    local_score += score
                    edge_count += 1

            # Check bottom neighbor
            if r < self.rows - 1:
                bottom_idx = idx + self.cols
                bottom_id, bottom_rot = arrangement[bottom_idx]
                key = (piece_id, rotation, bottom_id, bottom_rot, 'V')
                if key in self.compatibility_cache:
                    score = self.compatibility_cache[key]
                    local_score += score
                    edge_count += 1

        # Normalize local score
        local_score = local_score / max(edge_count, 1)

        # Global coherence (secondary objective - lower weight)
        if use_global:
            global_score = compute_global_coherence(self.pieces, arrangement, self.rows, self.cols)
            # Return weighted combination - emphasize local edges
            return local_score + global_weight * global_score

        return local_score

    def solve(self, time_limit: float = 115) -> Tuple[List[Tuple[int, int]], float]:
        """Solve using hybrid approach.
        
        CONSTRAINT: Each puzzle piece must be used exactly once in the arrangement.
        This is enforced through:
        1. Initial population generation ensures all pieces appear exactly once
        2. Mutation operations swap positions but don't change piece IDs
        3. Crossover results are validated; invalid children are discarded
        4. Assertions catch any constraint violations
        """

        def create_individual():
            """Create individual with all pieces used exactly once."""
            order = list(range(self.n_pieces))
            random.shuffle(order)
            rots = [random.choice(self.rotations) for _ in range(self.n_pieces)]
            return list(zip(order, rots))
        
        def is_valid_arrangement(individual):
            """Verify each piece is used exactly once."""
            piece_ids = [piece_id for piece_id, _ in individual]
            return len(piece_ids) == len(set(piece_ids)) and set(piece_ids) == set(range(self.n_pieces))

        def mutate(individual):
            """Mutate while preserving constraint: each piece used exactly once."""
            new_ind = individual.copy()
            mutation_type = random.choice(['swap', 'rotate', 'both', 'block_swap'])

            if mutation_type == 'swap':
                # Swap piece positions (preserves uniqueness)
                i, j = random.sample(range(len(individual)), 2)
                new_ind[i], new_ind[j] = new_ind[j], new_ind[i]
            elif mutation_type == 'rotate':
                # Only change rotation, not piece ID (preserves uniqueness)
                for _ in range(random.randint(1, 2)):
                    i = random.randint(0, len(individual) - 1)
                    piece_id, _ = new_ind[i]
                    new_ind[i] = (piece_id, random.choice(self.rotations))
            elif mutation_type == 'both':
                # Swap positions and change rotation (preserves uniqueness)
                i, j = random.sample(range(len(individual)), 2)
                new_ind[i], new_ind[j] = new_ind[j], new_ind[i]
                piece_id, _ = new_ind[i]
                new_ind[i] = (piece_id, random.choice(self.rotations))
            else:  # block_swap - swap entire rows or columns (preserves uniqueness)
                if random.random() < 0.5 and self.rows > 1:
                    r1, r2 = random.sample(range(self.rows), 2)
                    for c in range(self.cols):
                        idx1 = r1 * self.cols + c
                        idx2 = r2 * self.cols + c
                        new_ind[idx1], new_ind[idx2] = new_ind[idx2], new_ind[idx1]
                elif self.cols > 1:
                    c1, c2 = random.sample(range(self.cols), 2)
                    for r in range(self.rows):
                        idx1 = r * self.cols + c1
                        idx2 = r * self.cols + c2
                        new_ind[idx1], new_ind[idx2] = new_ind[idx2], new_ind[idx1]
            
            return new_ind

        print(f"Starting hybrid optimization (time limit: {time_limit}s)...\n")

        population_size = 80
        population = [create_individual() for _ in range(population_size)]
        
        # Validate initial population
        for ind in population:
            assert is_valid_arrangement(ind), "Initial population contains invalid arrangement!"

        best_ever = None
        best_ever_score = float('inf')

        start_time = time.time()
        generation = 0
        last_improvement = 0

        # Start with low global weight, keep it low to focus on edge matching
        global_weight = 0.05

        while (time.time() - start_time) < time_limit:
            # Keep global weight low throughout - focus on edge matching
            progress = (time.time() - start_time) / time_limit
            global_weight = 0.05 + 0.05 * progress  # Only go up to 0.10

            # Evaluate
            fitness = []
            for ind in population:
                score = self.evaluate_arrangement(ind, use_global=True, global_weight=global_weight)
                fitness.append((ind, score))

            fitness.sort(key=lambda x: x[1])

            if fitness[0][1] < best_ever_score:
                best_ever_score = fitness[0][1]
                best_ever = fitness[0][0]
                last_improvement = generation
                print(f"Gen {generation}: Score = {best_ever_score:.2f} (global_weight={global_weight:.2f}) ⭐")
            elif generation % 50 == 0:
                print(f"Gen {generation}: Best = {best_ever_score:.2f} (global_weight={global_weight:.2f})")

            # Diversify if stuck (but less aggressively)
            if generation - last_improvement > 300:
                print(f"  Diversifying population...")
                elite = [ind for ind, _ in fitness[:10]]
                population = elite + [create_individual() for _ in range(population_size - 10)]
                last_improvement = generation

            # New generation
            elite_size = 15
            elite = [ind for ind, _ in fitness[:elite_size]]
            new_pop = elite.copy()

            while len(new_pop) < population_size:
                if random.random() < 0.9:
                    p1 = random.choice(elite)
                    p2 = random.choice(fitness[:population_size // 2])[0]

                    # Order-based crossover (OX) - maintains piece uniqueness
                    # Select a crossover segment and combine parents
                    crossover_point1 = random.randint(0, len(p1) - 1)
                    crossover_point2 = random.randint(crossover_point1, len(p1) - 1)
                    
                    # Start with segment from p1
                    child = [None] * len(p1)
                    child[crossover_point1:crossover_point2 + 1] = p1[crossover_point1:crossover_point2 + 1]
                    
                    # Fill remaining positions with pieces from p2 in order
                    p2_idx = 0
                    used_piece_ids = set(piece_id for piece_id, _ in child[crossover_point1:crossover_point2 + 1])
                    
                    for i in range(len(child)):
                        if child[i] is None:
                            # Find next piece in p2 that hasn't been used
                            while p2_idx < len(p2):
                                candidate_piece_id, _ = p2[p2_idx]
                                p2_idx += 1
                                if candidate_piece_id not in used_piece_ids:
                                    child[i] = (candidate_piece_id, random.choice(self.rotations))
                                    used_piece_ids.add(candidate_piece_id)
                                    break
                else:
                    child = random.choice(elite).copy()

                if random.random() < 0.85:
                    child = mutate(child)
                
                # Verify child maintains constraint
                if is_valid_arrangement(child):
                    new_pop.append(child)
                else:
                    # If crossover violates constraint, just use parent
                    new_pop.append(random.choice(elite).copy())

            population = new_pop
            generation += 1

        print(f"\nCompleted {generation} generations")
        return best_ever, best_ever_score


def visualize_solution(pieces: List[PuzzlePiece],
                       arrangement: List[Tuple[int, int]],
                       rows: int, cols: int,
                       padding: int = 2) -> np.ndarray:
    """Create visualization."""
    rotated_pieces = []
    max_h, max_w = 0, 0

    for piece_id, rotation in arrangement:
        patch, mask = rotate_piece(pieces[piece_id].patch, pieces[piece_id].mask, rotation)
        rotated_pieces.append((patch, mask))
        max_h = max(max_h, patch.shape[0])
        max_w = max(max_w, patch.shape[1])

    canvas_h = rows * (max_h + padding) + padding
    canvas_w = cols * (max_w + padding) + padding
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 20

    for idx, (patch, mask) in enumerate(rotated_pieces):
        r = idx // cols
        c = idx % cols
        h, w = patch.shape[:2]

        y = padding + r * (max_h + padding) + (max_h - h) // 2
        x = padding + c * (max_w + padding) + (max_w - w) // 2

        for i in range(h):
            for j in range(w):
                if mask[i, j] > 0:
                    cy, cx = y + i, x + j
                    if 0 <= cy < canvas_h and 0 <= cx < canvas_w:
                        canvas[cy, cx] = patch[i, j]

    # Grid lines
    for r in range(rows + 1):
        y = r * (max_h + padding)
        cv2.line(canvas, (0, y), (canvas_w, y), (70, 70, 70), 1)
    for c in range(cols + 1):
        x = c * (max_w + padding)
        cv2.line(canvas, (x, 0), (x, canvas_h), (70, 70, 70), 1)

    return canvas


def solve_puzzle(components: List, rows: int, cols: int,
                 time_limit: float = 115) -> Tuple[List[Tuple[int, int]], np.ndarray, float]:
    """
    Solve puzzle using hybrid local+global optimization.
    """
    pieces = [PuzzlePiece(patch=comp.patch, mask=comp.mask, id=i)
              for i, comp in enumerate(components)]

    print(f"\n{'=' * 70}")
    print(f"HYBRID PUZZLE SOLVER (Edge Matching + Global Coherence)")
    print(f"{'=' * 70}")
    print(f"Puzzle: {rows}x{cols} = {len(pieces)} pieces")
    print(f"Time budget: {time_limit}s\n")

    solver = PuzzleSolver(pieces, rows, cols)
    arrangement, score = solver.solve(time_limit=time_limit)

    print(f"\n{'=' * 70}")
    print(f"SOLUTION")
    print(f"{'=' * 70}")
    print(f"Final score: {score:.2f}\n")

    for r in range(rows):
        row_str = ""
        for c in range(cols):
            idx = r * cols + c
            piece_id, rotation = arrangement[idx]
            row_str += f"[{piece_id:2d}@{rotation:3d}°] "
        print(row_str)

    result_image = visualize_solution(pieces, arrangement, rows, cols)

    return arrangement, result_image, score