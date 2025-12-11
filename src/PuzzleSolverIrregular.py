import cv2
import numpy as np
import os
import shutil
import itertools
import sys

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_IMAGE = '../images/train_irregular_translate.png'
DEBUG_DIR = './debug_candidates/'
MATCH_CROP = 0      
WIDTH_TOLERANCE = 0.01  # 20% to handle uneven row splits (6 vs 7 pieces)

# Strictness for initial row grouping
MAX_HORIZONTAL_COST = 3000.0  

class PuzzlePiece:
    def __init__(self, original_img, straight_img, center, angle, idx):
        self.id = idx
        self.img = straight_img
        self.h, self.w = straight_img.shape[:2]
        crop = int(MATCH_CROP)
        if crop > 0 and self.h > 2*crop and self.w > 2*crop:
            safe = straight_img[crop:self.h-crop, crop:self.w-crop]
        else:
            safe = straight_img
        self.lab = cv2.cvtColor(safe, cv2.COLOR_BGR2LAB).astype("float32")

# ==========================================
# 2. METRICS
# ==========================================
def get_mse_cost(img1, img2):
    """ Standard Mean Squared Error """
    diff = img1 - img2
    return np.mean(np.square(diff))

def get_horizontal_cost(p_left, p_right):
    """
    Gradient-aware horizontal cost between two neighboring pieces.
    Uses:
      - color continuity across the seam
      - gradient continuation across the seam
    """
    lab1 = p_left.lab
    lab2 = p_right.lab

    # Align height
    h = min(lab1.shape[0], lab2.shape[0])

    # Use last column of left and first column of right
    edge1 = lab1[:h, -1, :]          # right edge of left piece
    edge2 = lab2[:h, 0, :]           # left edge of right piece

    # Inner columns for gradient prediction (one pixel inside)
    inner1 = lab1[:h, -2, :] if lab1.shape[1] > 1 else edge1
    inner2 = lab2[:h, 1, :]  if lab2.shape[1] > 1 else edge2

    # 1) Color error (simple seam difference)
    color_err = np.linalg.norm(edge1 - edge2)

    # 2) Gradient continuation:
    #    extrapolate how each side "wants" to continue, and see
    #    how compatible that is with the other side.
    pred_1 = edge1 + (edge1 - inner1)   # extend left → right
    pred_2 = edge2 + (edge2 - inner2)   # extend right → left

    grad_err = np.linalg.norm(pred_1 - edge2) + np.linalg.norm(pred_2 - edge1)

    grad_weight = 2 # you can tune this (e.g., 0.5–2.0)
    return color_err + grad_weight * grad_err

def get_vertical_seam_cost(strip_top, strip_bot):
    """
    Gradient-aware vertical cost between two full row strips.
    Assumes both strips are in LAB space (as returned by stitch_row).
    Uses:
      - color continuity across horizontal seam
      - gradient continuation across that seam
    """
    lab1 = strip_top
    lab2 = strip_bot

    # Align width
    w = min(lab1.shape[1], lab2.shape[1])

    # Bottom row of top strip and top row of bottom strip
    edge1 = lab1[-1, :w, :]            # bottom edge of top row
    edge2 = lab2[0,  :w, :]            # top edge of bottom row

    # Inner rows for gradient prediction (one pixel inside)
    inner1 = lab1[-2, :w, :] if lab1.shape[0] > 1 else edge1
    inner2 = lab2[1,  :w, :] if lab2.shape[0] > 1 else edge2

    # 1) Color seam error
    color_err = np.linalg.norm(edge1 - edge2)

    # 2) Gradient continuation vertical:
    #    extend top row downwards, bottom row upwards and compare.
    pred_1 = edge1 + (edge1 - inner1)   # extend downward
    pred_2 = edge2 + (edge2 - inner2)   # extend upward

    grad_err = np.linalg.norm(pred_1 - edge2) + np.linalg.norm(pred_2 - edge1)

    grad_weight = 1  # tune if needed
    return color_err + grad_weight * grad_err

def stitch_row(row_pieces):
    """ Renders list of pieces into a single image strip """
    w = sum(p.w for p in row_pieces)
    h = max(p.h for p in row_pieces)
    strip = np.zeros((h, w, 3), dtype=np.uint8)
    cx = 0
    for p in row_pieces:
        y_off = (h - p.h)//2
        strip[y_off:y_off+p.h, cx:cx+p.w] = p.img
        cx += p.w
    crop = int(MATCH_CROP)
    if crop > 0: safe = strip[crop:h-crop, crop:w-crop]
    else: safe = strip
    return cv2.cvtColor(safe, cv2.COLOR_BGR2LAB).astype("float32")

# ==========================================
# 3. ROW PARTITIONING
# ==========================================
def find_row_partitions(pieces, target_width, pairwise_costs):
    """
    Groups pieces into 3 disjoint sets based on width and horizontal fit.
    """
    neighbors_map = {}
    for p in pieces:
        candidates = []
        for other in pieces:
            if p.id == other.id: continue
            c = pairwise_costs[p.id][other.id]
            if c < MAX_HORIZONTAL_COST:
                candidates.append((c, other))
        candidates.sort(key=lambda x: x[0])
        neighbors_map[p.id] = candidates[:8]

    valid_chains = []
    
    def dfs(chain, cw, rem):
        # Found a valid width?
        if abs(cw - target_width) / target_width <= WIDTH_TOLERANCE:
            score = 0
            for k in range(len(chain)-1):
                score += pairwise_costs[chain[k].id][chain[k+1].id]
            avg_score = score / max(1, len(chain)-1)
            valid_chains.append({'pieces': list(chain), 'score': avg_score})
        
        if cw > target_width * (1 + WIDTH_TOLERANCE): return

        last_p = chain[-1]
        best_next = neighbors_map.get(last_p.id, [])
        for cost, next_p in best_next:
            if next_p.id in remaining_ids:
                dfs(chain + [next_p], cw + next_p.w, remaining_ids - {next_p.id})

    row_sets = []
    current_pool = list(pieces)
    
    for i in range(3):
        if i == 2:
            row_sets.append(current_pool) # Remainder
            break
            
        remaining_ids = {p.id for p in current_pool}
        valid_chains = []
        
        # Run DFS for this iteration
        for p in current_pool:
            dfs([p], p.w, remaining_ids - {p.id})
            
        if not valid_chains: return None
        
        # Pick best chain
        valid_chains.sort(key=lambda x: x['score'])
        best_chain = valid_chains[0]['pieces']
        row_sets.append(best_chain)
        
        # Remove used
        used_ids = {p.id for p in best_chain}
        current_pool = [p for p in current_pool if p.id not in used_ids]
    
    for row in row_sets:
        print("row size : ", len(row))
        
    return row_sets

def solve_puzzle(pieces):
    n = len(pieces)
    total_width = sum(p.w for p in pieces)
    print(f"Total Width: {total_width}px | Pieces: {n}")

    # Precompute horizontal pairwise cost
    h_costs = np.full((n, n), float('inf'))
    for i in range(n):
        for j in range(n):
            if i != j:
                h_costs[i][j] = get_horizontal_cost(pieces[i], pieces[j])

    # Partition into 3 row sets
    target_w = total_width / 3.0
    row_sets = find_row_partitions(pieces, target_w, h_costs)
    if row_sets is None:
        print("No valid row groups found.")
        return None

    print("Row sets found. Beginning greedy global assembly...")

    # -----------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------

    def build_row_greedy(piece_list, start_piece):
        """ Build a row left→right purely by horizontal cost. """
        remaining = set(p.id for p in piece_list)
        row = [start_piece]
        remaining.remove(start_piece.id)

        while remaining:
            last = row[-1].id
            best, best_cost = None, float('inf')

            for r in remaining:
                c = h_costs[last][r]
                if c < best_cost:
                    best_cost = c
                    best = r

            row.append(pieces[best])
            remaining.remove(best)

        return row

    def pick_best_row_by_vertical(strip_top, candidate_rows):
        """ Return row (list of pieces) whose LEFTMOST piece gives best vertical seam vs. strip_top. """
        best_row = None
        best_mse = float('inf')

        for row in candidate_rows:
            for p in row:
                test_strip = stitch_row([p])
                mse = get_vertical_seam_cost(strip_top, test_strip)
                if mse < best_mse:
                    best_mse = mse
                    best_row = row

        return best_row

    def build_row_greedy_with_vertical(row_list, strip_above):
        """
        Greedy row building using:
            score = α * horizontal_cost + β * vertical_cost
        """
        α = 1.0
        β = 2.0

        remaining = set(p.id for p in row_list)
        built = []

        # Pick the best starting piece relative to the row above
        best_start = None
        best_v = float('inf')
        for p in row_list:
            test_strip = stitch_row([p])
            v = get_vertical_seam_cost(strip_above, test_strip)
            if v < best_v:
                best_v = v
                best_start = p

        built.append(best_start)
        remaining.remove(best_start.id)

        # Fill remaining pieces greedily
        while remaining:
            last = built[-1]
            best_id = None
            best_score = float('inf')

            for r in remaining:
                h = h_costs[last.id][r]
                test_strip = stitch_row([pieces[r]])
                v = get_vertical_seam_cost(strip_above, test_strip)
                score = α * h + β * v
                if score < best_score:
                    best_score = score
                    best_id = r

            built.append(pieces[best_id])
            remaining.remove(best_id)

        return built

    # -----------------------------------------------------------
    # GLOBAL SEARCH
    # -----------------------------------------------------------
    global_best = None
    global_best_score = float('inf')

    # Try each row as TOP
    for top_idx in range(3):

        top_row = row_sets[top_idx]
        print(f"Trying row {top_idx} as TOP...")

        # Try every piece in the top row as the starting element
        for start_piece in top_row:

            # --------------------------------------------------
            # IMPORTANT FIX — reset fresh copies per iteration
            # --------------------------------------------------
            other_rows = [
                row_sets[(top_idx + 1) % 3][:],  # copy list
                row_sets[(top_idx + 2) % 3][:]
            ]

            # 1. Build the top row greedily
            row1 = build_row_greedy(top_row, start_piece)
            strip1 = stitch_row(row1)

            # 2. Pick the best middle row by vertical seam
            mid_row_set = pick_best_row_by_vertical(strip1, other_rows)

            # Fallback safety
            if mid_row_set not in other_rows:
                mid_row_set = other_rows[0]

            # Remove selected middle row
            other_rows.remove(mid_row_set)

            if not other_rows:
                print("ERROR: No remaining row for bottom — skipping configuration")
                continue

            bot_row_set = other_rows[0]

            # 3. Build the second row
            row2 = build_row_greedy_with_vertical(mid_row_set, strip1)
            strip2 = stitch_row(row2)

            # 4. Build the bottom row
            row3 = build_row_greedy_with_vertical(bot_row_set, strip2)
            strip3 = stitch_row(row3)

            # 5. Compute total seam score
            v12 = get_vertical_seam_cost(strip1, strip2)
            v23 = get_vertical_seam_cost(strip2, strip3)
            total_score = v12 + v23
            
            attempt_img = render_full_board([row1, row2, row3])
            if attempt_img is not None:
                if not os.path.exists(DEBUG_DIR):
                    os.makedirs(DEBUG_DIR)
                attempt_path = os.path.join(
                    DEBUG_DIR,
                    f"ATTEMPT_top{top_idx}_start{start_piece.id}_score_{total_score:.2f}.png"
                )
                cv2.imwrite(attempt_path, attempt_img)
                print(f"[DEBUG] Saved attempt → {attempt_path}")

            # -------------------------
            # GLOBAL BEST check
            # ------------------------            

            if total_score < global_best_score:
                global_best_score = total_score
                global_best = [row1, row2, row3]
                print(f"  New Best Global Score = {total_score:.2f}")

    print("\n===== BEST SOLUTION FOUND =====")
    print(f"Score: {global_best_score:.2f}")

    # Save final stitched image
    final_img = render_full_board(global_best)
    if final_img is not None:
        if not os.path.exists(DEBUG_DIR):
            os.makedirs(DEBUG_DIR)
        out_path = os.path.join(DEBUG_DIR, "FINAL_RECONSTRUCTION.png")
        cv2.imwrite(out_path, final_img)
        print(f"[Saved] Final reconstructed puzzle → {out_path}")

    return global_best


def extract_pieces(image_path):
    if not os.path.exists(image_path): raise FileNotFoundError(image_path)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_pieces = []
    idx = 0
    for cnt in contours:
        if cv2.contourArea(cnt) < 500: continue
        rect = cv2.minAreaRect(cnt)
        pts = np.int64(cv2.boxPoints(rect)).astype("float32")
        s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
        tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]; bl = pts[np.argmax(diff)]
        w = int(np.linalg.norm(tr-tl)); h = int(np.linalg.norm(bl-tl))
        src = np.array([tl, tr, br, bl], dtype="float32")
        dst = np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]], dtype="float32")
        warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h))
        raw_pieces.append(PuzzlePiece(img, warped, rect[0], rect[2], idx))
        idx += 1
    return raw_pieces

def render_full_board(final_rows):
    if not final_rows: return None
    max_w = 0; total_h = 0
    for row in final_rows:
        w = sum(p.w for p in row)
        h = max(p.h for p in row)
        max_w = max(max_w, w); total_h += h
    canvas = np.zeros((total_h, max_w, 3), dtype=np.uint8)
    cur_y = 0
    for row in final_rows:
        row_w = sum(p.w for p in row)
        row_h = max(p.h for p in row)
        cur_x = (max_w - row_w) // 2 
        for p in row:
            y_off = (row_h - p.h) // 2
            canvas[cur_y+y_off:cur_y+y_off+p.h, cur_x:cur_x+p.w] = p.img
            cur_x += p.w
        cur_y += row_h
    return canvas

if __name__ == "__main__":
    pieces = extract_pieces(INPUT_IMAGE)
    solve_puzzle(pieces)