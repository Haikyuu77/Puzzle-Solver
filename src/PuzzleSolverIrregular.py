import cv2
import numpy as np
import os
import itertools
import shutil
import math

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_IMAGE = '../images/train_irregular.png'
DEBUG_DIR = './debug_candidates/'
MATCH_CROP = 4          # Pixels to strip (removes black extraction borders)
WIDTH_TOLERANCE = 0.05  # 2% width deviation allowed for geometric rows

# --- SEARCH THRESHOLDS ---
MIN_ROWS_TO_CHECK = 3   # Don't check 1 or 2 rows if you know it's a grid
MAX_ROWS_TO_CHECK = 3  # Stop checking after this many rows

# ==========================================
# 1. PUZZLE PIECE CLASS
# ==========================================
class PuzzlePiece:
    def __init__(self, original_img, straight_img, center, angle, idx):
        self.id = idx
        self.img = straight_img
        self.h, self.w = straight_img.shape[:2]
        
        # Prepare High-Res LAB Data (No Resizing)
        crop = int(MATCH_CROP)
        if self.h > 2*crop and self.w > 2*crop:
            safe = straight_img[crop:self.h-crop, crop:self.w-crop]
        else:
            safe = straight_img
            
        self.lab = cv2.cvtColor(safe, cv2.COLOR_BGR2LAB).astype("float32")

# ==========================================
# 2. ADVANCED METRIC: DERIVATIVE CONTINUITY
# ==========================================
def get_seam_cost(lab1, side1, lab2, side2):
    """
    Calculates cost based on 2nd Derivative (Smoothness of gradient).
    Lower score = Better match.
    """
    # 1. Extract Edges (Deep enough to calculate local gradient)
    depth = 2
    
    if side1 == 'right':  edge1 = lab1[:, -1, :]; inner1 = lab1[:, -2, :]
    elif side1 == 'bottom': edge1 = lab1[-1, :, :]; inner1 = lab1[-2, :, :]
    
    if side2 == 'left':   edge2 = lab2[:, 0, :];  inner2 = lab2[:, 1, :]
    elif side2 == 'top':  edge2 = lab2[0, :, :];  inner2 = lab2[1, :, :]
    
    # 2. Align Dimensions (Crop to overlap, NO RESIZING to prevent drift)
    if side1 == 'right': # Vertical Seam
        length = min(edge1.shape[0], edge2.shape[0])
        edge1 = edge1[:length]; inner1 = inner1[:length]
        edge2 = edge2[:length]; inner2 = inner2[:length]
    else: # Horizontal Seam
        length = min(edge1.shape[1], edge2.shape[1])
        edge1 = edge1[:length]; inner1 = inner1[:length]
        edge2 = edge2[:length]; inner2 = inner2[:length]
        
    if length == 0: return float('inf')

    # 3. Calculate "Velocity" of the pattern
    # Rate of change inside Piece A
    grad_A = edge1 - inner1
    
    # Rate of change crossing the seam (A -> B)
    grad_Seam = edge2 - edge1
    
    # 4. Continuity Cost
    # If the seam is good, the change across the seam should match the change inside A
    # Cost = || Grad_Seam - Grad_A ||
    delta = np.abs(grad_Seam - grad_A)
    
    # 5. Weighting (Punish Color jumps more than Lightness jumps)
    # L=0.5, A=2.0, B=2.0
    weighted_delta = (delta[:,0] * 0.5) + (delta[:,1] * 2.0) + (delta[:,2] * 2.0)
    
    # Robust Score: Trimmed Mean (Ignore worst 20% noise/artifacts)
    errors = np.square(weighted_delta)
    errors.sort()
    keep_n = int(len(errors) * 0.8)
    if keep_n < 1: keep_n = 1
    
    return np.mean(errors[:keep_n])

# ==========================================
# 3. GEOMETRIC PARTITIONER
# ==========================================
def find_partitions(pieces, num_rows, target_width, tolerance_px):
    if num_rows == 1:
        current_sum = sum(p.w for p in pieces)
        if abs(current_sum - target_width) <= tolerance_px:
            return [[pieces]] 
        else:
            return [] 

    valid_solutions = []
    pieces = sorted(pieces, key=lambda p: p.w, reverse=True)
    n = len(pieces)
    
    # Heuristic cap to prevent infinite recursion
    max_pieces = int((n / num_rows) + 4)
    
    for r_size in range(1, min(n, max_pieces)): 
        for subset in itertools.combinations(pieces, r_size):
            w_sum = sum(p.w for p in subset)
            if w_sum > target_width + tolerance_px: continue 
            
            if abs(w_sum - target_width) <= tolerance_px:
                remaining = [p for p in pieces if p not in subset]
                sub_solutions = find_partitions(remaining, num_rows - 1, target_width, tolerance_px)
                for sol in sub_solutions:
                    valid_solutions.append([list(subset)] + sol)
                if len(valid_solutions) > 5: return valid_solutions
    return valid_solutions

def generate_hypotheses(pieces):
    total_width = sum(p.w for p in pieces)
    print(f"Total Width: {total_width}px")
    
    hypotheses = []
    
    # Apply Min/Max Thresholds
    start_r = max(1, MIN_ROWS_TO_CHECK)
    end_r = MAX_ROWS_TO_CHECK
    
    for r in range(start_r, end_r + 1):
        target_w = total_width / r
        max_p_w = max(p.w for p in pieces)
        
        # Sanity check: Row cannot be smaller than the largest piece
        if target_w < max_p_w: continue 
        
        print(f"Checking {r} Rows (Target: ~{int(target_w)}px)...")
        tolerance = target_w * WIDTH_TOLERANCE
        partitions = find_partitions(pieces, r, target_w, tolerance)
        
        if partitions:
            print(f"  > FOUND {len(partitions)} valid partitions.")
            # Flatten partitions into hypothesis list
            for p in partitions:
                hypotheses.append(p)
                
    return hypotheses

# ==========================================
# 4. SOLVERS (Greedy + Permutations)
# ==========================================
def solve_intra_row(row_pieces):
    n = len(row_pieces)
    if n == 1: return row_pieces, 0.0
    
    costs = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: continue
            costs[i][j] = get_seam_cost(row_pieces[i].lab, 'right', row_pieces[j].lab, 'left')
            
    # Brute force small rows, greedy large rows
    if n <= 8:
        best_perm = None; min_err = float('inf')
        for perm in itertools.permutations(range(n)):
            err = 0
            for k in range(n-1): err += costs[perm[k]][perm[k+1]]
            if err < min_err: min_err = err; best_perm = perm
        return [row_pieces[i] for i in best_perm], min_err
    else:
        # Greedy fallback
        best_chain = None; min_err = float('inf')
        for start_node in range(n):
            chain = [start_node]; used = {start_node}; curr_cost = 0; curr = start_node
            while len(chain) < n:
                best_next = -1; best_c = float('inf')
                for cand in range(n):
                    if cand not in used:
                        if costs[curr][cand] < best_c: best_c = costs[curr][cand]; best_next = cand
                if best_next != -1: chain.append(best_next); used.add(best_next); curr_cost += best_c; curr = best_next
                else: break
            if len(chain) == n and curr_cost < min_err: min_err = curr_cost; best_chain = chain
        return [row_pieces[i] for i in best_chain], min_err

def stitch_strip(row_pieces):
    w = sum(p.w for p in row_pieces)
    h = max(p.h for p in row_pieces)
    strip = np.zeros((h, w, 3), dtype=np.uint8)
    cx = 0
    for p in row_pieces:
        # Align Center Vertically within row
        y_off = (h - p.h)//2
        strip[y_off:y_off+p.h, cx:cx+p.w] = p.img
        cx += p.w
    return strip

def solve_inter_row(ordered_rows):
    n = len(ordered_rows)
    if n == 1: return ordered_rows, 0.0
    
    # 1. Render Strips (No Resizing)
    strips_data = []
    for r in ordered_rows:
        img = stitch_strip(r)
        
        # Prepare LAB
        crop = int(MATCH_CROP)
        h, w = img.shape[:2]
        safe = img[crop:h-crop, crop:w-crop] # Crop border only
        lab = cv2.cvtColor(safe, cv2.COLOR_BGR2LAB).astype("float32")
        strips_data.append(lab)
        
    # 2. Match
    costs = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: continue
            costs[i][j] = get_seam_cost(strips_data[i], 'bottom', strips_data[j], 'top')
            
    # 3. Permute Rows
    best_perm = None; min_err = float('inf')
    
    # Rows are usually < 8, so permutations are safe
    for perm in itertools.permutations(range(n)):
        err = 0
        for k in range(n-1):
            err += costs[perm[k]][perm[k+1]]
        if err < min_err:
            min_err = err
            best_perm = perm
            
    return [ordered_rows[i] for i in best_perm], min_err

# ==========================================
# 5. DEBUG RENDERER
# ==========================================
def render_full_board(final_rows):
    # Calculate size
    max_w = 0; total_h = 0
    for row in final_rows:
        w = sum(p.w for p in row); h = max(p.h for p in row)
        max_w = max(max_w, w); total_h += h
    
    canvas = np.zeros((total_h, max_w, 3), dtype=np.uint8)
    cur_y = 0
    for row in final_rows:
        row_w = sum(p.w for p in row)
        row_h = max(p.h for p in row)
        
        # Center Row horizontally
        cur_x = (max_w - row_w) // 2
        
        for p in row:
            # Center Piece Vertically in Row
            y_off = (row_h - p.h) // 2
            canvas[cur_y+y_off:cur_y+y_off+p.h, cur_x:cur_x+p.w] = p.img
            cur_x += p.w
        cur_y += row_h
        
    return canvas

def save_debug_candidates(candidates_with_scores):
    # Sort by score ascending (lowest error first)
    candidates_with_scores.sort(key=lambda x: x['score'])
    
    if os.path.exists(DEBUG_DIR): shutil.rmtree(DEBUG_DIR)
    os.makedirs(DEBUG_DIR)
    
    print(f"\n--- Saving Top Candidates to {DEBUG_DIR} ---")
    
    # Save Top 10
    for i, item in enumerate(candidates_with_scores[:10]):
        img = render_full_board(item['layout'])
        score = int(item['score'])
        rows = len(item['layout'])
        filename = f"{DEBUG_DIR}rank_{i+1}_score_{score}_rows_{rows}.png"
        cv2.imwrite(filename, img)
        print(f"  Saved Rank {i+1}: Score {score} (Rows: {rows})")

# ==========================================
# 6. MAIN PIPELINE
# ==========================================
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

def solve_and_debug(pieces):
    candidates = generate_hypotheses(pieces)
    if not candidates:
        print("No valid partition found.")
        return

    results = []
    
    print(f"Evaluating {len(candidates)} partitions...")
    for i, rows in enumerate(candidates):
        # 1. Solve Intra
        solved_rows = []
        intra_cost = 0
        for r in rows:
            ordered, cost = solve_intra_row(r)
            solved_rows.append(ordered)
            intra_cost += cost
            
        # 2. Solve Inter
        final_layout, inter_cost = solve_inter_row(solved_rows)
        
        # 3. Normalized Score
        total_connections = (len(rows) * (len(rows[0])-1)) + (len(rows)-1)
        total_cost = intra_cost + inter_cost
        avg_score = total_cost / max(1, total_connections)
        
        results.append({
            'layout': final_layout,
            'score': avg_score
        })
        
    save_debug_candidates(results)

if __name__ == "__main__":
    try:
        pieces = extract_pieces(INPUT_IMAGE)
        solve_and_debug(pieces)
    except Exception as e:
        print(f"Error: {e}")