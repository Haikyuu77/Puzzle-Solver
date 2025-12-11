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

# Horizontal: Threshold for Block MSE
MAX_HORIZONTAL_COST = 1000.0  

MIN_ROWS_TO_CHECK = 3
MAX_ROWS_TO_CHECK = 3 

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
# 2. ADVANCED METRICS
# ==========================================
def get_block_horizontal_cost(lab1, side1, lab2, side2):
    """ 
    Horizontal: 4-pixel deep Block MSE.
    Using more depth makes it robust against single-pixel noise.
    """
    # Depth of comparison
    DEPTH = 4
    
    if side1 == 'right':  
        # Take rightmost 4 columns
        block1 = lab1[:, -DEPTH:, :]
    else: 
        block1 = lab1[:, :DEPTH, :] # Should not happen for side1='right' logic
        
    if side2 == 'left':   
        # Take leftmost 4 columns
        block2 = lab2[:, :DEPTH, :]
    else:
        block2 = lab2[:, -DEPTH:, :]

    # Align heights
    h1 = block1.shape[0]
    h2 = block2.shape[0]
    length = min(h1, h2)
    if length < 5: return float('inf')
    
    b1 = block1[:length]
    b2 = block2[:length]
    
    # Calculate MSE on the blocks
    diff_sq = np.square(b1 - b2)
    mse = np.mean(diff_sq)
    
    return mse

def calculate_zncc(patch1, patch2):
    """
    Zero-Normalized Cross-Correlation (ZNCC).
    Returns value between -1.0 and 1.0.
    1.0 = Perfect pattern match (ignoring brightness shift).
    0.0 = No correlation (random noise).
    """
    # Flatten arrays
    v1 = patch1.flatten()
    v2 = patch2.flatten()
    
    # Zero-Normalize (Subtract Mean)
    v1 = v1 - np.mean(v1)
    v2 = v2 - np.mean(v2)
    
    # Compute Norms
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-6 or norm2 < 1e-6:
        # Flat color area (no texture) -> Correlation is undefined/low
        return 0.0
        
    # Correlation
    score = np.dot(v1, v2) / (norm1 * norm2)
    return score

def get_vertical_zncc_score(row1_lab, row2_lab):
    """ 
    Vertical: Deep Band ZNCC.
    Uses a deep strip of data (16 pixels) to match texture patterns.
    Higher Score is Better.
    """
    # Depth of vertical overlap check
    DEPTH = 16
    
    h1, w1 = row1_lab.shape[:2]
    h2, w2 = row2_lab.shape[:2]
    
    # Extract bottom strip of Top Row
    strip1 = row1_lab[-DEPTH:, :, 0] # Use L channel for structure
    
    # Extract top strip of Bottom Row
    strip2 = row2_lab[:DEPTH, :, 0]
    
    # Allow sliding
    search_range = int(min(w1, w2) * 0.2)
    min_overlap = int(min(w1, w2) * 0.8)
    
    best_score = -1.0 # Initialize with worst possible correlation
    
    for offset in range(-search_range, search_range + 1):
        # Calculate overlap indices
        start1 = max(0, -offset)
        end1 = min(w1, w2 - offset)
        start2 = max(0, offset)
        end2 = min(w2, w1 + offset)
        
        len1 = end1 - start1
        if len1 < min_overlap: continue
            
        s1 = strip1[:, start1:end1]
        s2 = strip2[:, start2:end2]
        
        score = calculate_zncc(s1, s2)
        
        if score > best_score:
            best_score = score
            
    return best_score

def stitch_row(row_pieces):
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
# 3. ROW BUILDER (MBM Optimized)
# ==========================================
def find_valid_rows(pieces, target_width, pairwise_costs):
    n = len(pieces)
    
    # Find Mutual Best Matches
    best_right = {}
    best_left = {}
    
    for i in range(n):
        # Right Neighbors
        cands = []
        for j in range(n):
            if i == j: continue
            cands.append((pairwise_costs[pieces[i].id][pieces[j].id], pieces[j].id))
        cands.sort(key=lambda x: x[0])
        best_right[pieces[i].id] = cands[0]
        
        # Left Neighbors
        cands_l = []
        for j in range(n):
            if i == j: continue
            cands_l.append((pairwise_costs[pieces[j].id][pieces[i].id], pieces[j].id))
        cands_l.sort(key=lambda x: x[0])
        best_left[pieces[i].id] = cands_l[0]

    neighbors_map = {}
    for p in pieces:
        candidates = []
        for other in pieces:
            if p.id == other.id: continue
            c = pairwise_costs[p.id][other.id]
            if c < MAX_HORIZONTAL_COST:
                # Prioritize Mutual Matches
                is_mutual = (best_right[p.id][1] == other.id and best_left[other.id][1] == p.id)
                if is_mutual:
                    c /= 100.0 
                candidates.append((c, other))
        
        candidates.sort(key=lambda x: x[0])
        neighbors_map[p.id] = candidates[:8]

    valid_rows = []
    
    def dfs_build_row(chain, current_w, remaining_ids):
        if abs(current_w - target_width) / target_width <= WIDTH_TOLERANCE:
            valid_rows.append(list(chain))
        
        if current_w > target_width * (1 + WIDTH_TOLERANCE): return

        last_p = chain[-1]
        best_next = neighbors_map[last_p.id]
        
        for cost, next_p in best_next:
            if next_p.id in remaining_ids:
                dfs_build_row(chain + [next_p], current_w + next_p.w, remaining_ids - {next_p.id})

    print(f"  Searching for rows ~{int(target_width)}px wide...")
    for p in pieces:
        dfs_build_row([p], p.w, {x.id for x in pieces if x.id != p.id})
        
    unique_rows = []
    seen_hashes = set()
    for row in valid_rows:
        h = tuple(p.id for p in row)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_rows.append(row)
    return unique_rows

# ==========================================
# 4. PARTITION SOLVER (Maximizing ZNCC)
# ==========================================
def solve_by_partitioning(valid_rows, target_num_rows, total_pieces):
    row_info = []
    for i, row in enumerate(valid_rows):
        mask = 0
        for p in row: mask |= (1 << p.id)
        lab = stitch_row(row)
        row_info.append({'mask': mask, 'pieces': row, 'lab': lab})

    target_mask = (1 << total_pieces) - 1
    found_partitions = []

    def find_cover(current_selection, current_mask, start_idx):
        if len(found_partitions) > 100: return 

        if current_mask == target_mask:
            if len(current_selection) == target_num_rows:
                found_partitions.append(list(current_selection))
            return
        if len(current_selection) >= target_num_rows: return

        for i in range(start_idx, len(row_info)):
            r = row_info[i]
            if (current_mask & r['mask']) == 0:
                find_cover(current_selection + [r], current_mask | r['mask'], i + 1)

    print(f"  > Finding disjoint row sets...")
    find_cover([], 0, 0)
    print(f"  > Found {len(found_partitions)} valid partitions.")
    
    if not found_partitions: return None, float('-inf')

    print(f"  > Vertical Ordering (Maximizing ZNCC)...")
    
    best_overall_layout = None
    max_overall_score = float('-inf')
    
    for partition in found_partitions:
        local_best_layout = None
        local_max_score = float('-inf')

        for perm in itertools.permutations(partition):
            total_score = 0
            
            # Sum of ZNCC correlations for internal seams
            for k in range(len(perm) - 1):
                top_row = perm[k]
                btm_row = perm[k+1]
                
                # ZNCC Score (Higher is Better)
                score = get_vertical_zncc_score(top_row['lab'], btm_row['lab'])
                total_score += score
            
            if total_score > local_max_score:
                local_max_score = total_score
                local_best_layout = [r['pieces'] for r in perm]

        if local_max_score > max_overall_score:
            max_overall_score = local_max_score
            best_overall_layout = local_best_layout

    return best_overall_layout, max_overall_score

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
        w = sum(p.w for p in row); h = max(p.h for p in row)
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

def solve_puzzle(pieces):
    total_width = sum(p.w for p in pieces)
    total_pieces = len(pieces)
    print(f"Total Width: {total_width}px | Pieces: {total_pieces}")

    print("Pre-calculating pairwise costs...")
    costs = np.zeros((total_pieces, total_pieces))
    for i in range(total_pieces):
        for j in range(total_pieces):
            if i != j:
                # Use Block MSE for Horizontal
                costs[i][j] = get_block_horizontal_cost(pieces[i].lab, 'right', pieces[j].lab, 'left')
            else:
                costs[i][j] = float('inf')

    best_overall_layout = None
    best_overall_score = float('-inf')

    start_r = max(1, MIN_ROWS_TO_CHECK)
    end_r = min(total_pieces, MAX_ROWS_TO_CHECK)
    
    for r in range(start_r, end_r + 1):
        target_w = total_width / r
        if target_w < max(p.w for p in pieces): continue
        
        print(f"\n--- Checking {r} Rows (Target Width: {int(target_w)} px) ---")
        
        valid_rows = find_valid_rows(pieces, target_w, costs)
        print(f"  > Found {len(valid_rows)} potential row segments.")
        if len(valid_rows) < r: continue
            
        row_scores = []
        for row in valid_rows:
            s = 0
            for k in range(len(row)-1): s += costs[row[k].id][row[k+1].id]
            row_scores.append((s, row))
        row_scores.sort(key=lambda x: x[0])
        
        # Increased search space
        top_rows = [x[1] for x in row_scores[:4000]] 
        
        layout, score = solve_by_partitioning(top_rows, r, total_pieces)
        
        if layout:
            print(f"  > Valid Solution Found! ZNCC Score: {score:.2f}")
            if score > best_overall_score:
                best_overall_score = score
                best_overall_layout = layout

    if best_overall_layout:
        if os.path.exists(DEBUG_DIR): shutil.rmtree(DEBUG_DIR)
        os.makedirs(DEBUG_DIR)
        img = render_full_board(best_overall_layout)
        filename = f"{DEBUG_DIR}FINAL_BEST_RESULT.png"
        cv2.imwrite(filename, img)
        print(f"\n>>> SAVED BEST: {filename}")
    else:
        print("\nNo solution found.")

if __name__ == "__main__":
    pieces = extract_pieces(INPUT_IMAGE)
    solve_puzzle(pieces)