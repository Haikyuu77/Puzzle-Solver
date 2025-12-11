import cv2
import numpy as np
import os
import shutil
import itertools
import sys

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_IMAGE = '../images/train_irregular.png'
DEBUG_DIR = './debug_candidates/'
MATCH_CROP = 2          
WIDTH_TOLERANCE = 0.01  # 3% tolerance

# --- TUNING ---
# Horizontal: L2 Norm (Standard pixel difference is best for side-by-side)
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
# 2. METRICS
# ==========================================
def get_horizontal_cost(lab1, side1, lab2, side2):
    """ 
    Horizontal: Standard L2 Difference. 
    Side connections are usually clean cuts, so pixel diff is reliable.
    """
    if side1 == 'right':  edge1 = lab1[:, -1, :]; inner1 = lab1[:, -2, :]
    if side2 == 'left':   edge2 = lab2[:, 0, :];  inner2 = lab2[:, 1, :]
    
    length = min(edge1.shape[0], edge2.shape[0])
    if length == 0: return float('inf')
    
    e1 = edge1[:length]
    e2 = edge2[:length]
    
    # Simple L2 Norm
    delta = np.abs(e2 - e1)
    weighted_delta = (delta[:,0] * 1.0) + (delta[:,1] * 2.0) + (delta[:,2] * 2.0)
    errors = np.square(weighted_delta)
    errors.sort()
    keep_n = int(len(errors) * 0.8)
    if keep_n < 1: keep_n = 1
    
    return np.mean(errors[:keep_n])

def get_hybrid_vertical_cost(edge1, inner1, edge2, inner2):
    """
    Vertical: HYBRID Metric (Error / Correlation).
    
    Logic:
    1. Calculate Pixel Difference (L2 Error).
    2. Calculate Gradient Correlation (Structural Match).
    3. Cost = Error / (1 + Correlation * Weight).
    
    If structure matches (Correlation ~ 0.8), the cost is divided by ~5.
    If structure doesn't match (Correlation ~ 0.0), the cost is full.
    This allows high-error/high-structure matches (Bodies) to beat low-error/low-structure matches (Backgrounds).
    """
    # 1. Pixel Difference (L2)
    delta = np.abs(edge2 - edge1)
    dist = np.mean(np.square((delta[:,0]*1) + (delta[:,1]*2) + (delta[:,2]*2)))
    
    # 2. Gradient Correlation
    g1 = edge1 - inner1
    g2 = inner2 - edge2
    
    g1_flat = g1.reshape(-1) - np.mean(g1)
    g2_flat = g2.reshape(-1) - np.mean(g2)
    
    norm1 = np.linalg.norm(g1_flat)
    norm2 = np.linalg.norm(g2_flat)
    
    correlation = 0.0
    if norm1 > 1e-5 and norm2 > 1e-5:
        correlation = np.dot(g1_flat, g2_flat) / (norm1 * norm2)
        
    # Clip correlation 0..1 (we only care about positive structural matches)
    correlation = max(0.0, correlation)
    
    # 3. Hybrid Score
    # Weight of 10.0 means a perfect correlation makes the match 11x cheaper.
    # This aggressively prioritizes structural continuity.
    score = dist / (1.0 + (correlation * 10.0))
    
    return score

def get_inter_row_cost(row1_lab, row2_lab):
    """ 
    Calculates Vertical Stacking Cost using Sliding Window + Hybrid Metric.
    """
    edge1 = row1_lab[-1, :, :]
    inner1 = row1_lab[-2, :, :] 
    
    edge2 = row2_lab[0, :, :]
    inner2 = row2_lab[1, :, :]
    
    w1 = edge1.shape[0]
    w2 = edge2.shape[0]
    
    # Allow 15% sliding
    search_range = int(min(w1, w2) * 0.15)
    min_overlap = int(min(w1, w2) * 0.8)
    
    best_cost = float('inf')
    
    for offset in range(-search_range, search_range + 1):
        start1 = max(0, -offset)
        end1 = min(w1, w2 - offset)
        start2 = max(0, offset)
        end2 = min(w2, w1 + offset)
        
        len1 = end1 - start1
        if len1 < min_overlap: continue
            
        e1 = edge1[start1:end1]; i1 = inner1[start1:end1]
        e2 = edge2[start2:end2]; i2 = inner2[start2:end2]
        
        cost = get_hybrid_vertical_cost(e1, i1, e2, i2)
        
        if cost < best_cost:
            best_cost = cost
            
    return best_cost

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
# 3. SOLVER LOGIC
# ==========================================
def find_valid_rows(pieces, target_width, pairwise_costs):
    neighbors_map = {}
    for p in pieces:
        candidates = []
        for other in pieces:
            if p.id == other.id: continue
            c = pairwise_costs[p.id][other.id]
            if c < MAX_HORIZONTAL_COST:
                candidates.append((c, other))
        candidates.sort(key=lambda x: x[0])
        neighbors_map[p.id] = candidates[:10]

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

def solve_by_partitioning(valid_rows, target_num_rows, total_pieces):
    row_info = []
    for i, row in enumerate(valid_rows):
        mask = 0
        for p in row: mask |= (1 << p.id)
        # Pre-stitch for speed
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
    
    if not found_partitions: return None, float('inf')

    print(f"  > Vertical Ordering (Brute Force Permutations)...")
    best_layout = None
    min_score = float('inf')
    
    # Iterate all valid groupings of rows
    for partition in found_partitions:
        # Try every vertical order: (Row A, Row B, Row C), (Row A, Row C, Row B)...
        for perm in itertools.permutations(partition):
            score = 0
            
            # Check seams: 0->1, 1->2...
            for k in range(len(perm) - 1):
                top_row = perm[k]
                btm_row = perm[k+1]
                
                # Hybrid Cost (L2 / Correlation)
                cost = get_inter_row_cost(top_row['lab'], btm_row['lab'])
                score += cost
            
            if score < min_score:
                min_score = score
                best_layout = [r['pieces'] for r in perm]

    return best_layout, min_score

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
                costs[i][j] = get_horizontal_cost(pieces[i].lab, 'right', pieces[j].lab, 'left')
            else:
                costs[i][j] = float('inf')

    best_overall_layout = None
    best_overall_score = float('inf')

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
        
        # Keep ample candidates
        top_rows = [x[1] for x in row_scores[:2500]] 
        
        layout, score = solve_by_partitioning(top_rows, r, total_pieces)
        
        if layout:
            print(f"  > Valid Solution Found! Score: {score:.2f}")
            if score < best_overall_score:
                best_overall_score = score
                best_overall_layout = layout

    if best_overall_layout:
        if os.path.exists(DEBUG_DIR): shutil.rmtree(DEBUG_DIR)
        os.makedirs(DEBUG_DIR)
        img = render_full_board(best_overall_layout)
        filename = f"{DEBUG_DIR}FINAL_RESULT.png"
        cv2.imwrite(filename, img)
        print(f"\n>>> SAVED: {filename}")
    else:
        print("\nNo solution found.")

if __name__ == "__main__":
    pieces = extract_pieces(INPUT_IMAGE)
    solve_puzzle(pieces)