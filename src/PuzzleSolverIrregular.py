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
WIDTH_TOLERANCE = 0.01  # 20% to handle uneven row splits

# Strictness for valid row candidates during greedy search
MAX_HORIZONTAL_COST = 5000.0  

# How many permutations to test in the final vertical check
CANDIDATES_PER_ROW = 40

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
def get_mse_cost(img1, img2):
    """ Standard Mean Squared Error """
    diff = img1 - img2
    return np.mean(np.square(diff))

def get_horizontal_cost(p_left, p_right):
    """ Horizontal Pairwise Cost """
    edge1 = p_left.lab[:, -1, :]
    edge2 = p_right.lab[:, 0, :]
    
    # Align height
    h = min(edge1.shape[0], edge2.shape[0])
    return get_mse_cost(edge1[:h], edge2[:h])

def get_vertical_strip_cost(strip_top, strip_bot):
    """ 
    Compare two full stitched strips.
    Slide horizontally to find the best lock.
    """
    # Bottom of Top Strip vs Top of Bottom Strip
    edge1 = strip_top[-1, :, :]
    edge2 = strip_bot[0, :, :]
    
    w1 = edge1.shape[0]
    w2 = edge2.shape[0]
    
    # Small sliding window (allows ~10% shift) to align left edges
    search_range = int(min(w1, w2) * 0.1)
    min_overlap = int(min(w1, w2) * 0.9)
    
    best_mse = float('inf')
    
    for offset in range(-search_range, search_range + 1):
        start1 = max(0, -offset)
        end1 = min(w1, w2 - offset)
        start2 = max(0, offset)
        end2 = min(w2, w1 + offset)
        
        if end1 - start1 < min_overlap: continue
            
        mse = get_mse_cost(edge1[start1:end1], edge2[start2:end2])
        if mse < best_mse:
            best_mse = mse
            
    return best_mse

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
# 3. GREEDY ROW EXTRACTION
# ==========================================
def extract_best_chain(available_pieces, target_width, pairwise_costs):
    """
    Searches for the single best horizontal chain from the current pool.
    Returns: List of PuzzlePiece objects (the chain)
    """
    # Pre-compute neighbors for speed
    neighbors_map = {}
    for p in available_pieces:
        candidates = []
        for other in available_pieces:
            if p.id == other.id: continue
            c = pairwise_costs[p.id][other.id]
            if c < MAX_HORIZONTAL_COST:
                candidates.append((c, other))
        candidates.sort(key=lambda x: x[0])
        neighbors_map[p.id] = candidates[:8]

    valid_chains = []
    
    def dfs(chain, current_w, remaining_ids):
        # Check Width Constraint
        if abs(current_w - target_width) / target_width <= WIDTH_TOLERANCE:
            # Calculate score
            score = 0
            for k in range(len(chain)-1):
                score += pairwise_costs[chain[k].id][chain[k+1].id]
            # Normalize score by length
            avg_score = score / max(1, len(chain)-1)
            valid_chains.append({'pieces': list(chain), 'score': avg_score})
        
        if current_w > target_width * (1 + WIDTH_TOLERANCE): return

        last_p = chain[-1]
        best_next = neighbors_map.get(last_p.id, [])
        
        for cost, next_p in best_next:
            if next_p.id in remaining_ids:
                dfs(chain + [next_p], current_w + next_p.w, remaining_ids - {next_p.id})

    # Start DFS from every piece
    for p in available_pieces:
        rem = {x.id for x in available_pieces if x.id != p.id}
        dfs([p], p.w, rem)
        
    if not valid_chains: return None

    # Sort by Score (Lowest MSE is best)
    valid_chains.sort(key=lambda x: x['score'])
    
    # Return the pieces of the best chain
    return valid_chains[0]['pieces']

# ==========================================
# 4. INTRA-ROW PERMUTATION
# ==========================================
def generate_row_candidates(piece_set):
    """
    Takes a set of pieces.
    Returns Top K permutations sorted by Horizontal Coherence (MSE).
    """
    pieces = list(piece_set)
    valid_candidates = []
    
    # Brute force permutations (N <= 8 is fast)
    for perm in itertools.permutations(pieces):
        total_h_cost = 0
        valid = True
        
        for i in range(len(perm) - 1):
            # Recalculate cost on the fly or pass matrix. 
            # Recalc is fast enough for small N.
            cost = get_horizontal_cost(perm[i], perm[i+1])
            if cost > MAX_HORIZONTAL_COST:
                valid = False
                break
            total_h_cost += cost
            
        if valid:
            # Store data
            lab_strip = stitch_row(perm)
            valid_candidates.append({
                'pieces': perm,
                'h_score': total_h_cost,
                'lab': lab_strip
            })
            
    # Sort by lowest horizontal error
    valid_candidates.sort(key=lambda x: x['h_score'])
    
    # Keep Top K
    return valid_candidates[:CANDIDATES_PER_ROW]

# ==========================================
# 5. GLOBAL STACKING OPTIMIZATION
# ==========================================
def solve_global_stacking(candidate_groups):
    """
    candidate_groups: List of 3 lists. 
                      Group[0] = Top K candidates for Set 1
                      Group[1] = Top K candidates for Set 2...
    """
    print(f"  > Vertical Optimization: Checking stack combinations...")
    
    best_stack = None
    min_total_v_score = float('inf')
    
    # Try all permutations of the Sets (Top/Mid/Bot)
    for bag_perm in itertools.permutations(range(3)):
        cands_top = candidate_groups[bag_perm[0]]
        cands_mid = candidate_groups[bag_perm[1]]
        cands_bot = candidate_groups[bag_perm[2]]
        
        # Iterate all combinations of the chosen candidates
        for r1 in cands_top:
            for r2 in cands_mid:
                # Early Pruning
                cost_1_2 = get_vertical_strip_cost(r1['lab'], r2['lab'])
                if cost_1_2 > min_total_v_score: continue
                
                for r3 in cands_bot:
                    cost_2_3 = get_vertical_strip_cost(r2['lab'], r3['lab'])
                    total_v_cost = cost_1_2 + cost_2_3
                    
                    if total_v_cost < min_total_v_score:
                        min_total_v_score = total_v_cost
                        best_stack = [r1['pieces'], r2['pieces'], r3['pieces']]
                        print(f"    > New Best Stack! V-MSE: {total_v_cost:.2f}")

    return best_stack, min_total_v_score

# ==========================================
# MAIN PIPELINE
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

def solve_puzzle(pieces):
    total_width = sum(p.w for p in pieces)
    total_pieces = len(pieces)
    print(f"Total Width: {total_width}px | Pieces: {total_pieces}")

    # 0. Pre-calc Costs
    costs = np.zeros((total_pieces, total_pieces))
    for i in range(total_pieces):
        for j in range(total_pieces):
            if i != j:
                costs[i][j] = get_horizontal_cost(pieces[i], pieces[j])
            else:
                costs[i][j] = float('inf')

    target_w = total_width / 3.0
    current_pool = list(pieces)
    bag_candidates = []

    # 1. Greedy Row Extraction (Find Sets)
    print("\n--- Phase 1: Extracting Row Sets (Greedy) ---")
    for i in range(3):
        if i == 2:
            # Last row is simply the remainder
            row_set = current_pool
            print(f"  > Set 3 (Remainder): {len(row_set)} pieces")
        else:
            row_set = extract_best_chain(current_pool, target_w, costs)
            if not row_set:
                print("Failed to find valid row chain.")
                return
            
            # Remove from pool
            ids = {p.id for p in row_set}
            current_pool = [p for p in current_pool if p.id not in ids]
            print(f"  > Set {i+1} Found: {len(row_set)} pieces")

        # 2. Intra-Row Permutation (Find Best Arrangements)
        candidates = generate_row_candidates(set(row_set))
        print(f"    - Generated {len(candidates)} valid permutations.")
        bag_candidates.append(candidates)

    # 3. Inter-Row Optimization (Find Best Stack)
    best_layout, score = solve_global_stacking(bag_candidates)

    if best_layout:
        if os.path.exists(DEBUG_DIR): shutil.rmtree(DEBUG_DIR)
        os.makedirs(DEBUG_DIR)
        img = render_full_board(best_layout)
        cv2.imwrite(f"{DEBUG_DIR}FINAL_BEST_RESULT.png", img)
        print(f"\n>>> SAVED BEST: {DEBUG_DIR}FINAL_BEST_RESULT.png")
    else:
        print("\nNo solution found.")

if __name__ == "__main__":
    pieces = extract_pieces(INPUT_IMAGE)
    solve_puzzle(pieces)