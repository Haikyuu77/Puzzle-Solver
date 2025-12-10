import cv2
import numpy as np
import os

from PuzzlePiece import PuzzlePiece
from Animation import animate

# ==========================================
# CONFIGURATION
# ==========================================
# Make sure this path is correct relative to where you run the script
INPUT_IMAGE = '../images/starry_night_rotate.png' 

def extract_pieces(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    img = cv2.imread(image_path)
    if img is None: raise ValueError("Could not load image (check format).")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_pieces = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 1000: continue
        rect = cv2.minAreaRect(cnt)
        
        box = np.int64(cv2.boxPoints(rect))
        pts = box.astype("float32")
        pts = pts[np.argsort(pts[:,1])]
        top, bot = pts[:2], pts[2:]
        top = top[np.argsort(top[:,0])]
        bot = bot[np.argsort(bot[:,0])]
        
        w_rect = int(np.linalg.norm(top[1]-top[0]))
        h_rect = int(np.linalg.norm(bot[0]-top[0]))
        
        src = np.array([top[0], top[1], bot[1], bot[0]], dtype="float32")
        dst = np.array([[0,0], [w_rect-1,0], [w_rect-1,h_rect-1], [0,h_rect-1]], dtype="float32")
        warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w_rect, h_rect))
        
        raw_pieces.append({'img': warped, 'center': rect[0], 'angle': rect[2]})

    if not raw_pieces: return [], img.shape

    # Normalize to Median Dimension (Standardize grid size)
    avg_w = int(np.median([p['img'].shape[1] for p in raw_pieces]))
    avg_h = int(np.median([p['img'].shape[0] for p in raw_pieces]))
    
    print(f"Detected Piece Resolution: {avg_w}x{avg_h} px")
    
    final_pieces = []
    for i, p in enumerate(raw_pieces):
        # Resize to standard grid dimensions to fix minor extraction noise
        resized = cv2.resize(p['img'], (avg_w, avg_h))
        final_pieces.append(PuzzlePiece(img, resized, p['center'], p['angle'], i))
        
    return final_pieces, img.shape

# ==========================================
# 2. ADAPTIVE METRIC
# ==========================================
def analyze_roughness(pieces):
    total_var = sum(p.variance for p in pieces)
    avg = total_var / len(pieces)
    print(f"Image Roughness: {avg:.2f}")
    if avg > 40: return {'grad': 1.5, 'dark': False} # Starry Night
    else:        return {'grad': 0.5, 'dark': True}  # Mona Lisa

def calculate_score(p1, rot1, p2, rot2, relation, config):
    # 1. ASPECT RATIO CHECK
    d1 = p1.rotations[rot1]
    d2 = p2.rotations[rot2]
    
    # Tolerance for size mismatch (in pixels)
    tol = 5 
    
    if relation == 'h':
        # Matching Left-Right: Heights must match
        if abs(d1['h'] - d2['h']) > tol: return float('inf')
    else:
        # Matching Top-Bottom: Widths must match
        if abs(d1['w'] - d2['w']) > tol: return float('inf')

    # 2. METRIC
    if relation == 'h': 
        edge1, inner1 = p1.get_edge_data(rot1, 'right')
        edge2, inner2 = p2.get_edge_data(rot2, 'left')
    else: 
        edge1, inner1 = p1.get_edge_data(rot1, 'bottom')
        edge2, inner2 = p2.get_edge_data(rot2, 'top')

    color_err = np.linalg.norm(edge1 - edge2)
    
    pred_1 = edge1 + (edge1 - inner1)
    pred_2 = edge2 + (edge2 - inner2)
    grad_err = np.linalg.norm(pred_1 - edge2) + np.linalg.norm(pred_2 - edge1)
    
    score = color_err + (config['grad'] * grad_err)
    
    # if config['dark']:
    #     lum = (np.mean(edge1[:,0]) + np.mean(edge2[:,0])) / 2.0
    #     if lum < 15: score *= 3.0
        
    return score

# ==========================================
# 3. GLOBAL SOLVER
# ==========================================
def solve_puzzle(pieces):
    n = len(pieces)
    grid_n = int(np.sqrt(n))
    config = analyze_roughness(pieces)
    
    print("Computing compatibility matrix...")
    costs = np.full((n, 4, n, 4, 2), float('inf'))
    
    for i in range(n):
        for j in range(n):
            if i == j: continue
            for r1 in range(4):
                for r2 in range(4):
                    costs[i][r1][j][r2][0] = calculate_score(pieces[i], r1, pieces[j], r2, 'h', config)
                    costs[i][r1][j][r2][1] = calculate_score(pieces[i], r1, pieces[j], r2, 'v', config)

    print("Running Global Search...")
    best_grid = None
    min_score = float('inf')
    
    # Global Search
    for start_p in range(n):
        for start_r in range(4):
            grid = [-1] * n
            grid[0] = (start_p, start_r)
            used = {start_p}
            total_score = 0
            valid = True
            
            for pos in range(1, n):
                r = pos // grid_n
                c = pos % grid_n
                left = grid[pos-1] if c > 0 else None
                top  = grid[pos-grid_n] if r > 0 else None
                
                best_match = None
                best_s = float('inf')
                
                for cp in range(n):
                    if cp in used: continue
                    for cr in range(4):
                        s = 0
                        cnt = 0
                        if left:
                            s += costs[left[0]][left[1]][cp][cr][0]
                            cnt += 1
                        if top:
                            s += costs[top[0]][top[1]][cp][cr][1]
                            cnt += 1
                        if cnt > 0 and s < best_s:
                            best_s = s
                            best_match = (cp, cr)
                
                if best_match and best_s != float('inf'):
                    grid[pos] = best_match
                    used.add(best_match[0])
                    total_score += best_s
                    if total_score >= min_score: 
                        valid = False
                        break
                else:
                    valid = False
                    break
            
            if valid and total_score < min_score:
                min_score = total_score
                best_grid = list(grid)
                print(f"Best Grid: Start P{start_p} (Err: {int(min_score)})")

    layout = [None] * n
    for i, item in enumerate(best_grid):
        layout[i] = item[0]
        pieces[item[0]].final_rotation_idx = item[1]
    return layout


if __name__ == "__main__":
    try:
        # Check if file exists before starting
        if not os.path.exists(INPUT_IMAGE):
             print(f"ERROR: Image not found at {INPUT_IMAGE}")
             print("Please update the INPUT_IMAGE path at the top of the script.")
        else:
            pieces, _ = extract_pieces(INPUT_IMAGE)
            if pieces:
                print(f"Detected {len(pieces)} pieces.")
                layout = solve_puzzle(pieces)
                # FIX: animate now only expects 2 arguments
                animate(pieces, layout)
            else:
                print("No pieces found.")
    except Exception as e:
        print(f"Error: {e}")