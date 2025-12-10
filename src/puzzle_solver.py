import cv2
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Make sure this path is correct relative to where you run the script
INPUT_IMAGE = '../images/starry_night_rotate.png' 

# Settings
MATCH_CROP = 2          # Remove 2px borders for math (avoids black lines)
MATCH_SIZE = 64         # Fixed size for mathematical comparison only
DISPLAY_CROP = 0        # 0 = Show full original piece, 1 = Trim edge

class PuzzlePiece:
    def __init__(self, original_img, straight_img, center, angle, idx):
        self.id = idx
        self.original_view = original_img
        self.center = center
        self.detected_angle = angle
        
        # Base straightened image (Original Resolution)
        base = straight_img
        
        # 1. GENERATE 4 ORIENTATIONS (0, 90, 180, 270)
        self.rotations = [] 
        
        current_img = base
        for r in range(4):
            # Dimensions of this specific rotation
            h, w = current_img.shape[:2]
            
            # Prepare Math Image (LAB Color Space)
            # We resize a copy to small square for consistent math
            # but we keep 'visual' as the original high-res shape
            safe = current_img[MATCH_CROP:h-MATCH_CROP, MATCH_CROP:w-MATCH_CROP]
            if safe.size == 0: safe = current_img # Fallback for tiny pieces
            
            small = cv2.resize(safe, (MATCH_SIZE, MATCH_SIZE))
            lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype("float32")
            
            self.rotations.append({
                'visual': current_img, # High-Res Display
                'lab': lab,            # Low-Res Math
                'h': h,                
                'w': w
            })
            
            # Rotate 90 degrees for next iteration
            current_img = cv2.rotate(current_img, cv2.ROTATE_90_CLOCKWISE)

        # Calculate Variance on the base image
        self.variance = np.std(self.rotations[0]['lab'][:,:,0])
        
        # Animation State
        self.final_rotation_idx = 0 
        self.current_x = center[0]
        self.current_y = center[1]
        self.target_x = 0
        self.target_y = 0

    def get_edge_data(self, rot_idx, side):
        lab = self.rotations[rot_idx]['lab']
        if side == 'top':    return lab[0,:,:], lab[1,:,:]
        if side == 'bottom': return lab[-1,:,:], lab[-2,:,:]
        if side == 'left':   return lab[:,0,:], lab[:,1,:]
        if side == 'right':  return lab[:,-1,:], lab[:,-2,:]

# ==========================================
# 1. EXTRACT PIECES (PRESERVE SIZE)
# ==========================================
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
    
    if config['dark']:
        lum = (np.mean(edge1[:,0]) + np.mean(edge2[:,0])) / 2.0
        if lum < 15: score *= 3.0
        
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

# ==========================================
# 4. ANIMATION (With 2-Second Intro)
# ==========================================
# FIX: Removed unused 'canvas_shape' parameter
def animate(pieces, layout):
    # Determine Grid Size
    n = len(pieces)
    gw = int(np.sqrt(n))
    
    # Get dimensions of a piece in the solved state
    p0 = pieces[layout[0]]
    ref_img = p0.rotations[p0.final_rotation_idx]['visual']
    piece_h, piece_w = ref_img.shape[:2]
    
    # Calculate Final Canvas Size
    final_w = piece_w * gw
    final_h = piece_h * gw
    
    # Add a visual margin
    margin = 100
    canvas_w = final_w + (margin * 2)
    canvas_h = final_h + (margin * 2)
    
    # Calculate Targets
    grid_start_x = margin
    grid_start_y = margin
    
    for i, idx in enumerate(layout):
        r, c = i // gw, i % gw
        pieces[idx].target_x = grid_start_x + (c * piece_w) + piece_w/2
        pieces[idx].target_y = grid_start_y + (r * piece_h) + piece_h/2

    # Shift Start Positions to Center (so the pile isn't off-screen)
    all_x = [p.current_x for p in pieces]
    all_y = [p.current_y for p in pieces]
    center_x = (min(all_x) + max(all_x)) / 2
    center_y = (min(all_y) + max(all_y)) / 2
    shift_x = (canvas_w / 2) - center_x
    shift_y = (canvas_h / 2) - center_y
    for p in pieces:
        p.current_x += shift_x
        p.current_y += shift_y

    base_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # --- HELPER: DRAW FRAME ---
    def draw_state(ease_val):
        frame = base_canvas.copy()
        for p in pieces:
            # Interpolate
            cx = p.current_x + (p.target_x - p.current_x) * ease_val
            cy = p.current_y + (p.target_y - p.current_y) * ease_val
            
            start_angle = p.detected_angle
            end_angle = p.final_rotation_idx * -90
            cur_angle = start_angle + (end_angle - start_angle) * ease_val
            
            src_img = p.rotations[0]['visual']
            h, w = src_img.shape[:2]
            center = (w // 2, h // 2)
            
            M = cv2.getRotationMatrix2D(center, cur_angle, 1.0)
            abs_cos, abs_sin = abs(M[0,0]), abs(M[0,1])
            new_w = int(h * abs_sin + w * abs_cos)
            new_h = int(h * abs_cos + w * abs_sin)
            M[0, 2] += new_w / 2 - center[0]
            M[1, 2] += new_h / 2 - center[1]
            
            rot_img = cv2.warpAffine(src_img, M, (new_w, new_h))
            
            tl_x = int(cx - new_w / 2)
            tl_y = int(cy - new_h / 2)
            
            ih, iw = rot_img.shape[:2]
            x1 = max(tl_x, 0)
            y1 = max(tl_y, 0)
            x2 = min(tl_x + iw, canvas_w)
            y2 = min(tl_y + ih, canvas_h)
            
            if x2 > x1 and y2 > y1:
                ix1 = x1 - tl_x
                iy1 = y1 - tl_y
                ix2 = ix1 + (x2 - x1)
                iy2 = iy1 + (y2 - y1)
                
                # Check bounds for source image slicing
                if ix2 <= iw and iy2 <= ih:
                    frame[y1:y2, x1:x2] = rot_img[iy1:iy2, ix1:ix2]
        
        # Smart Scaling for Display
        disp_h, disp_w = frame.shape[:2]
        MAX_DISPLAY = 900
        if disp_h > MAX_DISPLAY:
            scale = MAX_DISPLAY / disp_h
            return cv2.resize(frame, (int(disp_w * scale), MAX_DISPLAY))
        return frame

    # --- STEP 1: SHOW ORIGINAL STATE (2 SECONDS) ---
    print("Displaying original state...")
    initial_view = draw_state(0.0) # Ease = 0 means original position
    cv2.imshow("Puzzle Solver", initial_view)
    cv2.waitKey(2000) # Freeze for 2000ms

    # --- STEP 2: ANIMATE ---
    print("Animating solution...")
    total_frames = 300 
    
    for t in range(total_frames):
        alpha = min(t / (total_frames - 40), 1.0)
        ease = 1 - pow(1 - alpha, 3)
        
        view = draw_state(ease)
        cv2.imshow("Puzzle Solver", view)
        cv2.waitKey(5)
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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