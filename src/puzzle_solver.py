import cv2
import numpy as np
import math

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_IMAGE = '../images/mona_lisa_translate.png'
GRID_SIZE = 4            # 4x4 = 16 pieces
ANIMATION_SPEED = 0.05
WINDOW_SIZE = (800, 800)
CROP_PIXELS = 4          # CRITICAL: Removes black artifacts from edges

class PuzzlePiece:
    def __init__(self, original_img, straight_img, center, angle):
        self.original_view = original_img
        # Crop the image to remove black interpolation artifacts
        h, w = straight_img.shape[:2]
        self.image = straight_img[CROP_PIXELS:h-CROP_PIXELS, CROP_PIXELS:w-CROP_PIXELS]
        
        # Resize to standard block size (e.g., 100x100) for consistent matching
        self.image = cv2.resize(self.image, (100, 100))
        
        self.id = -1
        self.current_x = center[0]
        self.current_y = center[1]
        self.current_angle = angle
        self.target_x = 0
        self.target_y = 0
        self.h, self.w = self.image.shape[:2]

        # Pre-compute LAB color space for better human-eye matching
        self.lab_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB).astype("float32")

    def get_edge_data(self, side):
        """
        Returns the outer edge pixels AND the inner neighbor pixels 
        to calculate gradients.
        """
        if side == 'top':    
            return self.lab_image[0, :, :], self.lab_image[1, :, :]
        if side == 'bottom': 
            return self.lab_image[-1, :, :], self.lab_image[-2, :, :]
        if side == 'left':   
            return self.lab_image[:, 0, :], self.lab_image[:, 1, :]
        if side == 'right':  
            return self.lab_image[:, -1, :], self.lab_image[:, -2, :]

# ==========================================
# 1. PRE-PROCESSING
# ==========================================
def extract_pieces(image_path):
    img = cv2.imread(image_path)
    if img is None: raise FileNotFoundError("Image not found")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pieces = []
    print(f"Detecting pieces... Found {len(contours)} contours.")
    
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 1000: continue
        
        # De-rotate
        rect = cv2.minAreaRect(cnt)
        box = np.int64(cv2.boxPoints(rect))
        
        # Sort corners: TL, TR, BR, BL
        pts = box.astype("float32")
        pts = pts[np.argsort(pts[:, 1])] # Sort by Y
        top, bottom = pts[:2], pts[2:]
        top = top[np.argsort(top[:, 0])] # Sort by X
        bottom = bottom[np.argsort(bottom[:, 0])]
        
        # Warp
        w_rect, h_rect = int(rect[1][0]), int(rect[1][1])
        if w_rect < h_rect: w_rect, h_rect = h_rect, w_rect # Ensure landscape logic
        
        dst_pts = np.array([[0,0], [w_rect-1, 0], [w_rect-1, h_rect-1], [0, h_rect-1]], dtype="float32")
        src_pts = np.array([top[0], top[1], bottom[1], bottom[0]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, (w_rect, h_rect))
        
        p = PuzzlePiece(img, warped, rect[0], rect[2])
        p.id = i
        pieces.append(p)

    return pieces, img.shape

# ==========================================
# 2. ADVANCED MATCHING METRIC
# ==========================================
def calculate_match_score(p1, p2, side_p1):
    """
    Calculates score between P1 and P2. Lower is better.
    side_p1: 'right' means we compare P1's Right to P2's Left.
    """
    if side_p1 == 'right':
        edge1, inner1 = p1.get_edge_data('right')
        edge2, inner2 = p2.get_edge_data('left')
    elif side_p1 == 'bottom':
        edge1, inner1 = p1.get_edge_data('bottom')
        edge2, inner2 = p2.get_edge_data('top')
    
    # 1. Color Difference (Euclidean distance in LAB)
    # We compare the average of the edge and the inner pixel to smooth noise
    val1 = (edge1 + inner1) / 2.0
    val2 = (edge2 + inner2) / 2.0
    
    diff = np.linalg.norm(val1 - val2, axis=1)
    color_score = np.mean(diff)
    
    # 2. Penalize matching dark edges to dark edges? 
    # (Optional: prevents the "black border" trap if crop didn't work)
    # If both edges are very dark (L < 10), add penalty
    # avg_L = (np.mean(val1[:,0]) + np.mean(val2[:,0])) / 2
    # if avg_L < 20: color_score += 50 
    
    return color_score

def solve_puzzle_robust(pieces):
    n = len(pieces)
    grid_n = int(np.sqrt(n))
    
    # Precompute all pairwise scores
    # [i][j] = Score if J is to the RIGHT of I
    right_scores = np.full((n, n), np.inf)
    # [i][j] = Score if J is BELOW I
    down_scores = np.full((n, n), np.inf)
    
    print("Calculating compatibility matrix...")
    for i in range(n):
        for j in range(n):
            if i == j: continue
            right_scores[i][j] = calculate_match_score(pieces[i], pieces[j], 'right')
            down_scores[i][j] = calculate_match_score(pieces[i], pieces[j], 'bottom')
            
    best_layout = None
    min_global_error = float('inf')
    
    # Try creating a grid starting with EVERY piece as the top-left corner
    # The real solution will have the lowest Total Error Score
    for start_node in range(n):
        used = {start_node}
        layout = [-1] * n
        layout[0] = start_node
        current_error = 0
        
        valid = True
        
        # Fill grid raster-style
        for pos in range(1, n):
            row = pos // grid_n
            col = pos % grid_n
            
            best_next = -1
            best_local_score = float('inf')
            
            left_neighbor = layout[pos - 1] if col > 0 else -1
            top_neighbor  = layout[pos - grid_n] if row > 0 else -1
            
            # Check all unused pieces for this slot
            for candidate in range(n):
                if candidate in used: continue
                
                score = 0
                div = 0
                
                if left_neighbor != -1:
                    score += right_scores[left_neighbor][candidate]
                    div += 1
                if top_neighbor != -1:
                    score += down_scores[top_neighbor][candidate]
                    div += 1
                
                if div > 0:
                    avg = score / div
                    if avg < best_local_score:
                        best_local_score = avg
                        best_next = candidate
            
            if best_next != -1:
                layout[pos] = best_next
                used.add(best_next)
                current_error += best_local_score
            else:
                valid = False
                break
        
        if valid and current_error < min_global_error:
            min_global_error = current_error
            best_layout = layout
            print(f"New best layout found (Error: {int(current_error)})")
            
    return best_layout

# ==========================================
# 3. ANIMATION
# ==========================================
def animate(pieces, layout, canvas_shape):
    grid_n = int(np.sqrt(len(pieces)))
    p_size = pieces[0].w # Assuming square/consistent
    
    # Setup target coordinates
    margin_x = (canvas_shape[1] - (p_size * grid_n)) // 2
    margin_y = (canvas_shape[0] - (p_size * grid_n)) // 2
    
    for i, idx in enumerate(layout):
        r, c = i // grid_n, i % grid_n
        pieces[idx].target_x = margin_x + (c * p_size) + p_size/2
        pieces[idx].target_y = margin_y + (r * p_size) + p_size/2
    
    # Loop
    base_canvas = np.zeros((canvas_shape[0], canvas_shape[1], 3), dtype=np.uint8)
    frames = 60
    
    for t in range(frames + 30):
        frame = base_canvas.copy()
        alpha = min(t/frames, 1.0)
        # Ease out cubic
        ease = 1 - pow(1 - alpha, 3)
        
        for p in pieces:
            # Interpolate
            cx = p.current_x + (p.target_x - p.current_x) * ease
            cy = p.current_y + (p.target_y - p.current_y) * ease
            
            # Simple rotation visualization (fading rotation)
            # To do real rotation, we need getRotationMatrix2D
            M = cv2.getRotationMatrix2D((p.w/2, p.h/2), p.current_angle * (1-ease), 1.0)
            cur_img = cv2.warpAffine(p.image, M, (p.w, p.h))
            
            tl_x = int(cx - p.w/2)
            tl_y = int(cy - p.h/2)
            
            # Boundary check
            h_img, w_img = cur_img.shape[:2]
            if tl_y >= 0 and tl_x >= 0 and tl_y+h_img < canvas_shape[0] and tl_x+w_img < canvas_shape[1]:
                frame[tl_y:tl_y+h_img, tl_x:tl_x+w_img] = cur_img

        cv2.imshow("Solved", cv2.resize(frame, WINDOW_SIZE))
        cv2.waitKey(20)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        pieces, shape = extract_pieces(INPUT_IMAGE)
        if len(pieces) >= 4:
            solution = solve_puzzle_robust(pieces)
            if solution:
                animate(pieces, solution, (shape[0]*2, shape[1]*2))
            else:
                print("Could not find solution.")
    except Exception as e:
        print(f"Error: {e}")