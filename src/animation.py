import numpy as np
import cv2

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
    margin = 50
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