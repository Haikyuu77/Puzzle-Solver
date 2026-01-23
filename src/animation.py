import numpy as np
import cv2

def animate(pieces, layout):
    # Flip the rendered view 180° without changing piece math
    FLIP_VIEW_180 = True
    # Scale up visual size of pieces/canvas for display (animation only)
    DISPLAY_SCALE = 1
    # Seam fill (target only near-black gaps instead of closing everything)
    SEAM_FILL = True
    SEAM_THRESH = 8  # pixel values below this are treated as gaps
    # Optional morph close applied only on dark seam regions
    USE_MORPH_CLOSE = False
    MORPH_KERNEL = 2
    MORPH_MASK_THRESH = 8
    MORPH_DILATE = 1
    
    # Determine Grid Size
    n = len(pieces)
    gw = int(np.sqrt(n))
    
    # Get dimensions of a piece in the solved state
    p0 = pieces[layout[0]]
    ref_img = p0.rotations[p0.final_rotation_idx]['visual']
    piece_h, piece_w = ref_img.shape[:2]
    
    # Calculate Final Canvas Size based on scaled pieces and grid
    scaled_piece_w = int(piece_w * DISPLAY_SCALE)
    scaled_piece_h = int(piece_h * DISPLAY_SCALE)
    final_w = scaled_piece_w * gw
    final_h = scaled_piece_h * gw
    
    # Add a visual margin
    margin = int(200 * DISPLAY_SCALE)
    canvas_w = final_w + (margin * 2)
    canvas_h = final_h + (margin * 2)
    
    # Calculate Targets
    grid_start_x = margin
    grid_start_y = margin
    
    for i, idx in enumerate(layout):
        r, c = i // gw, i % gw
        pieces[idx].target_x = grid_start_x + (c * scaled_piece_w) + scaled_piece_w/2
        pieces[idx].target_y = grid_start_y + (r * scaled_piece_h) + scaled_piece_h/2

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
            # Snap to exact targets on the final frame to avoid subpixel gaps
            if ease_val >= 0.999:
                cx, cy = float(p.target_x), float(p.target_y)
            
            # Compensate the typical -90° bias from minAreaRect so starting orientation matches the source photo
            start_angle = p.detected_angle
            end_angle = p.final_rotation_idx * -90
            cur_angle = start_angle + (end_angle - start_angle) * ease_val
            
            # Pad to reduce edge artifacts during rotation
            src_img = cv2.copyMakeBorder(
                p.rotations[0]['visual'], 1, 1, 1, 1, cv2.BORDER_REPLICATE
            )
            if DISPLAY_SCALE != 1.0:
                src_img = cv2.resize(src_img, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_CUBIC)
            h, w = src_img.shape[:2]
            center = (w // 2, h // 2)
            
            M = cv2.getRotationMatrix2D(center, cur_angle, 1.0)
            abs_cos, abs_sin = abs(M[0,0]), abs(M[0,1])
            new_w = int(h * abs_sin + w * abs_cos)
            new_h = int(h * abs_cos + w * abs_sin)
            M[0, 2] += new_w / 2 - center[0]
            M[1, 2] += new_h / 2 - center[1]
            
            # Only replicate borders once fully placed; keep defaults during motion
            border_mode = cv2.BORDER_REPLICATE if ease_val >= 1.0 else cv2.BORDER_CONSTANT
            rot_img = cv2.warpAffine(
                src_img, M, (new_w, new_h),
                flags=cv2.INTER_CUBIC,  # smoother edges
                borderMode=border_mode
            )
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
        # MAX_DISPLAY = 1000
        # if disp_h > MAX_DISPLAY:
        #     scale = MAX_DISPLAY / disp_h
        #     return cv2.resize(frame, (int(disp_w * scale), MAX_DISPLAY))
        
        if ease_val >= 1.0:
            # Build masks for seam/morph operations
            seam_mask = (frame[:,:,0] < SEAM_THRESH) & (frame[:,:,1] < SEAM_THRESH) & (frame[:,:,2] < SEAM_THRESH)
            morph_mask = (frame[:,:,0] < MORPH_MASK_THRESH) & (frame[:,:,1] < MORPH_MASK_THRESH) & (frame[:,:,2] < MORPH_MASK_THRESH)
            if MORPH_DILATE > 0 and np.any(morph_mask):
                k = np.ones((MORPH_DILATE, MORPH_DILATE), np.uint8)
                morph_mask = cv2.dilate(morph_mask.astype(np.uint8), k, iterations=1).astype(bool)

            if SEAM_FILL and np.any(seam_mask):
                blurred = cv2.blur(frame, (3, 3))
                frame[seam_mask] = blurred[seam_mask]

            if USE_MORPH_CLOSE and np.any(morph_mask):
                kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
                closed = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations=1)
                frame[morph_mask] = closed[morph_mask]

        if FLIP_VIEW_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
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
