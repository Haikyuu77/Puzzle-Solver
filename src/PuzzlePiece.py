import cv2
import numpy as np

MATCH_CROP = 2          # Remove 2px borders for math (avoids black lines)

class PuzzlePiece:
    def __init__(self, original_img, straight_img, center, angle, idx, visual_img=None):
        self.id = idx
        self.original_view = original_img
        self.center = center
        self.detected_angle = angle
        
        # Math image (resized for solver) vs visual image (original warped for animation)
        math_base = straight_img
        visual_base = visual_img if visual_img is not None else straight_img

        def smooth_if_jagged(img):
            """
            Detect jagged borders; replace only the jagged sides with adjacent inner pixels.
            Runs border=1 twice, then border=2 twice to catch stubborn artifacts.
            """
            def measure_and_fix(im, border):
                if im.shape[0] <= 2 * border or im.shape[1] <= 2 * border:
                    return im, False
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                top = gray[:border, :]
                bottom = gray[-border:, :]
                left = gray[:, :border]
                right = gray[:, -border:]
                inner = gray[border:-border, border:-border]
                # Stricter thresholds to catch subtle jagged edges
                thr_std = 8
                thr_mean = 3
                jag_top = top.std() > thr_std and abs(top.mean() - inner.mean()) > thr_mean
                jag_bot = bottom.std() > thr_std and abs(bottom.mean() - inner.mean()) > thr_mean
                jag_left = left.std() > thr_std and abs(left.mean() - inner.mean()) > thr_mean
                jag_right = right.std() > thr_std and abs(right.mean() - inner.mean()) > thr_mean
                if not (jag_top or jag_bot or jag_left or jag_right):
                    return im, False
                out = im.copy()
                # Replace only jagged sides with adjacent inner pixels (keeps size)
                if jag_top:
                    out[:border, :] = np.repeat(out[border:border+1, :], border, axis=0)
                if jag_bot:
                    out[-border:, :] = np.repeat(out[-border-1:-border, :], border, axis=0)
                if jag_left:
                    out[:, :border] = np.repeat(out[:, border:border+1], border, axis=1)
                if jag_right:
                    out[:, -border:] = np.repeat(out[:, -border-1:-border], border, axis=1)
                return out, True

            out, _ = measure_and_fix(img, 1)
            out, _ = measure_and_fix(out, 1)
            out, _ = measure_and_fix(out, 2)
            out, _ = measure_and_fix(out, 2)
            return out

        visual_base = smooth_if_jagged(visual_base)
        
        # 1. GENERATE 4 ORIENTATIONS (0, 90, 180, 270)
        self.rotations = [] 
        
        current_math = math_base
        current_visual = visual_base
        for r in range(4):
            # Dimensions for math (kept from resized version to stabilize solver)
            h, w = current_math.shape[:2]
            
            # Prepare Math Image (LAB Color Space)
            # We resize a copy to small square for consistent math
            # but we keep 'visual' as the original high-res shape
            safe = current_math[MATCH_CROP:h-MATCH_CROP, MATCH_CROP:w-MATCH_CROP]
            if safe.size == 0: safe = current_math # Fallback for tiny pieces
            
            # small = cv2.resize(safe, (MATCH_SIZE, MATCH_SIZE))
            lab = cv2.cvtColor(safe, cv2.COLOR_BGR2LAB).astype("float32")
            
            self.rotations.append({
                'visual': current_visual, # High-Res Display (original warped size)
                'lab': lab,            # Low-Res Math
                'h': h,                
                'w': w
            })
            
            # Rotate 90 degrees for next iteration
            current_math = cv2.rotate(current_math, cv2.ROTATE_90_CLOCKWISE)
            current_visual = cv2.rotate(current_visual, cv2.ROTATE_90_CLOCKWISE)

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
