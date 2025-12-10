import cv2
import numpy as np

MATCH_CROP = 2          # Remove 2px borders for math (avoids black lines)
MATCH_SIZE = 300  

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
