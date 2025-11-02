import logging
import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def overlay_hat(background_frame, face_box, hat_data, eye_coords=None):
    """
    Menempelkan gambar topi ke frame background di atas kotak wajah.
    """
    (x, y, w, h) = face_box
    
    # Ambil gambar dan pengaturan dari hat_data
    if w == 0 or h == 0 or hat_data is None:
        return background_frame

    hat_img = hat_data["image"]
    settings = hat_data["settings"]

    # --- 1. Penskalaan (Scaling) ---
    hat_scale_factor = settings.get("scale_factor", 1.4) # Ambil dari JSON
    new_hat_w = int(w * hat_scale_factor)
    orig_hat_h, orig_hat_w = hat_img.shape[:2]
    new_hat_h = int(orig_hat_h * (new_hat_w / orig_hat_w))
    
    if new_hat_w == 0 or new_hat_h == 0:
        return background_frame
        
    hat_resized = cv2.resize(hat_img, (new_hat_w, new_hat_h), interpolation=cv2.INTER_AREA)

    # --- 2. Rotasi (Opsional) ---
    angle = 0
    if eye_coords and len(eye_coords) == 2:
        eye_left = min(eye_coords, key=lambda e: e[0])
        eye_right = max(eye_coords, key=lambda e: e[0])
        
        dx = eye_right[0] - eye_left[0]
        dy = eye_right[1] - eye_left[1]
        
        if dx != 0:
            angle = np.degrees(np.arctan2(dy, dx))
        angle = np.clip(angle, -25, 25)

    if angle != 0:
        center = (new_hat_w // 2, new_hat_h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        hat_rotated = cv2.warpAffine(hat_resized, M, (new_hat_w, new_hat_h),
                                         flags=cv2.INTER_LINEAR, 
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0, 0))
    else:
        hat_rotated = hat_resized

    y_offset_factor = settings.get("y_offset_factor", 0.8) 
    
    x_offset_factor = settings.get("x_offset_factor", 0.0) 
    
    hat_x1 = int(x + (w // 2) - (new_hat_w // 2) + (x_offset_factor * new_hat_w))
    
    hat_y1 = int(y - (y_offset_factor * new_hat_h))
    hat_x2 = hat_x1 + new_hat_w
    hat_y2 = hat_y1 + new_hat_h
    
    # --- 4. Blending (Alpha) ---
    frame_h, frame_w = background_frame.shape[:2]
    
    frame_y1 = max(hat_y1, 0)
    frame_y2 = min(hat_y2, frame_h)
    frame_x1 = max(hat_x1, 0)
    frame_x2 = min(hat_x2, frame_w)
    
    hat_y1_clip = frame_y1 - hat_y1
    hat_y2_clip = frame_y2 - hat_y1
    hat_x1_clip = frame_x1 - hat_x1
    hat_x2_clip = frame_x2 - hat_x1

    if (hat_x2_clip <= hat_x1_clip) or (hat_y2_clip <= hat_y1_clip):
        return background_frame

    frame_roi = background_frame[frame_y1:frame_y2, frame_x1:frame_x2]
    hat_roi = hat_rotated[hat_y1_clip:hat_y2_clip, hat_x1_clip:hat_x2_clip]

    hat_rgb = hat_roi[..., :3]
    alpha = hat_roi[..., 3] / 255.0
    alpha = np.expand_dims(alpha, axis=2)

    try:
        blended_roi = (1.0 - alpha) * frame_roi.astype(np.float32) + alpha * hat_rgb.astype(np.float32)
        background_frame[frame_y1:frame_y2, frame_x1:frame_x2] = blended_roi.astype(np.uint8)
    except ValueError as e:
        logger.warning(f"Error blending: {e}. Shapes: frame_roi={frame_roi.shape}, hat_roi={hat_roi.shape}, alpha={alpha.shape}")
    
    return background_frame