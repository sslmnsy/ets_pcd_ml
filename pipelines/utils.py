import logging
import sys
import cv2
import numpy as np
import json
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_logging():
    """Konfigurasi logging standar."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def resize_to_fixed(image, target_height):
    """Resize gambar sambil mempertahankan rasio aspek."""
    try:
        h, w = image.shape[:2]
        if h == 0: 
            return image
        ratio = target_height / h
        new_w = int(w * ratio)
        return cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_AREA)
    except Exception as e:
        logging.warning(f"Error resizing image: {e}")
        return image


def load_hat_image(hat_path: Path):
    """Memuat gambar topi dengan alpha channel (BGRA)."""
    hat_img = cv2.imread(str(hat_path), cv2.IMREAD_UNCHANGED)
    if hat_img is None:
        logger.error(f"Could not load hat image from {hat_path}")
        return None
    
    # Pastikan gambar punya 4 channel (BGRA)
    if hat_img.shape[2] != 4:
        logger.warning(f"Hat image {hat_path} does not have alpha channel. Creating one.")
        b, g, r = cv2.split(hat_img)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        hat_img = cv2.merge((b, g, r, alpha))
    
    return hat_img

def load_hat_data(hat_path: Path):
    """
    Memuat gambar topi DAN file .json metadatanya.
    Mengembalikan dict berisi gambar dan pengaturannya.
    """
    hat_image = load_hat_image(hat_path)
    if hat_image is None:
        return None

    # Cari file .json yang sesuai
    meta_path = hat_path.with_suffix('.json')
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    else:
        # Fallback default jika tidak ada .json
        logger.warning(f"Tidak ada file metadata {meta_path}. Menggunakan nilai default.")
        meta = {"scale_factor": 1.4, "y_offset_factor": 0.8}
    
    return {
        "image": hat_image,
        "settings": meta,
        "name": hat_path.stem 
    }