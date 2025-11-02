import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# Parameter LBP didefinisikan di sini agar konsisten
LBP_IMAGE_SIZE = (64, 64)
LBP_RADIUS = 3
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'

def extract_lbp_features(image):
    """
    Ekstrak fitur LBP dari satu gambar (ROI).
    """
    # Resize
    image_resized = cv2.resize(image, LBP_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    if len(image_resized.shape) == 3:
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image_resized
    
    # Histogram equalization
    image_gray = cv2.equalizeHist(image_gray)
    
    # Extract LBP
    lbp = local_binary_pattern(image_gray, LBP_N_POINTS, 
                               LBP_RADIUS, method=LBP_METHOD)
    
    # Calculate histogram
    n_bins = LBP_N_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    
    # Normalize histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    return hist