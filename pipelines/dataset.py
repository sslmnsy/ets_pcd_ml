import logging
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def load_dataset_from_dirs(pos_dir: Path, neg_dir: Path, test_size: float):
    """
    Memuat path gambar dan label dari direktori positif dan negatif.
    Mencari secara rekursif dan menghapus duplikat.
    """

    def get_image_paths(directory: Path):
        """Helper untuk mengambil semua ekstensi gambar umum secara rekursif."""
        paths_with_duplicates = []
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        for ext in extensions:
            paths_with_duplicates.extend(directory.rglob(ext))
        # Hapus duplikat yang mungkin muncul karena case-insensitivity (misal .jpg dan .JPG)
        return list(set(paths_with_duplicates))

    logger.info(f"Loading positive samples from: {pos_dir}")
    pos_paths = get_image_paths(pos_dir)
    pos_labels = [1] * len(pos_paths)
    logger.info(f"Found {len(pos_paths)} positive samples.")

    logger.info(f"Loading negative samples from: {neg_dir}")
    neg_paths = get_image_paths(neg_dir)
    neg_labels = [0] * len(neg_paths)
    logger.info(f"Found {len(neg_paths)} negative samples.")

    if len(pos_paths) == 0 or len(neg_paths) == 0:
        logger.error("No images found in one or both directories. Please check paths.")
        logger.error(f"Positive path: {pos_dir.resolve()}")
        logger.error(f"Negative path: {neg_dir.resolve()}")
        raise FileNotFoundError("Dataset images not found.")

    # Gabungkan dan bagi
    X = np.array(pos_paths + neg_paths)
    y = np.array(pos_labels + neg_labels)

    # Acak data sebelum split
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Split data (train + val) dan test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    logger.info(f"Dataset split: {len(y_train_val)} train/val samples, {len(y_test)} test samples.")
    
    return X_train_val, X_test, y_train_val, y_test