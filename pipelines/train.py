import logging
import joblib
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

from .dataset import load_dataset_from_dirs
from .features import extract_lbp_features

logger = logging.getLogger(__name__)

def augment_image(image):
    """Membuat versi gambar yang diaugmentasi."""
    augmented = [image]
    augmented.append(cv2.flip(image, 1)) # Flip
    
    for angle in [-10, 5, 10]: # Rotasi
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        augmented.append(rotated)
    return augmented

def process_paths_to_features(paths, labels, augment=False):
    """Mengekstrak fitur LBP dari daftar path gambar."""
    features_list = []
    labels_list = []
    
    for path, label in tqdm(zip(paths, labels), total=len(paths)):
        img = cv2.imread(str(path))
        if img is None:
            logger.warning(f"Could not read image {path}, skipping.")
            continue
            
        if augment:
            augmented_imgs = augment_image(img)
        else:
            augmented_imgs = [img]
            
        for aug_img in augmented_imgs:
            features = extract_lbp_features(aug_img)
            features_list.append(features)
            labels_list.append(label)
            
    return np.array(features_list), np.array(labels_list)

def train_pipeline_lbp(args):
    """
    Orkestrasi pipeline training LBP.
    """
    
    # 1. Muat Path Dataset
    X_train_paths, X_test_paths, y_train_labels, y_test_labels = load_dataset_from_dirs(
        args.pos_dir, args.neg_dir, args.test_size
    )

    # 2. Ekstrak Fitur LBP untuk data Training
    logger.info("Extracting LBP features for training data...")
    X_train_data, y_train_data = process_paths_to_features(
        X_train_paths, y_train_labels, args.augment
    )
    logger.info(f"Training data shape: {X_train_data.shape}")

    # 3. Ekstrak Fitur LBP untuk data Test
    logger.info("Extracting LBP features for test data...")
    X_test_data, y_test_data = process_paths_to_features(
        X_test_paths, y_test_labels, augment=False
    )
    logger.info(f"Test data shape: {X_test_data.shape}")
    
    if len(X_train_data) == 0:
        logger.error("No training features extracted. Check your dataset.")
        return

    # 4. Latih Classifier
    logger.info(f"Training {args.classifier.upper()} classifier...")
    param_grid = {'C': [0.01, 0.1, 1.0, 10.0]}
    base_model = LinearSVC(max_iter=20000, dual="auto", class_weight='balanced', random_state=42)
 

    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train_data, y_train_data)
    
    best_model = grid_search.best_estimator_
    logger.info(f"Best params found: {grid_search.best_params_}")

    # 5. Evaluasi pada Test Set
    logger.info("Evaluating on test set...")
    y_pred = best_model.predict(X_test_data)
    print("\n" + "="*30 + " TEST SET REPORT " + "="*30)
    print(classification_report(y_test_data, y_pred, target_names=['Non-Face', 'Face']))
    print("="*80)

    # 6. Simpan Model dan Data Tes
    model_path = args.model_dir / "svm_lbp.pkl"
    joblib.dump(best_model, model_path)
    logger.info(f"Trained model saved to: {model_path}")
    
    # Simpan data tes untuk 'app.py eval'
    test_data = {"X": X_test_data, "y": y_test_data}
    joblib.dump(test_data, args.model_dir / "test_data.pkl")
    logger.info(f"Test data saved to {args.model_dir / 'test_data.pkl'}")