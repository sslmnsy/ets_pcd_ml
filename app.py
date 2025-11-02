import argparse
import logging
import sys
from pathlib import Path

import cv2
import joblib  
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, PrecisionRecallDisplay,
                             RocCurveDisplay, ConfusionMatrixDisplay)

from pipelines.train import train_pipeline_lbp
from pipelines.infer import InferencePipelineLBP  
from pipelines.utils import setup_logging

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="LBP+SVM Face Detector with Hat Overlay.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Perintah Train
    p_train = subparsers.add_parser("train", help="Train the LBP+SVM/RF classifier.")
    p_train.add_argument("--pos_dir", type=Path, required=True, help="Directory of positive face crops.")
    p_train.add_argument("--neg_dir", type=Path, required=True, help="Directory of negative non-face images.")
    p_train.add_argument("--test_size", type=float, default=0.2, help="Fraction of data to use for testing.")
    p_train.add_argument("--augment", action='store_true', help="Enable image augmentation for training.")
    p_train.add_argument("--classifier", type=str, choices=['svm', 'rf'], 
                         default='svm', help="Tipe classifier (svm atau rf).")
    p_train.add_argument("--model_dir", type=Path, default=Path("models"), help="Directory to save models.")
    
    # 2. Perintah Eval
    p_eval = subparsers.add_parser("eval", help="Evaluate the trained model on the test set.")
    p_eval.add_argument("--model_dir", type=Path, default=Path("models"), help="Directory to load models from.")
    p_eval.add_argument("--model_name", type=str, default="svm_lbp.pkl", help="Name of the model file.")
    
    # 3. Perintah Infer
    p_infer = subparsers.add_parser("infer", help="Run inference on a single image.")
    p_infer.add_argument("--image", type=Path, required=True, help="Path to input image.")
    p_infer.add_argument("--out", type=Path, required=True, help="Path to save output image.")
    p_infer.add_argument("--model_dir", type=Path, default=Path("models"), help="Directory to load models from.")
    p_infer.add_argument("--model_name", type=str, default="svm_lbp.pkl", help="Name of the model file.")
    # Default hat path sekarang menunjuk ke folder baru
    p_infer.add_argument("--hat", type=Path, default=Path("assets/hats/top_hat.png"), help="Path to hat PNG.")
    
    # 4. Perintah Webcam
    p_webcam = subparsers.add_parser("webcam", help="Run real-time inference with webcam.")
    p_webcam.add_argument("--camera", type=int, default=0, help="Camera ID to use.")
    p_webcam.add_argument("--model_dir", type=Path, default=Path("models"), help="Directory to load models from.")
    p_webcam.add_argument("--model_name", type=str, default="svm_lbp.pkl", help="Name of the model file.")
    # Default hat path sekarang menunjuk ke folder baru
    p_webcam.add_argument("--hat", type=Path, default=Path("assets/hats/top_hat.png"), help="Path to hat PNG.")

    args = parser.parse_args()
    setup_logging()
    
    # Setel nama model default berdasarkan classifier jika sedang training
    if args.command == 'train':
        args.model_name = f"{args.classifier}_lbp.pkl"
    
    # Jika tidak train, pastikan model_name disetel
    if not hasattr(args, 'model_name'):
         args.model_name = "svm_lbp.pkl"

    try:
        if args.command == "train":
            logger.info(f"Starting LBP training pipeline with args: {args}")
            args.model_dir.mkdir(parents=True, exist_ok=True)
            train_pipeline_lbp(args)
            logger.info(f"Training complete. Models saved to '{args.model_dir}'.")

        elif args.command == "eval":
            logger.info(f"Starting evaluation on {args.model_name}...")
            
            # Muat model dan data tes
            test_data_path = args.model_dir / "test_data.pkl"
            if not test_data_path.exists():
                logger.error("test_data.pkl not found. Run training first.")
                sys.exit(1)
                
            test_data = joblib.load(test_data_path)
            X_test, y_test = test_data['X'], test_data['y']
            
            # Muat model untuk memprediksi
            pipeline = InferencePipelineLBP(args.model_dir, model_name=args.model_name)
            y_pred = pipeline.model.predict(X_test)
            
            if hasattr(pipeline.model, "decision_function"):
                y_score = pipeline.model.decision_function(X_test)
            elif hasattr(pipeline.model, "predict_proba"):
                y_score = pipeline.model.predict_proba(X_test)[:, 1]
            else:
                y_score = y_pred

            print("\n" + "="*30 + " EVALUATION REPORT " + "="*30)
            print(classification_report(y_test, y_pred, target_names=["non-face", "face"]))
            print("="*80)
            
            # Plotting (dengan Confusion Matrix)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
            
            RocCurveDisplay.from_predictions(y_test, y_score, ax=ax1)
            ax1.set_title("ROC Curve")
            
            PrecisionRecallDisplay.from_predictions(y_test, y_score, ax=ax2)
            ax2.set_title("Precision-Recall Curve")
            
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax3, 
                                                    display_labels=["Non-Face", "Face"], cmap='Blues')
            ax3.set_title("Confusion Matrix")
            
            plt.suptitle("Model Evaluation Metrics")
            plt.tight_layout()
            
            report_path = Path("reports")
            report_path.mkdir(exist_ok=True)
            save_path = report_path / "evaluation_metrics.png"
            plt.savefig(save_path)
            logger.info(f"Plot evaluasi disimpan ke: {save_path}")
            
            plt.show()

        elif args.command == "infer":
            logger.info(f"Running LBP inference on {args.image} using {args.model_name}...")
            pipeline = InferencePipelineLBP(args.model_dir, args.model_name)
            
            pipeline.process_image(args.image, args.out, args.hat)
            
            logger.info(f"Output saved to {args.out}")

        elif args.command == "webcam":
            logger.info(f"Starting webcam inference with {args.model_name}...")
            pipeline = InferencePipelineLBP(args.model_dir, args.model_name)
            
            pipeline.process_webcam(args.camera, args.hat)

    except FileNotFoundError as e:
        logger.error(f"Error: {e}. Did you forget to train or provide assets?")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
