import logging
import cv2
import joblib
import numpy as np
from pathlib import Path

from .features import extract_lbp_features 
from .overlay import overlay_hat
from .utils import resize_to_fixed, setup_logging, load_hat_data # <-- Impor helper baru

logger = logging.getLogger(__name__)

class InferencePipelineLBP:
    def __init__(self, model_dir: Path, model_name: str):
        logger.info(f"Loading LBP inference pipeline...")
        
        # 1. Muat Model LBP+SVM/RF Anda
        model_path = model_dir / model_name
        if not model_path.exists():
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model: {model_path}")

        # 2. Muat Haar Cascade (untuk Proposal)
        cascade_dir = Path("assets/cascades/")
        face_cascade_path = cascade_dir / "haarcascade_frontalface_default.xml"
        eye_cascade_path = cascade_dir / "haarcascade_eye.xml"
        
        self.face_cascade = cv2.CascadeClassifier(str(face_cascade_path))
        self.eye_cascade = cv2.CascadeClassifier(str(eye_cascade_path))
        
        if self.face_cascade.empty():
            logger.error(f"Could not load face cascade from {face_cascade_path}")
            raise FileNotFoundError("Haar cascade not found.")
        if self.eye_cascade.empty():
            logger.warning(f"Could not load eye cascade from {eye_cascade_path}. Hat rotation disabled.")
            self.eye_cascade = None
            
        # 3. Logika pemuatan topi DIHAPUS dari __init__

    def process_frame(self, frame, hat_data, show_hat=True, show_box=True):
        """
        Pipeline deteksi: Terima 'hat_data' sebagai argumen.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_out = frame.copy()

        # TAHAP 1: Proposal
        rois = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        verified_boxes = []

        # TAHAP 2: Verifikasi
        for (x, y, w, h) in rois:
            roi_gray = gray[y:y+h, x:x+w]
            features = extract_lbp_features(roi_gray)
            prediction = self.model.predict([features])[0]
            
            if prediction == 1:
                verified_boxes.append((x, y, w, h))
        
        # TAHAP 3: Overlay
        for (x, y, w, h) in verified_boxes:
            if show_box:
                cv2.rectangle(frame_out, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            eye_coords = []
            if show_hat and hat_data is not None: # Gunakan hat_data dari argumen
                if self.eye_cascade:
                    face_roi_gray = gray[y:y+h, x:x+w]
                    eyes = self.eye_cascade.detectMultiScale(
                        face_roi_gray, 
                        scaleFactor=1.1, 
                        minNeighbors=4, 
                        minSize=(int(w*0.15), int(h*0.15))
                    )
                    
                    if len(eyes) >= 2:
                        sorted_eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
                        eye_coords = [((ex + ew // 2) + x, (ey + eh // 2) + y) for (ex, ey, ew, eh) in sorted_eyes]
                
                # Pass hat_data ke overlay_hat
                frame_out = overlay_hat(frame_out, (x, y, w, h), hat_data, eye_coords)

        return frame_out

    def process_image(self, image_path: Path, out_path: Path, hat_path: Path):
        """
        Modifikasi: Muat hat_data di sini untuk satu gambar.
        """
        frame = cv2.imread(str(image_path))
        if frame is None:
            logger.error(f"Could not read image: {image_path}")
            return
        
        hat_data = load_hat_data(hat_path) # Muat hat_data
        
        processed_frame = self.process_frame(frame, hat_data) # Pass ke process_frame
        cv2.imwrite(str(out_path), processed_frame)

    def process_webcam(self, camera_id: int, hat_path: Path):
        """
        Modifikasi: Muat hat_data di sini untuk webcam lokal.
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Cannot open camera {camera_id}.")
            return
            
        hat_data = load_hat_data(hat_path) # Muat hat_data
        if hat_data is None:
            logger.error(f"Gagal memuat topi {hat_path} untuk webcam.")
            return

        show_hat = True
        show_box = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame.")
                break
            
            frame = cv2.flip(frame, 1) # Mirror mode
            frame_resized = resize_to_fixed(frame, 720) # Resize agar cepat
            
            fps_start = cv2.getTickCount()

            # Proses frame
            processed_frame = self.process_frame(frame_resized, hat_data, show_hat, show_box)
            
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - fps_start)
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            hat_status = "ON" if show_hat else "OFF"
            box_status = "ON" if show_box else "OFF"
            cv2.putText(processed_frame, f"Hat (h): {hat_status}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(processed_frame, f"Box (b): {box_status}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Tampilkan nama topi
            if hat_data:
                cv2.putText(processed_frame, f"Hat: {hat_data['name']}", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("LBP+SVM Face Detector (Tekan 'q' untuk keluar)", processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                show_hat = not show_hat
            elif key == ord('b'):
                show_box = not show_box

        cap.release()
        cv2.destroyAllWindows()
