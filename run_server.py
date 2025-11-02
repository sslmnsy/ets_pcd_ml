import cv2
import numpy as np
import socket
import struct
import threading
import time
import math
import logging
from pathlib import Path

from pipelines.infer import InferencePipelineLBP
from pipelines.utils import setup_logging, load_hat_data

setup_logging()
logger = logging.getLogger(__name__)

class HatTryOnServerUDP:
    
    def __init__(self, pipeline: InferencePipelineLBP, hats_dir: Path, host='localhost', port=8888):
        self.host = host
        self.port = port
        self.pipeline = pipeline  
        
        self.server_socket = None
        self.clients = set()
        self.cap = None
        self.running = False
        self.sequence_number = 0
        self.max_packet_size = 60000
        self.mirror_mode = True
        
        # --- LOGIKA MULTI-TOPI ---
        self.hats_list = []
        self.current_hat_index = 0
        self.load_all_hats(hats_dir)
        
        # Topi mati secara default
        self.hat_enabled = False

    def load_all_hats(self, hats_dir: Path):
        """Memuat semua topi dan .json-nya dari satu direktori."""
        logger.info(f"🎩 Memuat semua topi dari {hats_dir}...")
        if not hats_dir.exists():
            logger.error(f"Direktori topi {hats_dir} tidak ditemukan!")
            return

        # Cari semua file .png (atau ekstensi gambar lain)
        extensions = ["*.png", "*.jpg", "*.jpeg"]
        hat_paths = []
        for ext in extensions:
            hat_paths.extend(hats_dir.rglob(ext))
        
        for hat_path in sorted(list(set(hat_paths))): # Urutkan agar konsisten
            hat_data = load_hat_data(hat_path)
            if hat_data:
                hat_name = hat_path.stem.upper().replace("-", " ").replace("_", " ")
                hat_data["name"] = hat_name 
                self.hats_list.append(hat_data)
                logger.info(f"  -> Berhasil memuat topi: {hat_data['name']}")
        
        if not self.hats_list:
            logger.warning("Tidak ada topi yang ditemukan!")
        else:
            logger.info(f"Total {len(self.hats_list)} topi berhasil dimuat.")

    def get_current_hat(self):
        """Mengambil data topi yang sedang aktif."""
        if not self.hats_list or self.current_hat_index >= len(self.hats_list):
            return None
        return self.hats_list[self.current_hat_index]
        
    def find_hat_by_name(self, category_name: str):
        """Cari indeks topi berdasarkan nama kategori dari Godot."""
        for i, hat_data in enumerate(self.hats_list):
            if hat_data["name"] == category_name:
                return i
        return None # Tidak ditemukan

    def process_frame(self, frame):
        """
        Ambil topi saat ini dan berikan ke pipeline untuk diproses.
        """
        current_hat_data = self.get_current_hat()
        
        # MODIFIED: Kirim 'hat_enabled' ke pipeline
        processed_frame = self.pipeline.process_frame(
            frame, 
            hat_data=current_hat_data,
            show_hat=self.hat_enabled, # <-- MODIFIKASI KUNCI
            show_box=True
        )
        
        # Tambahkan nama topi ke layar HANYA jika aktif
        if self.hat_enabled and current_hat_data:
            cv2.putText(processed_frame, f"Hat: {current_hat_data['name']}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return processed_frame


    def start_server(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            logger.info(f"🚀 UDP Server started at {self.host}:{self.port}")
            
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.error("❌ Error: Cannot access webcam")
                return
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.running = True
            
            listen_thread = threading.Thread(target=self.listen_for_clients, daemon=True)
            listen_thread.start()
            
            stream_thread = threading.Thread(target=self.stream_webcam, daemon=True)
            stream_thread.start()
            
        except Exception as e:
            logger.error(f"❌ Error starting server: {e}")
    
    def listen_for_clients(self):
        """MODIFIED: Dengarkan perintah HAT_CATEGORY dan HAT_OFF."""
        self.server_socket.settimeout(1.0)
        
        while self.running:
            try:
                data, addr = self.server_socket.recvfrom(1024)
                message = data.decode('utf-8')
                
                if message == "REGISTER":
                    if addr not in self.clients:
                        self.clients.add(addr)
                        logger.info(f"✅ Client registered: {addr}")
                        self.server_socket.sendto("REGISTERED".encode('utf-8'), addr)
                
                elif message == "UNREGISTER":
                    if addr in self.clients:
                        self.clients.remove(addr)
                        logger.info(f"❌ Client unregistered: {addr}")
                
                # --- KONTROL TOPI BARU ---
                
                # NEW: Matikan topi
                elif message == "HAT_OFF":
                    logger.info(f"Perintah 'HAT_OFF' diterima. Menonaktifkan topi.")
                    self.hat_enabled = False
                
                # NEW: Ganti topi berdasarkan Kategori
                elif message.startswith("HAT_CATEGORY:"):
                    category_name = message.split(":", 1)[1]
                    logger.info(f"Perintah 'HAT_CATEGORY:{category_name}' diterima.")
                    
                    found_index = self.find_hat_by_name(category_name)
                    if found_index is not None:
                        self.current_hat_index = found_index
                        self.hat_enabled = True # <-- NEW: Aktifkan topi saat dipilih
                        logger.info(f"Topi diganti ke: {category_name}")
                    else:
                        logger.warning(f"Kategori topi tidak ditemukan: {category_name}")
                # ---
                        
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.warning(f"⚠️  Listen Error: {e}")
    
    def stream_webcam(self):
        # ... (Fungsi ini tidak berubah, karena process_frame() sudah diubah)
        while self.running:
            try:
                if len(self.clients) == 0:
                    time.sleep(0.1)
                    continue
                
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                if self.mirror_mode:
                    frame = cv2.flip(frame, 1)
                
                frame = self.process_frame(frame)
                
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                result, encoded_img = cv2.imencode('.jpg', frame, encode_param)
                
                if result:
                    self.send_frame_to_clients(encoded_img.tobytes())
                
            except Exception as e:
                logger.error(f"❌ Error streaming: {e}")
                break
    
    def send_frame_to_clients(self, frame_data):
        # ... (Fungsi ini tidak berubah) ...
        if not frame_data:
            return
        
        self.sequence_number = (self.sequence_number + 1) % 65536
        frame_size = len(frame_data)
        header_size = 12
        payload_size = self.max_packet_size - header_size
        total_packets = math.ceil(frame_size / payload_size)
        
        for client_addr in self.clients.copy():
            try:
                for packet_index in range(total_packets):
                    start_pos = packet_index * payload_size
                    end_pos = min(start_pos + payload_size, frame_size)
                    packet_data = frame_data[start_pos:end_pos]
                    
                    header = struct.pack("!III", self.sequence_number, total_packets, packet_index)
                    udp_packet = header + packet_data
                    
                    self.server_socket.sendto(udp_packet, client_addr)
                    
            except Exception as e:
                if hasattr(e, 'errno') and e.errno == 10054:
                    logger.warning(f"Klien {client_addr} terputus (errno 10054). Menghapus.")
                    self.clients.remove(client_addr)
                else:
                    logger.error(f"❌ Error sending to {client_addr}: {e}")

    def stop_server(self):
        # ... (Fungsi ini tidak berubah) ...
        logger.info("⏹️  Stopping server...")
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        if self.cap:
            self.cap.release()
        logger.info("✅ Server stopped")

if __name__ == "__main__":
    print("=" * 60)
    print("🎩 HAT TRY-ON SERVER (Multi-Hat / Category Version)")
    print("=" * 60)
    
    MODELS_DIR = Path("models")
    MODEL_NAME = "svm_lbp.pkl"
    HATS_DIR = Path("assets/hats") # <-- Folder baru
    
    try:
        pipeline = InferencePipelineLBP(
            model_dir=MODELS_DIR, 
            model_name=MODEL_NAME
        )
    except FileNotFoundError as e:
        logger.error(f"FATAL: Gagal memuat pipeline. {e}")
        logger.error("Pastikan Anda sudah menjalankan 'app.py train' dan aset ada.")
        exit()

    server = HatTryOnServerUDP(
        pipeline=pipeline,
        hats_dir=HATS_DIR, 
        host='0.0.0.0', 
        port=8888
    )
    
    try:
        server.start_server()
        logger.info(f"📺 Server running! Streaming on 0.0.0.0:8888")
        logger.info("⌨️  Press Ctrl+C to stop")
        
        while server.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("\n⏹️  Stopping server...")
        server.stop_server()
