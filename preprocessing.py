import os
import random
from pathlib import Path
import cv2  
import numpy as np 

# ==============================================================================
# KELAS 1: Untuk Pre-processing (Cropping) Wajah
# ==============================================================================

class DatasetPreprocessor:
    """
    Melakukan pre-processing pada dataset wajah yang belum di-crop.
    Menggunakan Haar Cascade untuk mendeteksi, memotong, dan menyimpan wajah.
    """
    
    def __init__(self):
        # Muat Haar Cascade classifier
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except Exception as e:
            print(f"❌ Error memuat Haar Cascade: {e}")
            print("  Pastikan OpenCV terinstal (pip install opencv-python-headless)")
            raise

    def get_all_images(self, directory):
        """Mendapatkan semua file gambar dari direktori secara rekursif"""
        dir_path = Path(directory)
        if not dir_path.exists(): 
            return []
        
        image_paths = set()
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.update(list(dir_path.rglob(ext)))
        return list(image_paths)

    def preprocess_faces(self, input_dir, output_dir, max_faces=10000):
        """
        Memindai 'input_dir' untuk gambar, mendeteksi wajah, dan menyimpan
        wajah yang di-crop ke 'output_dir'.
        """
        print("=" * 60)
        print("🎭 Face Pre-processing (Cropping)")
        print("=" * 60)
        print(f"🔍 Sumber (Un-cropped): {input_dir}")
        print(f"🎯 Tujuan (Cropped): {output_dir}")
        print(f"📊 Maksimum wajah untuk diekstrak: {max_faces}")
        
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        
        if not in_path.exists():
            print(f"❌ Error: Direktori input '{in_path}' tidak ditemukan!")
            return

        # Buat direktori output jika belum ada
        out_path.mkdir(parents=True, exist_ok=True)
        
        image_files = self.get_all_images(in_path)
        if not image_files:
            print(f"⚠️  Warning: Tidak ada gambar yang ditemukan di {in_path}")
            return
            
        print(f"ℹ️  Menemukan {len(image_files)} gambar sumber untuk dipindai.")
        random.shuffle(image_files)

        saved_count = 0
        processed_img_count = 0

        for img_path in image_files:
            if saved_count >= max_faces:
                print(f"\n✅ Telah mencapai batas {max_faces} wajah. Berhenti.")
                break
            
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"⚠️  Warning: Tidak bisa membaca {img_path}. Melewati.")
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(40, 40)
                )
                
                processed_img_count += 1

                for (x, y, w, h) in faces:
                    cropped_face = img[y:y+h, x:x+w]
                    save_name = f"face_crop_{saved_count:05d}.jpg"
                    save_path = out_path / save_name
                    cv2.imwrite(str(save_path), cropped_face)
                    saved_count += 1
                    
                    if saved_count % 100 == 0:
                        print(f"  -> Ekstrak {saved_count} wajah...")
                    
                    if saved_count >= max_faces:
                        break
                
                if processed_img_count % 50 == 0:
                   print(f"  ...memindai {processed_img_count}/{len(image_files)} gambar...")

            except Exception as e:
                print(f"❌ Error memproses {img_path}: {e}")

        print("\n" + "=" * 60)
        print("✅ Pre-processing Selesai!")
        print(f"Total gambar dipindai: {processed_img_count}")
        print(f"Total wajah diekstrak: {saved_count}")
        print(f"Disimpan di: {output_dir}")
        print("=" * 60)

# ==============================================================================
# KELAS 2: Untuk Cleaning (Cutting) Dataset
# ==============================================================================

class DatasetCleaner:
    """Membersihkan dataset dengan menghapus gambar berlebih"""
    
    def __init__(self):
        self.deleted_count = 0
        self.kept_count = 0
    
    def delete_empty_folders(self, directory):
        """Secara rekursif menghapus folder kosong"""
        dir_path = Path(directory)
        deleted_count = 0
        
        for dirpath, dirnames, filenames in os.walk(dir_path, topdown=False):
            current_dir = Path(dirpath)
            
            if current_dir == dir_path:
                continue
            
            try:
                if not any(current_dir.iterdir()):
                    current_dir.rmdir()
                    deleted_count += 1
            except Exception as e:
                pass
        
        return deleted_count
    
    def get_all_images(self, directory):
        """Mendapatkan semua file gambar tanpa duplikat (case-insensitive)"""
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return []
        
        image_paths = set()
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            found = list(dir_path.rglob(ext))
            for img in found:
                image_paths.add(img)
        
        return list(image_paths)
    
    def count_images(self, directory):
        """Menghitung total gambar di direktori (termasuk subdirektori)"""
        return len(self.get_all_images(directory))
    
    def cleanup_directory(self, directory, max_samples, dry_run=False):
        """Menghapus gambar berlebih dari direktori"""
        dir_path = Path(directory)
        
        if not dir_path.exists():
            print(f"❌ Error: {directory} tidak ditemukan!")
            return
        
        print(f"   🔍 Memindai gambar...")
        image_files = self.get_all_images(directory)
        total_images = len(image_files)
        
        print(f"\n📂 Direktori: {directory}")
        print(f"   Total gambar UNIK ditemukan: {total_images}")
        
        if total_images <= max_samples:
            print(f"   ✅ Tidak perlu pembersihan (di bawah batas {max_samples})")
            self.kept_count += total_images
            return
        
        to_delete_count = total_images - max_samples
        
        print(f"   ⚠️  Melebihi batas {to_delete_count} gambar")
        print(f"   📊 Akan menyimpan: {max_samples} gambar")
        print(f"   📊 Akan menghapus: {to_delete_count} gambar")
        
        if dry_run:
            print(f"   🔍 DRY RUN: Akan menghapus {to_delete_count} gambar")
            print(f"   🔍 DRY RUN: Akan menyimpan {max_samples} gambar")
            return
        
        print(f"   🔀 Mengacak gambar secara acak...")
        random.shuffle(image_files)
        
        images_to_keep = image_files[:max_samples]
        images_to_delete = image_files[max_samples:]
        
        print(f"\n   📋 Ringkasan sebelum penghapusan:")
        print(f"       Gambar untuk disimpan: {len(images_to_keep)}")
        print(f"       Gambar untuk dihapus: {len(images_to_delete)}")
        
        print(f"\n   🗑️  Akan MENGHAPUS {len(images_to_delete)} gambar...")
        confirm = input("   ⚠️  Apakah Anda yakin? Ini tidak bisa dibatalkan! (yes/no): ").strip().lower()
        
        if confirm != 'yes':
            print("   ❌ Pembersihan dibatalkan")
            return
        
        print(f"   🗑️  Menghapus {len(images_to_delete)} file...")
        deleted = 0
        errors = 0
        
        for i, img_path in enumerate(images_to_delete):
            try:
                if img_path.exists():
                    img_path.unlink()
                    deleted += 1
                    
                    if (i + 1) % 100 == 0:
                        print(f"       Progres: {i + 1}/{len(images_to_delete)} diproses, {deleted} dihapus...")
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"       ❌ Error menghapus {img_path}: {e}")
        
        if errors > 5:
            print(f"       ⚠️  ... dan {errors - 5} error lainnya")
        
        print(f"   ✅ Berhasil menghapus: {deleted} gambar")
        print(f"   ⚠️  Errors: {errors}")
        
        remaining_images = self.get_all_images(directory)
        print(f"   ✅ Verifikasi tersisa: {len(remaining_images)} gambar UNIK")
        
        print(f"   🗑️  Membersihkan folder kosong...")
        empty_folders_deleted = self.delete_empty_folders(dir_path)
        
        if empty_folders_deleted > 0:
            print(f"   ✅ Menghapus {empty_folders_deleted} folder kosong")
        else:
            print(f"   ℹ️  Tidak ada folder kosong ditemukan")
        
        self.deleted_count += deleted
        self.kept_count += len(remaining_images)
    
    def cleanup_dataset(self, faces_dir, non_faces_dir, max_samples_per_class, 
                        dry_run=False):
        """
        Membersihkan direktori wajah dan non-wajah
        (Sekarang menerima None untuk melewati direktori)
        """
        
        print("=" * 60)
        print("🗑️  DATASET CLEANUP TOOL")
        print("=" * 60)
        
        if dry_run:
            print("\n🔍 DRY RUN MODE - Tidak ada file yang akan dihapus")
        
        print(f"\n⚙️  Konfigurasi:")
        if faces_dir:
            print(f"   Direktori Wajah: {faces_dir}")
        if non_faces_dir:
            print(f"   Direktori Non-Wajah: {non_faces_dir}")
        print(f"   Max sampel per kelas: {max_samples_per_class}")
        
        faces_count = 0
        if faces_dir:
            faces_count = self.count_images(faces_dir)
            
        non_faces_count = 0
        if non_faces_dir:
            non_faces_count = self.count_images(non_faces_dir)
        
        print(f"\n📊 Status Saat Ini:")
        if faces_dir:
            print(f"   Wajah: {faces_count} gambar")
        if non_faces_dir:
            print(f"   Non-Wajah: {non_faces_count} gambar")
        print(f"   Total: {faces_count + non_faces_count} gambar")
        
        total_to_delete = 0
        if faces_dir:
            total_to_delete += max(0, faces_count - max_samples_per_class)
        if non_faces_dir:
            total_to_delete += max(0, non_faces_count - max_samples_per_class)
        
        if total_to_delete == 0:
            print(f"\n✅ Tidak perlu pembersihan! Semua direktori yang dipilih ada di bawah batas.")
            return
        
        print(f"\n⚠️  Total gambar untuk dihapus: {total_to_delete}")
        
        if not dry_run:
            print("\n" + "=" * 60)
            final_confirm = input("⚠️  KONFIRMASI AKHIR: Ketik 'DELETE' untuk melanjutkan: ").strip()
            
            if final_confirm != 'DELETE':
                print("❌ Pembersihan dibatalkan")
                return
        
        if faces_dir:
            print("\n" + "=" * 60)
            print("🗑️  Membersihkan direktori FACES...")
            print("=" * 60)
            self.cleanup_directory(faces_dir, max_samples_per_class, dry_run)
        
        if non_faces_dir:
            print("\n" + "=" * 60)
            print("🗑️  Membersihkan direktori NON-FACES...")
            print("=" * 60)
            self.cleanup_directory(non_faces_dir, max_samples_per_class, dry_run)
        
        print("\n" + "=" * 60)
        print("📊 RINGKASAN PEMBERSIHAN")
        print("=" * 60)
        
        if dry_run:
            print("\n🔍 Ini adalah DRY RUN - tidak ada file yang dihapus")
            print("   Jalankan dengan dry_run=False untuk benar-benar menghapus file")
        else:
            print(f"\n✅ Total dihapus: {self.deleted_count} gambar")
            print(f"✅ Total disimpan: {self.kept_count} gambar")
            
            print(f"\n📊 Status Akhir:")
            if faces_dir:
                faces_count_after = self.count_images(faces_dir)
                print(f"   Wajah: {faces_count_after} gambar")
            if non_faces_dir:
                non_faces_count_after = self.count_images(non_faces_dir)
                print(f"   Non-Wajah: {non_faces_count_after} images")

# ==============================================================================
# FUNGSI-FUNGSI UTAMA 
# ==============================================================================

def _execute_cleaner_logic(faces_dir, non_faces_dir, max_samples):
    """
    Fungsi helper internal untuk menjalankan logika cleaner (dry run + eksekusi)
    Menerima path None jika direktori itu dilewati.
    """
    if not faces_dir and not non_faces_dir:
        print("\nℹ️  Tidak ada direktori yang dipilih untuk dibersihkan.")
        return

    print(f"\nBatas maksimum per kelas diatur ke: {max_samples}")

    # --- Dry Run ---
    print("\n🔍 Dry Run Mode:")
    print("   Dry run akan MENUNJUKKAN apa yang akan dihapus tanpa benar-benar menghapus")
    dry_run_input = input("Lakukan dry run dulu? (y/n, default: y): ").strip().lower()
    
    if dry_run_input == '' or dry_run_input == 'y':
        print("\n" + "=" * 60)
        print("🔍 MENJALANKAN DRY RUN...")
        print("=" * 60)
        
        cleaner_dry = DatasetCleaner()
        cleaner_dry.cleanup_dataset(faces_dir, non_faces_dir, max_samples, dry_run=True)
        
        print("\n" + "=" * 60)
        proceed = input("\n💀 Lanjutkan dengan penghapusan AKTUAL? (yes/no): ").strip().lower()
        
        if proceed != 'yes':
            print("❌ Pembersihan dibatalkan. Tidak ada file yang dihapus.")
            return
    
    print("\n" + "=" * 60)
    print("💀 MENJALANKAN PEMBERSIHAN AKTUAL...")
    print("=" * 60)
    
    cleaner = DatasetCleaner()
    cleaner.cleanup_dataset(faces_dir, non_faces_dir, max_samples, dry_run=False)
    
    print("\n" + "=" * 60)
    print("✅ PEMBERSIHAN SELESAI!")
    print("=" * 60)

def run_full_workflow():
    """
    Menjalankan alur kerja penuh:
    1. Pre-process (Crop) data mentah untuk MENCIPTAKAN data wajah.
    2. Clean (Cut) data wajah/non-wajah yang dihasilkan ke jumlah maks.
    """
    print("\n" + "=" * 60)
    print("🚀 ALUR KERJA PENUH: PRE-PROCESS -> CLEAN")
    print("=" * 60)
    
    # --- LANGKAH 1: PRE-PROCESS (Hanya untuk Wajah) ---
    print("Langkah 1: Pre-process (Crop) Wajah")
    print("   Kita akan mengambil data mentah (un-cropped) dan membuat data wajah (cropped).")
    
    input_dir = input("\nPath ke gambar sumber UN-CROPPED (misal: caltech_10k/): ").strip()
    faces_output_dir = input("Path untuk menyimpan wajah CROPPED BARU (misal: dataset/faces): ").strip()
    
    if not input_dir or not faces_output_dir:
        print("\n❌ Error: Direktori Input dan Output diperlukan.")
        return

    max_extract = 20000  # Ekstrak sebanyak mungkin, kita cut nanti
    
    try:
        preprocessor = DatasetPreprocessor()
        preprocessor.preprocess_faces(input_dir, faces_output_dir, max_faces=max_extract)
    except Exception as e:
        print(f"\n❌ Terjadi error saat pre-processing: {e}")
        return

    # --- LANGKAH 2: OPSI UNTUK CLEAN (CUT) ---
    print("\n" + "=" * 60)
    print("Langkah 2: Clean (Cut) Dataset")
    print(f"   Pre-processing selesai. Wajah baru ada di: {faces_output_dir}")
    print("   Sekarang, apakah Anda ingin memotong (cut) folder dataset ke jumlah akhir?")
    
    run_cleaner_prompt = input("\nLanjutkan ke tahap Cut/Clean? (y/n): ").strip().lower()
    
    if run_cleaner_prompt != 'y':
        print("\n✅ Oke. Alur kerja dihentikan. Data pre-process Anda sudah tersimpan.")
        print("   Jalankan skrip lagi dan pilih [3] jika Anda ingin clean-up nanti.")
        return

    # --- LANGKAH 3: MENJALANKAN CLEANER (dengan pilihan) ---
    print("\n--- Menjalankan Cleaner ---")
    
    faces_dir_to_clean = None
    process_faces = input(f"Cut/Clean folder FACES ({faces_output_dir})? (y/n): ").strip().lower()
    if process_faces == 'y':
        faces_dir_to_clean = faces_output_dir
        
    non_faces_dir_to_clean = None
    process_non_faces = input("Cut/Clean folder NON-FACES juga? (y/n): ").strip().lower()
    if process_non_faces == 'y':
        non_faces_dir_input = input("  -> Path ke direktori non-faces (default: dataset/non_faces): ").strip()
        non_faces_dir_to_clean = non_faces_dir_input if non_faces_dir_input else "dataset/non_faces"

    if not faces_dir_to_clean and not non_faces_dir_to_clean:
        print("\nℹ️  Tidak ada direktori yang dipilih untuk dibersihkan. Mengakhiri alur kerja.")
        return

    max_samples_input = input(f"\nJumlah MAKSIMUM untuk disimpan (misal: 500): ").strip()
    max_samples = int(max_samples_input) if max_samples_input else 500
    
    # Panggil helper baru
    _execute_cleaner_logic(faces_dir_to_clean, non_faces_dir_to_clean, max_samples)

    print("\n" + "=" * 60)
    print("✅ ALUR KERJA SELESAI!")
    print("=" * 60)


def run_preprocessor_only():
    """Hanya menjalankan pre-processor"""
    print("\n" + "=" * 60)
    print("🎭 PRE-PROCESS FACES (CROPPING)")
    print("=" * 60)
    print("Tool ini akan memindai direktori gambar un-cropped (misal Caltech 10k),")
    print("mendeteksi wajah, dan menyimpan gambar wajah BARU yang di-crop ke direktori output.")
    
    input_dir = input("\nPath ke gambar sumber UN-CROPPED: ").strip()
    output_dir = input("Path untuk menyimpan wajah CROPPED BARU (misal: dataset/faces): ").strip()
    max_faces = input("Maksimum wajah untuk diekstrak (default: 10000): ").strip()

    if not input_dir or not output_dir:
        print("\n❌ Error: Direktori Input dan Output diperlukan.")
        return
    
    max_faces = int(max_faces) if max_faces else 10000
    
    try:
        preprocessor = DatasetPreprocessor()
        preprocessor.preprocess_faces(input_dir, output_dir, max_faces)
    except Exception as e:
        print(f"\n❌ Terjadi error saat pre-processing: {e}")
        print("  Pastikan 'opencv-python-headless' terinstal.")


def run_cleaner_only():
    """Hanya menjalankan cleaner (logika lama Anda)"""
    print("\n" + "=" * 60)
    print("🗑️ DATASET CLEANUP (MENGHAPUS)")
    print("=" * 60)
    print("Tool ini akan menghapus gambar *berlebih* dari folder training Anda")
    print("untuk menjaganya di bawah batas tertentu.")

    process_faces = input("\nBersihkan direktori FACES? (y/n): ").strip().lower()
    faces_dir = None
    if process_faces == 'y':
        faces_dir_input = input("  -> Path ke direktori faces (default: dataset/faces): ").strip()
        faces_dir = faces_dir_input if faces_dir_input else "dataset/faces"

    process_non_faces = input("\nBersihkan direktori NON-FACES? (y/n): ").strip().lower()
    non_faces_dir = None
    if process_non_faces == 'y':
        non_faces_dir_input = input("  -> Path ke direktori non-faces (default: dataset/non_faces): ").strip()
        non_faces_dir = non_faces_dir_input if non_faces_dir_input else "dataset/non_faces"

    if not faces_dir and not non_faces_dir:
        print("\nℹ️  Tidak ada direktori yang dipilih untuk dibersihkan. Kembali ke menu utama.")
        return
    
    max_samples_input = input("\nMaksimum gambar per kelas (default: 1000): ").strip()
    max_samples = int(max_samples_input) if max_samples_input else 1000
    
    # Panggil helper baru
    _execute_cleaner_logic(faces_dir, non_faces_dir, max_samples)


def main():
    """Fungsi main dengan mode interaktif"""
    
    while True:
        print("\n" + "=" * 60)
        print("🛠️  TOOL PERSIAPAN DATASET")
        print("=" * 60)
        print("\nApa yang ingin Anda lakukan?")
        print("  [1] Alur Kerja Penuh (Pre-process Un-cropped -> Clean)")
        print("  [2] HANYA Pre-process (Crop) data un-cropped")
        print("  [3] HANYA Clean (Cut) folder yang sudah ada")
        print("  [4] Keluar")
        choice = input("Masukkan pilihan (1, 2, 3, atau 4): ").strip()
        
        if choice == '1':
            run_full_workflow()
        elif choice == '2':
            run_preprocessor_only()
        elif choice == '3':
            run_cleaner_only()
        elif choice == '4':
            print("\n👋 Sampai jumpa!")
            break
        else:
            print("\n❌ Pilihan tidak valid. Silakan masukkan 1, 2, 3, atau 4.")


if __name__ == "__main__":
    main()