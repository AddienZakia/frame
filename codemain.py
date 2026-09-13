import sys
import cv2
import numpy as np
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QMessageBox, QFrame, QScrollArea)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt

from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
from sahi import AutoDetectionModel

def jalankan_deteksi_sahi(image_path, detection_model):
    """
    Fungsi adaptasi untuk memproses SATU gambar dari GUI.
    Hanya menghasilkan gambar visualisasi bounding box tanpa menyimpan CSV/TXT.
    """
    # 1. Jalankan SAHI Sliced Prediction (Parameter persis dari Kaggle kamu)
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=640,          # Menggunakan SLICE_SIZE = 640 dari Kaggle
        slice_width=640,
        overlap_height_ratio=0.2,   # OVERLAP_RATIO = 0.2
        overlap_width_ratio=0.2,
        verbose=0,
        postprocess_match_threshold=0.5
    )
    
    # 2. Proses Visualisasi Kustom OpenCV sesuai setelan Kaggle-mu
    img_np = cv2.imread(image_path)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    visual_result_dict = visualize_object_predictions(
        image=img_np,
        object_prediction_list=result.object_prediction_list,
        rect_th=2,
        text_size=0.6,
        text_th=2,
        hide_labels=False
    )
    visual_result_np = visual_result_dict['image']
    
    # 3. Simpan Gambar Visualisasi sementara (.png) untuk di-load PyQt6
    png_path = "temp_output.png"
    cv2.imwrite(png_path, cv2.cvtColor(visual_result_np, cv2.COLOR_RGB2BGR))
    
    # Kembalikan path gambar hasil visualisasi dan jumlah objek terdeteksi untuk dipakai GUI
    return png_path, len(result.object_prediction_list)

class ZoomableImageLabel(QLabel):
    """
    QLabel dengan fitur zoom menggunakan mouse wheel.
    Awalnya gambar tampil fit ke ukuran label (normal).
    Scroll up untuk zoom in, scroll down untuk zoom out.
    """
    def __init__(self):
        super().__init__()
        self.original_pixmap = None
        self.fit_pixmap = None  # Pixmap yang di-fit ke ukuran label
        self.zoom_level = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 3.0
        self.zoom_step = 0.1
        self.has_zoomed = False  # Flag untuk tracking apakah sudah di-zoom
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            border: 2px solid #D1D1D1; 
            background-color: white;
            color: #888888;
        """)

    def set_pixmap(self, pixmap):
        """Set pixmap awal, display dalam ukuran normal (fit ke label)"""
        self.original_pixmap = pixmap
        self.fit_pixmap = pixmap
        self.zoom_level = 1.0
        self.has_zoomed = False
        self.update_display_fit()

    def update_display_fit(self):
        """Tampilkan gambar fit ke ukuran label (normal view)"""
        if self.fit_pixmap:
            scaled_pixmap = self.fit_pixmap.scaled(
                self.width() - 10, 
                self.height() - 10,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            super().setPixmap(scaled_pixmap)

    def update_display_zoom(self):
        """Tampilkan gambar sesuai zoom level (saat user scroll)"""
        if self.original_pixmap:
            new_width = int(self.original_pixmap.width() * self.zoom_level)
            new_height = int(self.original_pixmap.height() * self.zoom_level)
            scaled_pixmap = self.original_pixmap.scaled(
                new_width, new_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            super().setPixmap(scaled_pixmap)

    def wheelEvent(self, event):
        """Handle mouse wheel untuk zoom"""
        if self.original_pixmap is None:
            return
        
        # Flag zoom pertama kali
        if not self.has_zoomed:
            self.has_zoomed = True
        
        # Scroll up = zoom in, scroll down = zoom out
        if event.angleDelta().y() > 0:
            self.zoom_level = min(self.zoom_level + self.zoom_step, self.max_zoom)
        else:
            self.zoom_level = max(self.zoom_level - self.zoom_step, self.min_zoom)
        
        self.update_display_zoom()
        event.accept()

    def reset_zoom(self):
        """Reset ke tampilan normal (fit ke label)"""
        self.zoom_level = 1.0
        self.has_zoomed = False
        self.update_display_fit()

    def resizeEvent(self, event):
        """Update tampilan saat window di-resize"""
        super().resizeEvent(event)
        if not self.has_zoomed and self.fit_pixmap:
            # Jika belum di-zoom, update fit display saat resize
            self.update_display_fit()

class PegonApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistem Deteksi Aksara Pegon")
        self.setMinimumSize(1000, 600)
        
        # Variabel State
        self.image_path = None
        self.is_detected = False # Flag untuk mengecek apakah sudah ada hasil deteksi
        
        # --- LOAD MODEL YOLOv8 + SAHI ---
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'runs', 'detect', 'pegon_model_v1', 'weights', 'best.pt')
        
        # Fallback jika model di folder runs tidak ditemukan
        if not os.path.exists(model_path):
            alt_path = os.path.join(base_dir, 'yolov8s.pt')
            if os.path.exists(alt_path):
                model_path = alt_path

        # Deteksi device (GPU jika CUDA tersedia, jika tidak pakai CPU)
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            device = 'cpu'

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path, 
            confidence_threshold=0.3,
            device=device
        )

        self.initUI()

    def initUI(self):
        # 1. Widget Utama & Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 2. Header (Sesuai desain biru di mockup)
        header = QFrame()
        header.setFixedHeight(60)
        header.setStyleSheet("background-color: #4A90E2;") 
        header_layout = QHBoxLayout(header)
        
        title_label = QLabel("Sistem Deteksi Aksara Pegon")
        title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title_label.setStyleSheet("color: white; margin-left: 20px;")
        
        logo_label = QLabel() 
        base_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(base_dir, "logo.png")
        
        if os.path.exists(logo_path):
            pixmap_logo = QPixmap(logo_path)
            # Menyesuaikan ukuran logo agar pas dengan tinggi header (misal tinggi 45px)
            logo_label.setPixmap(pixmap_logo.scaled(60, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            logo_label.setText("Logo ITS")
            logo_label.setStyleSheet("color: white; font-weight: bold;")
            
        logo_label.setStyleSheet("margin-right: 20px;")
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(logo_label)
        main_layout.addWidget(header)

        # 3. Area Konten (Tengah)
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(20)

        # --- Sisi Kiri: Display Gambar dengan Zoom ---
        scroll_area = QScrollArea()
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: 2px solid #D1D1D1;
                background-color: white;
            }
            QScrollBar:vertical {
                border: 1px solid #D1D1D1;
                background-color: white;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background-color: #4A90E2;
                border-radius: 6px;
            }
            QScrollBar:horizontal {
                border: 1px solid #D1D1D1;
                background-color: white;
                height: 12px;
            }
            QScrollBar::handle:horizontal {
                background-color: #4A90E2;
                border-radius: 6px;
            }
        """)
        scroll_area.setWidgetResizable(True)
        
        self.display_area = ZoomableImageLabel()
        self.display_area.setText("Gambar akan tampil di sini\n(Scroll mouse untuk zoom in/out)")
        scroll_area.setWidget(self.display_area)
        content_layout.addWidget(scroll_area, stretch=4)

        # --- Sisi Kanan: Panel Tombol ---
        button_panel = QVBoxLayout()
        
        btn_style = """
            QPushButton {
                background-color: #4A90E2;
                color: white;
                font-size: 18px;
                font-weight: bold;
                border-radius: 5px;
                padding: 15px;
                min-width: 200px;
            }
            QPushButton:hover { background-color: #357ABD; }
            QPushButton:pressed { background-color: #2A5F91; }
        """

        self.btn_input = QPushButton("Input Citra")
        self.btn_input.setStyleSheet(btn_style)
        self.btn_input.clicked.connect(self.action_input)

        self.btn_start = QPushButton("Start")
        self.btn_start.setStyleSheet(btn_style)
        self.btn_start.clicked.connect(self.action_start)

        self.btn_back = QPushButton("Back")
        self.btn_back.setStyleSheet(btn_style.replace("#4A90E2", "#AABBC3")) 
        self.btn_back.clicked.connect(self.action_back)

        button_panel.addWidget(self.btn_input)
        button_panel.addWidget(self.btn_start)
        button_panel.addWidget(self.btn_back)
        button_panel.addStretch() 

        content_layout.addLayout(button_panel, stretch=1)
        main_layout.addLayout(content_layout)

    # --- LOGIKA TOMBOL ---

    def action_input(self):
        if self.is_detected:
            QMessageBox.warning(self, "Peringatan", "Harap tekan 'Back' terlebih dahulu untuk melakukan input baru.")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih Citra Pegon", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            self.display_area.set_pixmap(pixmap)
            self.display_area.setText("")

    def action_start(self):
        if not self.image_path:
            QMessageBox.critical(self, "Error", "Silahkan input citra terlebih dahulu!")
            return
        
        if self.is_detected:
            QMessageBox.warning(self, "Peringatan", "Harap tekan 'Back' terlebih dahulu untuk mengulang deteksi.")
            return

        try:
            self.display_area.setText("Sedang memproses SAHI...")
            QApplication.processEvents() 

            # --- PANGGIL FUNGSI DETEKSI BERSIH ---
            path_gambar_hasil, jumlah_objek = jalankan_deteksi_sahi(
                image_path=self.image_path,
                detection_model=self.detection_model
            )
            
            # Tampilkan gambar hasil visualisasi kotak ke layar GUI
            res_pixmap = QPixmap(path_gambar_hasil)
            self.display_area.set_pixmap(res_pixmap)
            
            self.is_detected = True 
            
            # Berikan notifikasi pop-up sukses beserta jumlah objek
            QMessageBox.information(
                self, 
                "Deteksi Selesai", 
                f"✅ Berhasil memproses citra!\nDitemukan: {jumlah_objek} objek aksara Pegon."
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Terjadi kesalahan saat deteksi: {str(e)}")

    def action_back(self):
        # Reset semua state & bersihkan file temp jika ada
        self.image_path = None
        self.is_detected = False
        self.display_area.reset_zoom()
        self.display_area.original_pixmap = None
        self.display_area.clear()
        self.display_area.setText("Gambar akan tampil di sini\n(Scroll mouse untuk zoom in/out)")
        
        if os.path.exists("temp_output.png"):
            try:
                os.remove("temp_output.png")
            except:
                pass
                
        print("Sistem di-reset.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PegonApp()
    window.show()
    sys.exit(app.exec())