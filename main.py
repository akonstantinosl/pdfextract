import os
os.environ['KIVY_ASYNC_LIB'] = 'trio'
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.screenmanager import MDScreenManager
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFillRoundFlatButton, MDFillRoundFlatIconButton
from kivymd.uix.label import MDLabel
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.list import OneLineListItem
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.dialog import MDDialog
from kivy.clock import mainthread, Clock
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, StringProperty, NumericProperty

import threading
import io
import sys
import gc
import zipfile
import subprocess

# Dapatkan direktori poppler
if getattr(sys, 'frozen', False):
    # Jika dijalankan sebagai aplikasi yang di-bundle
    application_path = os.path.dirname(sys.executable)
else:
    # Jika dijalankan sebagai skrip python biasa
    application_path = os.path.dirname(os.path.abspath(__file__))

# Tambahkan path Poppler (yang di-bundle) ke environment PATH
poppler_path = os.path.join(application_path, 'poppler', 'Library', 'bin')
os.environ['PATH'] = poppler_path + os.pathsep + os.environ.get('PATH', '')

# Monkey patch subprocess.Popen untuk menyembunyikan console di Windows
if sys.platform == 'win32':
    _original_popen = subprocess.Popen
    
    def _silent_popen(*args, **kwargs):
        """Wrapper untuk subprocess.Popen yang menyembunyikan console window"""
        if 'startupinfo' not in kwargs:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            kwargs['startupinfo'] = startupinfo
        
        # Tambahkan CREATE_NO_WINDOW flag
        if 'creationflags' not in kwargs:
            kwargs['creationflags'] = 0
        kwargs['creationflags'] |= subprocess.CREATE_NO_WINDOW
        
        return _original_popen(*args, **kwargs)
    
    # Ganti subprocess.Popen dengan versi silent
    subprocess.Popen = _silent_popen

from plyer import filechooser
from rapidocr_onnxruntime import RapidOCR
import numpy as np
from PIL import Image
import pdf2image
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
import cv2
from collections import defaultdict
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from datetime import datetime

# Tentukan path absolut ke model ONNX
det_model_path = os.path.join(application_path, 'models', 'ch_PP-OCRv4_det_infer.onnx')
cls_model_path = os.path.join(application_path, 'models', 'ch_ppocr_mobile_v2.0_cls_infer.onnx')
rec_model_path = os.path.join(application_path, 'models', 'ch_PP-OCRv4_rec_infer.onnx')
# Inisialisasi engine OCR
print("Memuat model OCR...")
engine = RapidOCR(
    det_model_path=det_model_path,
    cls_model_path=cls_model_path,
    rec_model_path=rec_model_path,
    rec_batch_num=6,
    use_angle_cls=True,
    det_limit_side_len=960,
    det_db_thresh=0.3,
    det_db_box_thresh=0.5,
    det_db_unclip_ratio=1.6
)
print("Model OCR berhasil dimuat.")

# --- Logika Backend (Pemrosesan Gambar & PDF) ---

def preprocess_image_for_ocr(img_array):
    """Mempersiapkan gambar untuk OCR (grayscale, thresholding, sharpen) agar teks lebih jelas."""
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    edge_enhanced = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
    sharpened = cv2.addWeighted(gray, 1.5, edge_enhanced, -0.5, 0)
    return sharpened

def detect_lines(img_gray, min_length=50, orientation='horizontal', line_type='any'):
    """Mendeteksi garis horizontal atau vertikal menggunakan pemrosesan morfologi dan analisis profil."""
    height, width = img_gray.shape
    binary = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    if line_type in ['dotted', 'any']:
        if orientation == 'horizontal':
            profile = np.sum(binary, axis=1)
            smooth_profile = gaussian_filter1d(profile, sigma=2)
            lines_img = np.zeros_like(binary)
            peaks, _ = find_peaks(smooth_profile, height=width*0.1, distance=15)
            for peak in peaks:
                if peak > 0 and peak < height:
                    lines_img[peak, :] = 255
        else:
            profile = np.sum(binary, axis=0)
            smooth_profile = gaussian_filter1d(profile, sigma=2)
            lines_img = np.zeros_like(binary)
            peaks, _ = find_peaks(smooth_profile, height=height*0.1, distance=15)
            for peak in peaks:
                if peak > 0 and peak < width:
                    lines_img[:, peak] = 255
        if line_type == 'any':
            if orientation == 'horizontal':
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_length, 1))
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_length))
            morph_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            combined_lines = cv2.bitwise_or(lines_img, morph_lines)
            return combined_lines
        return lines_img
    else:
        if orientation == 'horizontal':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_length, 1))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_length))
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        return lines

def detect_tables(img_array, line_sensitivity=25, use_grid=True, dotted_line_support=True):
    """Mendeteksi area tabel. Prioritaskan deteksi berbasis grid (garis). Jika gagal, gunakan deteksi berbasis tata letak teks."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    if use_grid:
        if dotted_line_support:
            horizontal_lines = detect_lines(gray, min_length=line_sensitivity, orientation='horizontal', line_type='any')
            vertical_lines = detect_lines(gray, min_length=line_sensitivity, orientation='vertical', line_type='any')
        else:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_sensitivity, 1))
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_sensitivity))
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        table_grid = cv2.bitwise_or(horizontal_lines, vertical_lines)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_grid = cv2.dilate(table_grid, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        table_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > (img_array.shape[0] * img_array.shape[1]) / 30:
                aspect_ratio = w / float(h)
                if 0.2 < aspect_ratio < 5:
                    table_contours.append((x, y, w, h, horizontal_lines, vertical_lines))
        if table_contours:
            return table_contours
    # Fallback jika deteksi grid gagal
    return text_based_table_detection(img_array, gray)

def text_based_table_detection(img_array, gray):
    """Menganalisis hasil OCR untuk mengelompokkan teks yang berdekatan secara horizontal dan vertikal untuk 'menebak' batas tabel."""
    preprocessed_img = preprocess_image_for_ocr(img_array)
    ocr_result = engine(preprocessed_img)
    if not ocr_result or len(ocr_result[0]) == 0:
        return []
    text_boxes = []
    for box, text, confidence in ocr_result[0]:
        if text.strip():
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)
            text_boxes.append({
                'box': box, 'text': text, 'confidence': confidence,
                'x': x_min, 'y': y_min, 'width': x_max - x_min, 'height': y_max - y_min,
                'x_center': (x_min + x_max) / 2, 'y_center': (y_min + y_max) / 2
            })
    if not text_boxes:
        return []
    text_boxes_sorted_y = sorted(text_boxes, key=lambda b: b['y'])
    avg_height = sum(box['height'] for box in text_boxes) / len(text_boxes)
    row_threshold = max(avg_height * 1.5, 15)
    rows = []
    current_row = [text_boxes_sorted_y[0]]
    current_y = text_boxes_sorted_y[0]['y_center']
    for box in text_boxes_sorted_y[1:]:
        if abs(box['y_center'] - current_y) <= row_threshold:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
            current_y = box['y_center']
    if current_row:
        rows.append(current_row)
    if len(rows) < 2:
        return []
    table_candidates = []
    min_rows_for_table = 3
    for i in range(len(rows) - min_rows_for_table + 1):
        for k in range(len(rows) - i - min_rows_for_table + 1):
            potential_table_rows = rows[i:i+min_rows_for_table+k]
            row_lengths = [len(row) for row in potential_table_rows]
            if max(row_lengths) - min(row_lengths) <= 1:
                all_boxes = [box for row in potential_table_rows for box in row]
                x_min = min(box['x'] for box in all_boxes)
                y_min = min(box['y'] for box in all_boxes)
                x_max = max(box['x'] + box['width'] for box in all_boxes)
                y_max = max(box['y'] + box['height'] for box in all_boxes)
                x_min = max(0, x_min - 10)
                y_min = max(0, y_min - 10)
                x_max = min(img_array.shape[1], x_max + 10)
                y_max = min(img_array.shape[0], y_max + 10)
                empty_lines = np.zeros_like(gray)
                table_candidates.append((int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min), empty_lines, empty_lines))
    return table_candidates

def detect_cells_from_text(text_items, table_width, table_height):
    """Mengelompokkan koordinat X dan Y dari teks untuk menyimpulkan batas baris dan kolom (grid imajiner)."""
    if not text_items:
        return None, None
    avg_height = sum(item['height'] for item in text_items) / len(text_items)
    avg_width = sum(item['width'] for item in text_items) / len(text_items)
    sorted_by_y = sorted(text_items, key=lambda t: t['y_center'])
    y_centers = [t['y_center'] for t in sorted_by_y]
    y_threshold = max(avg_height * 0.8, 10)
    def cluster_coordinates(coords, threshold):
        clusters = []
        if not coords:
            return clusters
        current_cluster = [coords[0]]
        current_value = coords[0]
        for val in coords[1:]:
            if abs(val - current_value) <= threshold:
                current_cluster.append(val)
            else:
                if current_cluster:
                    clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [val]
                current_value = val
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))
        return clusters
    row_centers = cluster_coordinates(y_centers, y_threshold)
    row_boundaries = [0]
    for i in range(len(row_centers) - 1):
        boundary = int((row_centers[i] + row_centers[i+1]) / 2)
        row_boundaries.append(boundary)
    row_boundaries.append(int(table_height))
    sorted_by_x = sorted(text_items, key=lambda t: t['x_center'])
    x_centers = [t['x_center'] for t in sorted_by_x]
    x_threshold = max(avg_width * 0.8, 20)
    col_centers = cluster_coordinates(x_centers, x_threshold)
    col_boundaries = [0]
    for i in range(len(col_centers) - 1):
        boundary = int((col_centers[i] + col_centers[i+1]) / 2)
        col_boundaries.append(boundary)
    col_boundaries.append(int(table_width))
    return row_boundaries, col_boundaries

def process_table_ocr(ocr_results, table_rect, use_inferred_grid=True):
    """Mengekstrak teks di dalam batas tabel dan memetakannya ke struktur sel (baris/kolom) berdasarkan grid yang disimpulkan."""
    x_table, y_table, w_table, h_table = table_rect[:4]
    if not ocr_results or len(ocr_results[0]) == 0:
        return None
    table_texts = []
    for box, text, confidence in ocr_results[0]:
        x_center = sum([p[0] for p in box]) / 4
        y_center = sum([p[1] for p in box]) / 4
        if ((x_table - 5 <= x_center <= x_table + w_table + 5) and
            (y_table - 5 <= y_center <= y_table + h_table + 5)):
            x_min = min([p[0] for p in box])
            y_min = min([p[1] for p in box])
            if text.strip():
                table_texts.append({
                    'text': text.strip(), 'confidence': confidence,
                    'x': x_min - x_table, 'y': y_min - y_table,
                    'width': max([p[0] for p in box]) - x_min, 'height': max([p[1] for p in box]) - y_min,
                    'x_center': x_center - x_table, 'y_center': y_center - y_table, 'original_box': box
                })
    if not table_texts:
        return None
    if use_inferred_grid:
        row_boundaries, col_boundaries = detect_cells_from_text(table_texts, w_table, h_table)
        if row_boundaries and col_boundaries and len(row_boundaries) > 1 and len(col_boundaries) > 1:
            num_rows = len(row_boundaries) - 1
            num_cols = len(col_boundaries) - 1
            table_data = [[''] * num_cols for _ in range(num_rows)]
            for text_item in table_texts:
                row_idx = None
                for i in range(len(row_boundaries) - 1):
                    if row_boundaries[i] <= text_item['y_center'] < row_boundaries[i+1]:
                        row_idx = i
                        break
                col_idx = None
                for i in range(len(col_boundaries) - 1):
                    if col_boundaries[i] <= text_item['x_center'] < col_boundaries[i+1]:
                        col_idx = i
                        break
                if row_idx is not None and col_idx is not None:
                    if row_idx < num_rows and col_idx < num_cols:
                        if table_data[row_idx][col_idx]:
                            table_data[row_idx][col_idx] += ' ' + text_item['text']
                        else:
                            table_data[row_idx][col_idx] = text_item['text']
            return table_data
    # Fallback jika grid imajiner gagal
    return reconstruct_table(table_texts, w_table, h_table)

def reconstruct_table(texts, table_width, table_height):
    """Metode alternatif untuk membangun tabel jika grid gagal, hanya berdasarkan pengelompokan Y (baris) dan X (kolom)."""
    if not texts:
        return None
    texts_sorted_by_y = sorted(texts, key=lambda t: t['y'])
    avg_text_height = sum(t['height'] for t in texts) / len(texts)
    row_threshold = max(avg_text_height * 0.8, 12)
    rows = []
    if not texts_sorted_by_y:
        return []
    current_row = [texts_sorted_by_y[0]]
    row_y = texts_sorted_by_y[0]['y_center']
    for t in texts_sorted_by_y[1:]:
        if abs(t['y_center'] - row_y) <= row_threshold:
            current_row.append(t)
        else:
            if current_row:
                rows.append(sorted(current_row, key=lambda t: t['x']))
            current_row = [t]
            row_y = t['y_center']
    if current_row:
        rows.append(sorted(current_row, key=lambda t: t['x']))
    all_x_centers = [t['x_center'] for row in rows for t in row]
    def cluster_x_centers(x_centers):
        if not x_centers:
            return []
        avg_text_width = sum(t['width'] for t in texts) / len(texts)
        col_threshold = max(avg_text_width * 1.5, 20)
        x_centers = sorted(x_centers)
        clusters = [[x_centers[0]]]
        for x in x_centers[1:]:
            if abs(x - clusters[-1][-1]) <= col_threshold:
                clusters[-1].append(x)
            else:
                clusters.append([x])
        return [sum(cluster) / len(cluster) for cluster in clusters]
    col_centers = cluster_x_centers(all_x_centers)
    num_cols = len(col_centers)
    if num_cols == 0:
        num_cols = 1
        col_centers = [table_width / 2]
    table_data = []
    for row_texts in rows:
        row_data = [''] * num_cols
        for text in row_texts:
            col_idx = min(range(num_cols),
                          key=lambda i: abs(text['x_center'] - col_centers[i]))
            if row_data[col_idx]:
                row_data[col_idx] += ' ' + text['text']
            else:
                row_data[col_idx] = text['text']
        table_data.append(row_data)
    return table_data

def create_tables_export(tables, output_format='xlsx', sheet_names=None):
    """Mengambil data tabel (list of DataFrame) dan mengekspornya ke format file yang diminta (XLSX, CSV, ODS) dalam bentuk bytes."""
    output = io.BytesIO()
    def auto_adjust_excel_column_widths(writer, sheet_name, dataframe):
        worksheet = writer.book[sheet_name]
        for idx, col_name in enumerate(dataframe.columns):
            max_length = len(str(col_name))
            for cell in dataframe[col_name].astype(str):
                if len(cell) > max_length:
                    max_length = len(cell)
            adjusted_width = (max_length + 2)
            worksheet.column_dimensions[get_column_letter(idx + 1)].width = adjusted_width

    file_ext = output_format
    mime_type = "application/octet-stream"

    if output_format == 'xlsx':
        try:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                if tables:
                    for i, df in enumerate(tables):
                        safe_sheet_name = "".join(c for c in sheet_names[i] if c.isalnum() or c in (' ', '_')).rstrip()[:31] if sheet_names and i < len(sheet_names) else f"Tabel_{i+1}"
                        temp_name = safe_sheet_name
                        count = 1
                        while temp_name in writer.book.sheetnames:
                            temp_name = f"{safe_sheet_name}_{count}"[:31]
                            count += 1
                        safe_sheet_name = temp_name
                        df.to_excel(writer, sheet_name=safe_sheet_name, index=False, header=False)
                        auto_adjust_excel_column_widths(writer, safe_sheet_name, df)
                else:
                    wb = Workbook()
                    wb.active.title = "Tidak_Ada_Tabel"
                    wb.save(output)
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            file_ext = "xlsx"
        except Exception as e:
            print(f"Gagal membuat XLSX: {e}")
            output = io.BytesIO()

    elif output_format == 'csv':
        if not tables:
            output.write(b'')
        else:
             # Hanya ekspor tabel pertama jika CSV
             output.write(tables[0].to_csv(index=False, header=False, encoding='utf-8-sig').encode('utf-8-sig'))
        mime_type = "text/csv"
        file_ext = "csv"

    elif output_format == 'ods':
        try:
            with pd.ExcelWriter(output, engine='odf') as writer:
                if tables:
                    for i, df in enumerate(tables):
                        sheet_name = "".join(c for c in sheet_names[i] if c.isalnum() or c in (' ', '_')).rstrip()[:31] if sheet_names and i < len(sheet_names) else f"Tabel_{i+1}"
                        df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            mime_type = "application/vnd.oasis.opendocument.spreadsheet"
            file_ext = "ods"
        except Exception as e:
            print(f"Gagal membuat ODS: {e}")
            output = io.BytesIO()

    output.seek(0)
    return output.getvalue(), mime_type, file_ext

def process_image_for_tables(img_array):
    """Orkestrator utama untuk satu gambar: menjalankan deteksi tabel, lalu OCR, lalu mem-parsing tabel."""
    use_grid = True
    line_sensitivity = 25
    dotted_line_support = True
    small_text_enhancement = True
    if small_text_enhancement:
        preprocessed_img = preprocess_image_for_ocr(img_array)
        ocr_result = engine(preprocessed_img)
    else:
        ocr_result = engine(img_array)
    table_rects = detect_tables(img_array, line_sensitivity, use_grid, dotted_line_support)
    if not table_rects:
        # Jika tidak ada tabel terdeteksi, kembalikan semua teks sebagai satu kolom
        if isinstance(ocr_result, tuple) and len(ocr_result) >= 1 and ocr_result[0]:
            all_texts = [item[1] for item in ocr_result[0]]
            if all_texts:
                return [pd.DataFrame({"Teks": all_texts})]
        return []
    tables = []
    for i, table_rect in enumerate(table_rects):
        table_data = process_table_ocr(ocr_result, table_rect, use_inferred_grid=True)
        if table_data:
            df = pd.DataFrame(table_data)
            tables.append(df)
    return tables


def process_pdf(pdf_path, progress_callback=None):
    """Mengkonversi setiap halaman PDF menjadi gambar, lalu memproses setiap gambar untuk tabel."""
    all_tables = []
    sheet_names = []

    try:
        pdf_info = pdf2image.pdfinfo_from_path(pdf_path)
        total_pages = pdf_info.get('Pages', 0)
        if total_pages == 0:
             raise ValueError("Tidak dapat menentukan jumlah halaman.")
    except Exception as e:
        print(f"Gagal mendapatkan info PDF untuk {pdf_path}: {e}")
        if progress_callback:
            progress_callback(0, 1, error=True, message=f"Gagal Info")
        return [], []

    for i in range(1, total_pages + 1):
        if progress_callback:
            # Kirim progres ke UI
            progress_callback(i, total_pages)

        try:
            page_image = pdf2image.convert_from_path(
                pdf_path,
                dpi=300,
                first_page=i,
                last_page=i,
                timeout=60
            )[0]

            img_array = np.array(page_image.convert('RGB'))
            page_tables = process_image_for_tables(img_array)

            for j, df in enumerate(page_tables):
                all_tables.append(df)
                sheet_names.append(f"Halaman {i}")

            # Bersihkan memori
            del page_image
            del img_array
            gc.collect()
        except Exception as e:
            print(f"Gagal memproses halaman {i} dari {pdf_path}: {e}")
            if progress_callback:
                 progress_callback(i, total_pages, error=True, message=f"Hal {i} Gagal")
            continue

    return all_tables, sheet_names

def process_image(image_path):
    """Wrapper sederhana untuk memproses file gambar tunggal."""
    try:
        image = Image.open(image_path)
        img_array = np.array(image.convert('RGB'))
        tables = process_image_for_tables(img_array)
        del image
        del img_array
        gc.collect()
        return tables
    except Exception as e:
        print(f"Gagal memproses gambar {image_path}: {e}")
        return []

# --- KivyMD UI Definition (KV Lang) ---
Window.clearcolor = (1, 1, 1, 1)

KV = """
# --- Definisi Cell (MainScreen) ---
<GridCellLabel@MDLabel>:
    theme_text_color: "Primary"
    halign: 'left'
    valign: 'middle'
    padding_x: dp(5)
    shorten: True
    text_size: (self.width - dp(10), None) if self.width > dp(20) else (dp(10), None)
    size_hint_y: None
    height: dp(40)
    canvas.before:
        Color:
            rgba: 1, 1, 1, 1
        Rectangle:
            pos: self.pos
            size: self.size

<GridCellLabelRight@GridCellLabel>:
    halign: 'center'
    padding_x: 0
    shorten: False
    text_size: (self.width, None)

<WaitingCellWithDelete>:
    orientation: 'horizontal'
    padding: dp(5)
    spacing: dp(5)
    md_bg_color: app.theme_cls.bg_normal
    
    MDLabel:
        text: 'Menunggu'
        theme_text_color: "Hint"
        halign: 'center'
        valign: 'middle'
        size_hint_x: 0.7
    
    MDIconButton:
        icon: 'close'
        theme_text_color: "Error"
        pos_hint: {'center_y': 0.5}
        size_hint: (None, None)
        size: (dp(30), dp(30))
        on_release: app.root.main_screen.remove_file_from_list(root.file_index)

<DownloadButtonCell>:
    orientation: 'vertical'
    padding: dp(5)
    md_bg_color: app.theme_cls.bg_normal
    
    MDBoxLayout:
        orientation: 'horizontal'
        Widget:
        MDFillRoundFlatIconButton:
            text: 'Unduh'
            icon: 'download'
            md_bg_color: app.theme_cls.accent_color
            size_hint: (None, None)
            size: (dp(100), dp(36))
            pos_hint: {'center_y': 0.5}
            on_release: app.root.main_screen.download_single_file(root.file_index)
        Widget:

<FailedStatusCell@MDBoxLayout>:
    padding: dp(5)
    md_bg_color: app.theme_cls.bg_normal
    
    MDBoxLayout:
        orientation: 'horizontal'
        Widget:
        MDLabel:
            text: 'Gagal'
            theme_text_color: "Error"
            bold: True
            halign: 'center'
            valign: 'middle'
            size_hint: (None, None)
            size: (dp(80), dp(36))
        Widget:

<NoTablesStatusCell@MDBoxLayout>:
    padding: dp(5)
    md_bg_color: app.theme_cls.bg_normal
    
    MDBoxLayout:
        orientation: 'horizontal'
        Widget:
        MDLabel:
            text: 'Tak Ada Tabel'
            theme_text_color: "Hint"
            halign: 'center'
            valign: 'middle'
            size_hint: (None, None)
            size: (dp(100), dp(36))
        Widget:

<ProgressTextCell@GridCellLabelRight>:

# --- Definisi Cell (ResultScreen) ---
<ResultCellLabel@MDLabel>:
    theme_text_color: "Primary"
    halign: 'left'
    valign: 'middle'
    padding_x: dp(5)
    shorten: True
    text_size: (self.width - dp(10), None) if self.width > dp(20) else (dp(10), None)
    size_hint_y: None
    height: dp(40)
    canvas.before:
        Color:
            rgba: 1, 1, 1, 1
        Rectangle:
            pos: self.pos
            size: self.size

<ResultCellLabelRight@ResultCellLabel>:
    halign: 'center'
    padding_x: 0
    shorten: False
    text_size: (self.width, None)

<ResultSuccessCell>:
    orientation: 'vertical'
    padding: dp(5)
    md_bg_color: app.theme_cls.bg_normal
    
    MDBoxLayout:
        orientation: 'horizontal'
        Widget:
        MDFillRoundFlatIconButton:
            text: 'Unduh'
            icon: 'download'
            md_bg_color: app.theme_cls.accent_color
            size_hint: (None, None)
            size: (dp(100), dp(36))
            pos_hint: {'center_y': 0.5}
            on_release: app.root.main_screen.download_single_file(root.file_index)
        Widget:

<ResultFailedCell@MDBoxLayout>:
    padding: dp(5)
    md_bg_color: app.theme_cls.bg_normal
    
    MDBoxLayout:
        orientation: 'horizontal'
        Widget:
        MDLabel:
            text: 'Gagal'
            theme_text_color: "Error"
            bold: True
            halign: 'center'
            valign: 'middle'
            size_hint: (None, None)
            size: (dp(80), dp(36))
        Widget:

<ResultNoTablesCell@MDBoxLayout>:
    padding: dp(5)
    md_bg_color: app.theme_cls.bg_normal
    
    MDBoxLayout:
        orientation: 'horizontal'
        Widget:
        MDLabel:
            text: 'Tak Ada Tabel'
            theme_text_color: "Hint"
            halign: 'center'
            valign: 'middle'
            size_hint: (None, None)
            size: (dp(100), dp(36))
        Widget:

# --- LAYAR UTAMA (KONVERSI) ---
<MainScreen>:
    MDBoxLayout:
        orientation: 'vertical'
        padding: dp(20)
        spacing: dp(15)

        MDLabel:
            text: 'Konverter PDF ke Spreadsheet'
            font_style: 'H5'
            bold: True
            halign: 'center'
            size_hint_y: None
            height: self.texture_size[1]

        MDLabel:
            id: status_label
            text: 'Silakan pilih file (maks 5) dan klik Konversi.'
            font_style: 'Subtitle1'
            halign: 'center'
            size_hint_y: None
            height: self.texture_size[1]
            padding_y: dp(10)

        # Toolbar Tombol
        MDBoxLayout:
            id: button_toolbar
            size_hint_y: None
            height: dp(48)
            spacing: dp(10)
            adaptive_height: True
            
            MDFillRoundFlatIconButton:
                id: select_button
                text: 'Pilih File'
                icon: 'file-upload'
                on_release: root.select_file()
                md_bg_color: app.theme_cls.accent_color
                size_hint_x: 0.33

            MDFillRoundFlatIconButton:
                id: format_spinner_button
                text: 'Pilih Format'
                icon: 'menu-down'
                on_release: root.format_menu.open()
                size_hint_x: 0.33

            MDFillRoundFlatIconButton:
                id: convert_button
                text: 'Konversi Sekarang'
                icon: 'cog-refresh'
                on_release: root.start_conversion()
                size_hint_x: 0.33
                disabled: True

        # Kontainer Daftar File (Processing)
        MDBoxLayout:
            id: file_list_container
            orientation: 'vertical'
            size_hint_y: None
            height: 0
            spacing: dp(1)
            opacity: 0

            MDLabel:
                text: "File untuk Dikonversi"
                font_style: 'H6'
                halign: 'left'
                size_hint_y: None
                height: self.texture_size[1]
                padding_y: dp(5)
            
            # Header
            MDGridLayout:
                cols: 3
                size_hint_y: None
                height: dp(40)
                md_bg_color: (0.92, 0.92, 0.92, 1) # Abu-abu muda
                
                MDLabel:
                    text: "Nama File"
                    bold: True
                    halign: "left"
                    valign: "middle"
                    padding_x: dp(5)
                    size_hint_x: 0.6
                MDLabel:
                    text: "Ukuran"
                    bold: True
                    halign: "center"
                    valign: "middle"
                    size_hint_x: 0.15
                MDLabel:
                    text: "Progres"
                    bold: True
                    halign: "center"
                    valign: "middle"
                    size_hint_x: 0.25

            # Scrollable List
            MDScrollView:
                size_hint_y: 1
                bar_width: dp(10)
                MDGridLayout:
                    id: file_list_grid
                    cols: 3
                    size_hint_y: None
                    height: self.minimum_height
                    row_default_height: dp(40)
                    row_force_default: True
                    spacing: dp(1)
                    md_bg_color: app.theme_cls.divider_color
        
        # Spacer
        MDBoxLayout:
            id: main_screen_spacer
            size_hint_y: 1

# --- LAYAR HASIL (UNDUH) ---
<ResultScreen>:
    MDBoxLayout:
        orientation: 'vertical'
        padding: dp(20)
        spacing: dp(15)
        
        MDLabel:
            text: "Proses Selesai"
            font_style: 'H5'
            bold: True
            halign: 'center'
            size_hint_y: None
            height: self.texture_size[1]
            padding_y: dp(10)

        MDLabel:
            id: result_status_label
            text: "Silakan unduh hasil Anda."
            font_style: 'Subtitle1'
            halign: 'center'
            size_hint_y: None
            height: self.texture_size[1]
            padding_y: dp(5)

        # Daftar File Hasil
        MDLabel:
            text: "File Hasil Konversi"
            font_style: 'H6'
            halign: 'left'
            size_hint_y: None
            height: self.texture_size[1]
            padding_y: dp(5)
        
        # Header Hasil
        MDScrollView:
            size_hint_y: 1
            bar_width: dp(10)
            
            MDBoxLayout:
                orientation: 'vertical'
                size_hint_y: None
                height: self.minimum_height
                spacing: 0
                
                MDGridLayout:
                    cols: 3
                    size_hint_y: None
                    height: dp(40)
                    md_bg_color: (0.92, 0.92, 0.92, 1) # Abu-abu muda
                    
                    MDLabel:
                        text: "Nama File"
                        bold: True
                        halign: "left"
                        valign: "middle"
                        padding_x: dp(5)
                        size_hint_x: 0.6
                    MDLabel:
                        text: "Ukuran"
                        bold: True
                        halign: "center"
                        valign: "middle"
                        size_hint_x: 0.15
                    MDLabel:
                        text: "Status"
                        bold: True
                        halign: "center"
                        valign: "middle"
                        size_hint_x: 0.25
                
                MDGridLayout:
                    id: result_file_list_grid
                    cols: 3
                    size_hint_y: None
                    height: self.minimum_height
                    row_default_height: dp(40)
                    row_force_default: True
                    spacing: dp(1)
                    md_bg_color: app.theme_cls.divider_color

        # Layout container untuk menengahkan tombol
        MDBoxLayout:
            orientation: 'horizontal'
            adaptive_height: True
            adaptive_width: True
            pos_hint: {'center_x': 0.5}
            spacing: dp(15)
            padding: dp(15), 0
            
            MDFillRoundFlatIconButton:
                text: "Konversi Lagi"
                icon: 'refresh'
                on_release: app.root.go_to_main_screen()
                md_bg_color: app.theme_cls.accent_color
            
            MDFillRoundFlatIconButton:
                id: save_button_result
                text: 'Unduh Semua'
                icon: 'download-multiple'
                on_release: app.root.main_screen.save_result() 
                disabled: True

# --- SCREEN MANAGER (ROOT WIDGET) ---
<RootScreenManager>:
    main_screen: main_screen_id
    result_screen: result_screen_id

    MainScreen:
        id: main_screen_id
        name: 'main'
    
    ResultScreen:
        id: result_screen_id
        name: 'result'
"""

# --- Definisi Kelas Widget Kustom ---
class DownloadButtonCell(MDBoxLayout):
    """Widget kustom untuk sel di grid yang berisi tombol unduh."""
    file_index = NumericProperty(-1)

class ResultSuccessCell(MDBoxLayout):
    """Widget kustom untuk sel di grid hasil yang berisi tombol unduh."""
    file_index = NumericProperty(-1)
    
class WaitingCellWithDelete(MDBoxLayout):
    """Sel kustom yang menampilkan 'Menunggu' dengan tombol X untuk menghapus."""
    file_index = NumericProperty(-1)
    
# Untuk MainScreen
class GridCellLabel(MDLabel):
    """Sel label dasar untuk grid di MainScreen."""
    pass

class GridCellLabelRight(GridCellLabel):
    """Sel label dengan perataan tengah/kanan untuk grid di MainScreen."""
    pass

class ProgressTextCell(GridCellLabelRight):
    """Sel label spesifik untuk menampilkan status progres di MainScreen."""
    pass

class FailedStatusCell(MDBoxLayout):
    """Sel kustom yang menampilkan status 'Gagal' di grid MainScreen."""
    pass

class NoTablesStatusCell(MDBoxLayout):
    """Sel kustom yang menampilkan status 'Tak Ada Tabel' di grid MainScreen."""
    pass

# Untuk ResultScreen
class ResultCellLabel(MDLabel):
    """Sel label dasar untuk grid di ResultScreen."""
    pass

class ResultCellLabelRight(ResultCellLabel):
    """Sel label dengan perataan tengah/kanan untuk grid di ResultScreen."""
    pass

class ResultFailedCell(MDBoxLayout):
    """Sel kustom yang menampilkan status 'Gagal' di grid ResultScreen."""
    pass

class ResultNoTablesCell(MDBoxLayout):
    """Sel kustom yang menampilkan status 'Tak Ada Tabel' di grid ResultScreen."""
    pass

Builder.load_string(KV)

# --- Definisi Layar ---
class ResultScreen(MDScreen):
    """Definisi kelas untuk layar yang menampilkan hasil (kosong, hanya mengandalkan KV)."""
    pass

class RootScreenManager(MDScreenManager):
    """Pengelola layar utama, mengatur transisi antara 'main' dan 'result'."""
    main_screen = ObjectProperty(None)
    result_screen = ObjectProperty(None)

    def go_to_main_screen(self):
        """Kembali ke layar utama dan mereset statusnya."""
        if self.main_screen:
            self.main_screen.reset_ui_state()
        
        if self.result_screen:
            self.result_screen.ids.result_file_list_grid.clear_widgets()
            
        self.current = 'main'
    
    def go_to_result_screen(self):
        """Pindah ke layar hasil."""
        self.current = 'result'


# --- Logika Aplikasi Utama (Layar Utama) ---

class MainScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Inisialisasi variabel state aplikasi
        self.extracted_tables = {}
        self.sheet_names = {}
        self.output_data = {}
        self.file_list = []
        self.current_processing_index = -1
        self.is_processing = False
        
        # Ikat fungsi on_file_drop ke window untuk fungsionalitas drag-and-drop
        self._dropped_files = []
        self._drop_event = None
        Window.bind(on_dropfile=self.on_file_drop)
        
        # Format output default (None berarti harus dipilih dulu)
        self.selected_format = None
        Clock.schedule_once(self.create_format_menu, 0)
        
        # Atur batas maksimal file
        self.max_files = 5
        self.dialog = None

    def show_alert(self, title, text):
        """Menampilkan dialog peringatan."""
        if self.dialog:
            self.dialog.dismiss()
        
        # Buat label teks dengan binding untuk halign
        text_label = MDLabel(
            text=text,
            theme_text_color="Primary",
            halign='center',
            valign='top',
            size_hint_y=None,
            height=dp(100),
            markup=True
        )
        
        # Binding untuk halign='center' bekerja
        text_label.bind(
            width=lambda instance, value: setattr(instance, 'text_size', (value, None))
        )
        
        # Container untuk content
        content_box = MDBoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=dp(150),
            padding=[dp(20), dp(10), dp(20), dp(10)],
            spacing=dp(15)
        )
        content_box.add_widget(text_label)
        
        # Container untuk tombol dengan spacer kiri-kanan
        button_box = MDBoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(40)
        )
        
        # Spacer kiri
        button_box.add_widget(Widget())
        
        # Tombol OK di tengah
        ok_button = MDFillRoundFlatButton(
            text="OK",
            size_hint=(None, None),
            size=(dp(100), dp(40)),
            on_release=lambda x: self.dialog.dismiss()
        )
        button_box.add_widget(ok_button)
        
        # Spacer kanan
        button_box.add_widget(Widget())
        
        # Tambahkan button box ke content
        content_box.add_widget(button_box)
        
        # Buat dialog tanpa parameter buttons (karena kita sudah masukkan ke content)
        self.dialog = MDDialog(
            title=title,
            type="custom",
            content_cls=content_box,
            size_hint=(0.85, None),
            height=dp(300),
            auto_dismiss=False
        )
        
        # Atur title alignment dengan binding
        def set_title_center(*args):
            try:
                title_widget = self.dialog.ids.title
                title_widget.halign = 'center'
                title_widget.bind(
                    width=lambda inst, val: setattr(inst, 'text_size', (val, None))
                )
            except:
                pass
        
        # Delay sedikit untuk memastikan widget sudah dibuat
        Clock.schedule_once(set_title_center, 0.1)
        
        self.dialog.open()

    def create_format_menu(self, dt):
        """Membangun item menu dropdown untuk pilihan format (XLSX, CSV, ODS)."""
        menu_items = [
            {
                "text": "XLSX",
                "viewclass": "OneLineListItem",
                "height": dp(48),
                "on_release": lambda x="XLSX": self.set_format(x),
            },
            {
                "text": "CSV",
                "viewclass": "OneLineListItem",
                "height": dp(48),
                "on_release": lambda x="CSV": self.set_format(x),
            },
            {
                "text": "ODS",
                "viewclass": "OneLineListItem",
                "height": dp(48),
                "on_release": lambda x="ODS": self.set_format(x),
            },
        ]
        self.format_menu = MDDropdownMenu(
            caller=self.ids.format_spinner_button,
            items=menu_items,
            width_mult=3,
        )

    def set_format(self, text_item):
        """Dipanggil saat format dipilih. Menyimpan pilihan & mengaktifkan tombol konversi."""
        self.ids.format_spinner_button.text = text_item
        self.selected_format = text_item
        self.format_menu.dismiss()
        self.ids.convert_button.disabled = False # Aktifkan tombol konversi

    def _clear_results(self):
        """Membersihkan data hasil konversi sebelumnya dari memori."""
        self.extracted_tables.clear()
        self.sheet_names.clear()
        self.output_data.clear()
        for f_info in self.file_list:
             f_info['output_data'] = None
             f_info['tables'] = None
             f_info['sheet_names'] = None
        gc.collect()
        print("Hasil ekstraksi sebelumnya telah dibersihkan.")

    def get_file_size_str(self, file_path):
        """Mengubah ukuran file (bytes) menjadi string yang mudah dibaca (KB, MB)."""
        try:
            size_bytes = os.path.getsize(file_path)
            if size_bytes < 1024: return f"{size_bytes} B"
            elif size_bytes < 1024**2: return f"{size_bytes/1024:.1f} KB"
            elif size_bytes < 1024**3: return f"{size_bytes/1024**2:.1f} MB"
            else: return f"{size_bytes/1024**3:.1f} GB"
        except Exception: return "N/A"

    @mainthread
    def reset_ui_state(self):
        """Mereset UI ke kondisi awal saat kembali ke layar utama."""
        self.ids.status_label.text = f"Silakan pilih file (maks {self.max_files}) dan klik Konversi."
        self.ids.file_list_grid.clear_widgets()
        self.file_list = []
        
        # Sembunyikan container file list
        self.ids.file_list_container.opacity = 0
        self.ids.file_list_container.size_hint_y = None
        self.ids.file_list_container.height = 0
        self.ids.main_screen_spacer.size_hint_y = 1
        self.ids.main_screen_spacer.height = dp(1)
        
        self.ids.button_toolbar.opacity = 1
        self.ids.button_toolbar.size_hint_y = None
        self.ids.button_toolbar.height = dp(48)
        
        self.ids.select_button.disabled = False
        
        # Reset tombol format dan konversi
        self.ids.convert_button.disabled = True
        self.ids.format_spinner_button.text = "Pilih Format"
        self.selected_format = None
        
        self.is_processing = False
        self.current_processing_index = -1
        self._clear_results()

    def select_file(self):
        """Membuka dialog pilih file, tetapi hanya jika batas file belum tercapai."""
        if self.is_processing: return

        # Cek apakah sudah penuh SEBELUM membuka dialog
        if len(self.file_list) >= self.max_files:
            self.show_alert("Batas File Tercapai", f"Anda sudah memilih {self.max_files} file. Batas maksimal adalah {self.max_files} file.")
            return

        # Logika "tambah file", bukan "ganti file"
        filechooser.open_file(
            on_selection=self.handle_selection,
            filters=[("PDF dan Gambar", "*.pdf", "*.jpeg", "*.png")],
            multiple=True
        )

    def handle_selection(self, selection):
        """Logika utama untuk menangani file yang dipilih. Mencegah duplikat dan melebihi batas."""
        if not selection:
            return

        current_file_count = len(self.file_list)
        allowed_new_count = self.max_files - current_file_count

        if allowed_new_count <= 0:
            self.show_alert("Batas File Tercapai", f"Anda sudah memilih {self.max_files} file. Tidak dapat menambah lagi.")
            return

        # Filter file duplikat (berdasarkan path)
        existing_paths = [f['path'] for f in self.file_list]
        new_unique_selection = [p for p in selection if p not in existing_paths]
        
        files_to_add = new_unique_selection[:allowed_new_count]
        files_rejected_count = len(selection) - len(files_to_add)

        if files_rejected_count > 0:
            self.show_alert(
                "Batas File Terlampaui",
                f"Anda memilih {len(selection)} file.\n"
                f"Batas maksimal adalah {self.max_files} file.\n"
                f"{len(files_to_add)} file berhasil ditambahkan.\n"
                f"{files_rejected_count} file ditolak (karena batas atau duplikat)."
            )
        
        if files_to_add:
            # Panggil fungsi untuk memperbarui UI (dijalankan di mainthread)
            self._append_files_to_ui(files_to_add) 
        elif not new_unique_selection and len(selection) > 0:
             self.show_alert("File Duplikat", "Semua file yang Anda pilih sudah ada dalam daftar.")
        else:
             pass
             
    def on_file_drop(self, window, file_path_bytes):
        """Handler event saat file di-drag ke jendela aplikasi."""
        if self.is_processing:
            return
        
        if len(self.file_list) + len(self._dropped_files) >= self.max_files:
            if not self._drop_event:
                 self.show_alert("Batas File Tercapai", f"Batas maksimal {self.max_files} file akan tercapai. Beberapa file yang di-drop mungkin ditolak.")

        try:
            file_path_str = file_path_bytes.decode('utf-8')
        except Exception as e:
            print(f"Gagal decode path file: {e}")
            return

        allowed_extensions = ('.pdf', '.jpeg', '.jpg', '.png')
        if not file_path_str.lower().endswith(allowed_extensions):
            print(f"File ditolak (format tidak didukung): {file_path_str}")
            return

        # Cek duplikat DENGAN daftar yang sudah ada
        existing_paths = [f['path'] for f in self.file_list]
        if file_path_str not in self._dropped_files and file_path_str not in existing_paths:
            self._dropped_files.append(file_path_str)

        if self._drop_event:
            self._drop_event.cancel()
        # Tunda pemrosesan sedikit untuk menangani multiple drop
        self._drop_event = Clock.schedule_once(self.process_dropped_files, 0.2)

    def process_dropped_files(self, dt):
        """Meneruskan file yang di-drop ke logika 'handle_selection'."""
        if not self._dropped_files:
            return
        files_to_process = list(self._dropped_files)
        self._dropped_files.clear()
        self._drop_event = None
        if files_to_process:
            self.handle_selection(files_to_process)
            
    @mainthread
    def _append_files_to_ui(self, new_paths):
        """Menambahkan file baru (path) ke 'self.file_list' dan memperbarui UI grid."""
        if not new_paths: 
            return
        
        # Tampilkan container file list jika masih tersembunyi
        if self.ids.file_list_container.opacity == 0:
            self.ids.file_list_container.opacity = 1
            self.ids.file_list_container.size_hint_y = 1
            self.ids.file_list_container.height = 0
            self.ids.main_screen_spacer.size_hint_y = None
            self.ids.main_screen_spacer.height = 0
        
        start_index = len(self.file_list)

        for i, path in enumerate(new_paths):
            current_index = start_index + i
            
            try:
                file_name = os.path.basename(path)
                file_size = self.get_file_size_str(path)

                # Buat label dengan color eksplisit
                label_name = GridCellLabel(
                    text=file_name, 
                    size_hint_x=0.6,
                    color=(0, 0, 0, 1)  # Hitam eksplisit
                )
                label_size = GridCellLabelRight(
                    text=file_size, 
                    size_hint_x=0.15,
                    color=(0, 0, 0, 1)  # Hitam eksplisit
                )
                
                # Gunakan WaitingCellWithDelete yang memiliki tombol X
                progress_widget = WaitingCellWithDelete(
                    file_index=current_index,
                    size_hint_x=0.25
                )

                self.ids.file_list_grid.add_widget(label_name)
                self.ids.file_list_grid.add_widget(label_size)
                self.ids.file_list_grid.add_widget(progress_widget)

                self.file_list.append({
                    'index': current_index,
                    'path': path,
                    'name': file_name,
                    'original_name': os.path.splitext(file_name)[0],
                    'size': file_size,
                    'status': 'pending',
                    'progress_widget': progress_widget,
                    'progress_container': progress_widget,
                    'output_data': None,
                    'tables': None,
                    'sheet_names': None
                })

            except Exception as e:
                print(f"Gagal menambahkan file {path} ke daftar: {e}")
                error_label = GridCellLabel(
                    text=f"Error: {os.path.basename(path)}", 
                    size_hint_x=0.6, 
                    color=(0.8, 0.2, 0.2, 1)  # Merah
                )
                na_label = GridCellLabelRight(
                    text="N/A", 
                    size_hint_x=0.15,
                    color=(0.5, 0.5, 0.5, 1)
                )
                fail_label = ProgressTextCell(
                    text="Gagal Muat", 
                    size_hint_x=0.25, 
                    color=(0.8, 0.2, 0.2, 1)  # Merah
                )
                self.ids.file_list_grid.add_widget(error_label)
                self.ids.file_list_grid.add_widget(na_label)
                self.ids.file_list_grid.add_widget(fail_label)
            
        # Perbarui label status dengan jumlah file total
        total_files = len(self.file_list)
        if total_files > 0:
            self.ids.status_label.text = f"{total_files} file siap dikonversi."

    def start_conversion(self):
        """Memulai proses konversi. Mengunci UI dan memulai pemrosesan file pertama."""
        if self.is_processing:
            self.ids.status_label.text = "Proses konversi sedang berjalan."
            return
        if not self.file_list:
            self.ids.status_label.text = "Error: Tidak ada file yang dipilih!"
            return

        # Jika semua file sudah diproses, reset statusnya untuk konversi ulang
        if not any(f['status'] == 'pending' for f in self.file_list):
            self.ids.status_label.text = "Menyiapkan konversi ulang..."
            for i, f_info in enumerate(self.file_list):
                if f_info['status'] != 'pending':
                    self.reset_progress_widget(i)

        selected_format = self.selected_format
        if selected_format not in ['XLSX', 'CSV', 'ODS']:
            self.ids.status_label.text = "Error: Silakan pilih format output terlebih dahulu!"
            return

        self._clear_results() # Bersihkan hasil sebelumnya
        self.is_processing = True
        
        # Sembunyikan tombol dan kunci UI
        self.ids.button_toolbar.opacity = 0
        self.ids.button_toolbar.size_hint_y = None
        self.ids.button_toolbar.height = 0
        self.ids.select_button.disabled = True
        
        self.ids.status_label.text = "Memulai konversi..."

        grid = self.ids.file_list_grid
        for i, f_info in enumerate(self.file_list):
            if f_info['status'] == 'pending':
                current_widget = f_info.get('progress_widget')
                
                if current_widget and isinstance(current_widget, WaitingCellWithDelete):
                    try:
                        widget_index = grid.children.index(current_widget)
                        grid.remove_widget(current_widget)
                        
                        # Buat widget "Menunggu" tanpa tombol X
                        new_progress_widget = ProgressTextCell(
                            text="Menunggu",
                            size_hint_x=0.25,
                            color=(0.5, 0.5, 0.5, 1)  # Abu-abu
                        )
                        
                        grid.add_widget(new_progress_widget, index=widget_index)
                        f_info['progress_widget'] = new_progress_widget
                        f_info['progress_container'] = new_progress_widget
                        
                    except Exception as e:
                        print(f"Error mengganti widget untuk index {i}: {e}")

        first_pending_index = -1
        for i, f_info in enumerate(self.file_list):
            if f_info['status'] == 'pending':
                if first_pending_index == -1:
                    first_pending_index = i

        if first_pending_index != -1:
            self.current_processing_index = first_pending_index
            self._process_next_file()
        else:
            # Tidak ada file yang valid untuk diproses
            self.is_processing = False
            self.ids.button_toolbar.opacity = 1
            self.ids.button_toolbar.size_hint_y = None
            self.ids.button_toolbar.height = dp(48)
            self.ids.select_button.disabled = False
            self.ids.status_label.text = "Tidak ada file yang perlu diproses."

    @mainthread
    def remove_file_from_list(self, file_index):
        """Menghapus file dari daftar berdasarkan index."""
        if self.is_processing:
            self.show_alert("Sedang Memproses", "Tidak dapat menghapus file saat proses konversi sedang berjalan.")
            return
        
        if not (0 <= file_index < len(self.file_list)):
            print(f"Error: Index {file_index} tidak valid untuk dihapus.")
            return
        
        file_info = self.file_list[file_index]
        
        # Hanya boleh dihapus jika statusnya 'pending'
        if file_info['status'] != 'pending':
            self.show_alert("Tidak Dapat Dihapus", "File ini sudah diproses atau sedang dalam proses.")
            return
        
        grid = self.ids.file_list_grid
        
        # Hapus 3 widget (nama, ukuran, progres) dari grid
        widgets_to_remove = []
        try:
            # Index widget di grid (dari belakang): file ke-N ada di posisi (total_files - N - 1) * 3
            widget_start_index = (len(self.file_list) - file_index - 1) * 3
            for i in range(3):
                idx = widget_start_index + i
                if idx < len(grid.children):
                    widgets_to_remove.append(grid.children[idx])
        except Exception as e:
            print(f"Error saat mencari widget untuk dihapus: {e}")
            return
        
        # Hapus widget dari grid
        for widget in widgets_to_remove:
            grid.remove_widget(widget)
        
        # Hapus dari file_list
        removed_file_name = file_info['name']
        self.file_list.pop(file_index)
        
        # Re-index semua file yang tersisa dan update widget mereka
        for i, f in enumerate(self.file_list):
            f['index'] = i
            # Update file_index pada widget WaitingCellWithDelete jika ada
            if f['status'] == 'pending' and isinstance(f['progress_widget'], WaitingCellWithDelete):
                f['progress_widget'].file_index = i
        
        # Update status label
        total_files = len(self.file_list)
        if total_files > 0:
            self.ids.status_label.text = f"{total_files} file siap dikonversi."
        else:
            self.ids.status_label.text = f"Silakan pilih file (maks {self.max_files}) dan klik Konversi."
            # Sembunyikan container jika tidak ada file
            self.ids.file_list_container.opacity = 0
            self.ids.file_list_container.size_hint_y = None
            self.ids.file_list_container.height = 0
            self.ids.main_screen_spacer.size_hint_y = 1
            self.ids.main_screen_spacer.height = dp(1)
        
        print(f"File '{removed_file_name}' berhasil dihapus dari daftar.")
    @mainthread
    def reset_progress_widget(self, file_index):
        """Mereset status visual widget di grid ke 'Menunggu' dengan tombol X."""
        if 0 <= file_index < len(self.file_list):
            file_info = self.file_list[file_index]
            grid = self.ids.file_list_grid

            current_widget = file_info.get('progress_widget')
            
            if not current_widget or current_widget not in grid.children: 
                print(f"Fallback: Mencari widget progress untuk index {file_index}")
                try:
                    widget_list_index = (len(grid.children) - 1) - (file_index * 3)
                    current_widget = grid.children[widget_list_index]
                except IndexError as e:
                    print(f"Error: Fallback gagal. Tidak dapat reset widget untuk index {file_index}: {e}")
                    file_info['status'] = 'pending'
                    return

            widget_index = grid.children.index(current_widget)
            grid.remove_widget(current_widget)

            # Buat widget WaitingCellWithDelete yang baru (dengan tombol X)
            new_widget = WaitingCellWithDelete(
                file_index=file_index,
                size_hint_x=0.25
            )
            
            grid.add_widget(new_widget, index=widget_index)

            file_info['progress_widget'] = new_widget
            file_info['progress_container'] = new_widget
            file_info['status'] = 'pending'

    def _process_next_file(self):
        """Fungsi rekursif/loop: menemukan file 'pending' berikutnya dan memulainya di thread baru. Jika selesai, pindah ke layar hasil."""
        next_index_to_process = -1
        # Cari file berikutnya yang masih 'pending'
        for i in range(self.current_processing_index, len(self.file_list)):
            if self.file_list[i]['status'] == 'pending':
                next_index_to_process = i
                break

        if next_index_to_process != -1:
            # File ditemukan, proses file ini
            self.current_processing_index = next_index_to_process
            file_info = self.file_list[self.current_processing_index]
            file_info['status'] = 'processing'
            self.ids.status_label.text = f"Memproses file {self.current_processing_index + 1}/{len(self.file_list)}: {file_info['name']}..."

            progress_widget = file_info['progress_widget']
            if progress_widget and isinstance(progress_widget, (MDLabel, ProgressTextCell)):
                progress_widget.text = "Memproses..."
                progress_widget.color = (0, 0, 0, 1)  # Hitam eksplisit

            # Mulai thread baru untuk pemrosesan
            thread = threading.Thread(
                target=self._run_single_conversion_thread,
                args=(file_info['path'], file_info['index'])
            )
            thread.daemon = True
            thread.start()
        
        else: 
            # Tidak ada file 'pending' lagi, proses selesai
            self.is_processing = False
            self.current_processing_index = -1
            
            any_success = any(f['status'] == 'success' for f in self.file_list)
            result_screen = self.manager.result_screen
            
            if result_screen:
                # Bangun UI di layar hasil
                result_grid = result_screen.ids.result_file_list_grid
                result_grid.clear_widgets() 
                
                for f_info in self.file_list:
                    # Tambahkan color eksplisit untuk setiap label
                    result_grid.add_widget(ResultCellLabel(
                        text=f_info['name'], 
                        size_hint_x=0.6,
                        color=(0, 0, 0, 1)  # Hitam eksplisit
                    ))
                    result_grid.add_widget(ResultCellLabelRight(
                        text=f_info['size'], 
                        size_hint_x=0.15,
                        color=(0, 0, 0, 1)  # Hitam eksplisit
                    ))
                    
                    if f_info['status'] == 'success':
                        result_grid.add_widget(ResultSuccessCell(
                            file_index=f_info['index'],
                            size_hint_x=0.25
                        ))
                    elif f_info['status'] == 'failed':
                        result_grid.add_widget(ResultFailedCell(
                            size_hint_x=0.25
                        ))
                    else: # 'no_tables'
                        result_grid.add_widget(ResultNoTablesCell(
                            size_hint_x=0.25
                        ))

                result_screen.ids.save_button_result.disabled = not any_success
                
                if len([f for f in self.file_list if f['status'] == 'success']) > 1:
                    result_screen.ids.save_button_result.text = 'Unduh Semua (ZIP)'
                else:
                    result_screen.ids.save_button_result.text = 'Unduh Semua'
                
                if any_success:
                    result_screen.ids.result_status_label.text = "Semua file selesai diproses. Silakan unduh."
                else:
                    result_screen.ids.result_status_label.text = "Proses selesai, namun tidak ada tabel yang berhasil diekspor."

            self.ids.status_label.text = "Semua file selesai diproses."
            self.manager.go_to_result_screen() # Pindah ke layar hasil

    @mainthread
    def _update_progress_label(self, file_index, current_page, total_pages, error=False, message=""):
        """Callback untuk thread (khususnya PDF) untuk memperbarui label progres di main thread."""
        if not (0 <= file_index < len(self.file_list)):
            print(f"Peringatan: Index {file_index} di luar jangkauan untuk update progress.")
            return
        
        file_info = self.file_list[file_index]
        if file_info['status'] != 'processing': return

        progress_widget = file_info['progress_widget']
        if progress_widget and isinstance(progress_widget, MDLabel):
            if error:
                if message:
                        progress_widget.text = message
                        progress_widget.color = (0.8, 0.2, 0.2, 1)  # Merah eksplisit
            elif file_info['path'].lower().endswith('.pdf'):
                progress_widget.text = f"Halaman {current_page}/{total_pages}"
                progress_widget.color = (0, 0, 0, 1)  # Hitam eksplisit
            else:
                progress_widget.text = "Memproses..."
                progress_widget.color = (0, 0, 0, 1)  # Hitam eksplisit


    def _run_single_conversion_thread(self, file_path, file_index):
        """Fungsi yang berjalan di thread terpisah. Memanggil process_pdf atau process_image."""
        tables = None
        sheet_names = None
        error_occurred = False
        error_object = None

        try:
            file_extension = file_path.split('.')[-1].lower()
            if file_extension == 'pdf':
                # Sediakan callback untuk update progres
                progress_callback = lambda current, total, error=False, message="": self._update_progress_label(file_index, current, total, error, message)
                tables, sheet_names = process_pdf(file_path, progress_callback)
            else:
                 # File gambar tidak memiliki progres multi-halaman
                 self._update_progress_label(file_index, 0, 1)
                 tables = process_image(file_path)
                 sheet_names = [f"Hal_1_Tabel_{i+1}" for i in range(len(tables or []))]

        except Exception as e:
            error_occurred = True
            error_object = e
            print(f"Gagal memproses file index {file_index}: {e}")
            self._update_progress_label(file_index, 0, 1, error=True, message="Gagal Konversi")

        # Kirim hasil kembali ke main thread
        Clock.schedule_once(lambda dt: self.on_single_conversion_result(file_index, tables, sheet_names, error_object))

    @mainthread
    def on_single_conversion_result(self, file_index, tables, sheet_names, error):
        """Callback setelah thread selesai. Memperbarui UI grid dengan status (Sukses, Gagal, Tak Ada Tabel)."""
        if not (0 <= file_index < len(self.file_list)): 
            print(f"Error: Index hasil {file_index} di luar jangkauan.")
            return

        file_info = self.file_list[file_index]
        
        grid = self.ids.file_list_grid 

        current_widget = file_info.get('progress_widget')
        # Fallback jika referensi widget hilang
        if not current_widget or current_widget not in grid.children:
             print(f"Error: Tidak dapat menemukan widget progress untuk index {file_index}.")
             try:
                 widget_list_index = (len(grid.children) - 1) - (file_index * 3)
                 current_widget = grid.children[widget_list_index]
                 if not (isinstance(current_widget, MDLabel) or isinstance(current_widget, MDBoxLayout)): 
                     current_widget = None
             except IndexError: 
                 current_widget = None

             if not current_widget:
                 print(f"Error: Fallback gagal. Tidak dapat update UI hasil untuk index {file_index}.")
                 self.current_processing_index = file_index + 1
                 self._process_next_file() # Lanjutkan ke file berikutnya
                 return

        widget_index = grid.children.index(current_widget)
        grid.remove_widget(current_widget)
        new_widget = None

        if error:
             print(f"Melaporkan kegagalan untuk index {file_index}: {error}")
             file_info['status'] = 'failed'
             new_widget = FailedStatusCell(size_hint_x=0.25)
        elif tables:
            # Sukses, ada tabel ditemukan
            file_info['tables'] = tables
            file_info['sheet_names'] = sheet_names
            selected_format = self.selected_format.lower()
            try:
                # Buat file output di memori
                output_bytes, _, file_ext = create_tables_export(tables, selected_format, sheet_names)
                if output_bytes:
                     file_info['output_data'] = output_bytes
                     file_info['file_ext'] = file_ext
                     file_info['status'] = 'success'
                else:
                     raise ValueError(f"Ekspor ke {selected_format} gagal.")
            except Exception as e:
                 print(f"Gagal membuat ekspor untuk file {file_index}: {e}")
                 file_info['status'] = 'failed'
                 file_info['output_data'] = None

            if file_info['status'] == 'success':
                 # Tampilkan tombol Unduh
                 new_widget = DownloadButtonCell(file_index=file_info['index'], size_hint_x=0.25)
            else:
                 new_widget = FailedStatusCell(size_hint_x=0.25)
        else:
            # Sukses, tapi tidak ada tabel ditemukan
            file_info['status'] = 'no_tables'
            new_widget = NoTablesStatusCell(size_hint_x=0.25)

        if new_widget:
            # Tambahkan widget status baru ke grid
            grid.add_widget(new_widget, index=widget_index)
            file_info['progress_widget'] = new_widget

        # Lanjutkan ke file berikutnya
        self.current_processing_index = file_index + 1
        self._process_next_file()

    def download_single_file(self, file_index):
        """Menangani logika untuk mengunduh satu file hasil."""
        if not (0 <= file_index < len(self.file_list)):
            self.ids.status_label.text = "Error: Index file tidak valid."
            return

        file_info = self.file_list[file_index]

        if file_info['status'] != 'success' or not file_info['output_data']:
            self.ids.status_label.text = f"Error: File '{file_info['name']}' belum selesai atau gagal."
            return

        selected_format = self.selected_format.lower()
        output_data = file_info['output_data']
        file_ext = file_info.get('file_ext', selected_format)

        # Jika CSV dan ada banyak tabel, zip otomatis
        if selected_format == 'csv' and file_info['tables'] and len(file_info['tables']) > 1:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                 for i, df in enumerate(file_info['tables']):
                      sheet_name = file_info['sheet_names'][i].replace(" ", "_") if file_info['sheet_names'] and i < len(file_info['sheet_names']) else f"Tabel_{i+1}"
                      zf.writestr(f"{sheet_name}.csv", df.to_csv(index=False, header=False, encoding='utf-8-sig'))
            output_data = zip_buffer.getvalue()
            file_ext = 'zip'

        timestamp = datetime.now().strftime("%d %b %Y, %H-%M")
        default_filename = f"{file_info['original_name']}_{timestamp}.{file_ext}"

        # Buka dialog simpan file
        filechooser.save_file(
            on_selection=lambda path: self.write_saved_file(path, output_data, expected_ext=file_ext, is_single_download=True),
            path=default_filename,
            title="Simpan File Hasil Konversi"
        )

    def save_result(self):
        """Menangani logika 'Unduh Semua', menggabungkan file sukses ke ZIP jika perlu."""
        successful_files = [f for f in self.file_list if f['status'] == 'success' and f.get('output_data')]
        result_screen = self.manager.result_screen

        if not successful_files:
            if result_screen:
                result_screen.ids.result_status_label.text = "Tidak ada file yang berhasil dikonversi untuk disimpan."
            return

        selected_format = self.selected_format.lower()
        final_output_data = None
        final_file_ext = selected_format
        base_filename = "hasil_konversi"

        if len(successful_files) > 1:
            # Jika lebih dari 1 file sukses, buat ZIP
            final_file_ext = 'zip'
            base_filename = "hasil_konversi_batch"
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_info in successful_files:
                    try:
                        # Regenerasi data untuk format yang benar (jika format diubah setelah konversi)
                        output_data_single, _, file_ext_single = create_tables_export(
                            file_info['tables'], selected_format, file_info['sheet_names']
                        )
                        timestamp_individual = datetime.now().strftime("%H-%M-%S")
                        if selected_format == 'csv' and file_info['tables'] and len(file_info['tables']) > 1:
                            # Jika CSV multi-tabel, pecah di dalam zip
                            for i, df in enumerate(file_info['tables']):
                                sheet_name = file_info['sheet_names'][i].replace(" ", "_") if file_info['sheet_names'] and i < len(file_info['sheet_names']) else f"Tabel_{i+1}"
                                filename_in_zip = f"{file_info['original_name']}_{timestamp_individual}_{sheet_name}.csv"
                                zf.writestr(filename_in_zip, df.to_csv(index=False, header=False, encoding='utf-8-sig'))
                        elif output_data_single:
                            filename_in_zip = f"{file_info['original_name']}_{timestamp_individual}.{file_ext_single}"
                            zf.writestr(filename_in_zip, output_data_single)
                    except Exception as e:
                         print(f"Gagal menambahkan file {file_info['name']} ke zip: {e}")
                         continue
            final_output_data = zip_buffer.getvalue()

        else: 
            # Hanya satu file sukses
            file_info = successful_files[0]
            try:
                final_output_data, _, final_file_ext = create_tables_export(
                       file_info['tables'], selected_format, file_info['sheet_names']
                )
            except Exception as e:
                 print(f"Gagal regenerasi ekspor untuk simpan {file_info['name']}: {e}")
                 if result_screen:
                     result_screen.ids.result_status_label.text = "Error saat menyiapkan file untuk disimpan."
                 return

            if selected_format == 'csv' and file_info['tables'] and len(file_info['tables']) > 1:
                 # Zip otomatis jika CSV multi-tabel
                 zip_buffer = io.BytesIO()
                 try:
                     with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                         timestamp_individual = datetime.now().strftime("%H-%M-%S")
                         for i, df in enumerate(file_info['tables']):
                             sheet_name = file_info['sheet_names'][i].replace(" ", "_") if file_info['sheet_names'] and i < len(file_info['sheet_names']) else f"Tabel_{i+1}"
                             filename_in_zip = f"{file_info['original_name']}_{timestamp_individual}_{sheet_name}.csv"
                             zf.writestr(filename_in_zip, df.to_csv(index=False, header=False, encoding='utf-8-sig'))
                     final_output_data = zip_buffer.getvalue()
                     final_file_ext = 'zip'
                 except Exception as e:
                      print(f"Gagal membuat zip untuk CSV multi-tabel {file_info['name']}: {e}")
                      if result_screen:
                          result_screen.ids.result_status_label.text = "Error saat membuat file ZIP."
                      return

            base_filename = file_info['original_name']

        if not final_output_data:
            if result_screen:
                result_screen.ids.result_status_label.text = "Error: Tidak ada data hasil konversi yang valid untuk disimpan."
            return

        timestamp_outer = datetime.now().strftime("%d %b %Y, %H-%M")
        default_filename = f"{base_filename}_{timestamp_outer}.{final_file_ext}"

        filechooser.save_file(
            on_selection=lambda path: self.write_saved_file(path, final_output_data, expected_ext=final_file_ext, is_single_download=False),
            path=default_filename,
            title="Simpan Semua Hasil"
        )

    def write_saved_file(self, path, data, expected_ext, is_single_download=False):
        """Menulis data bytes ke file di disk yang dipilih pengguna."""
        if not path:
            return
        
        result_screen = self.manager.result_screen
        
        try:
            save_path = path[0] if isinstance(path, list) else path
            # Pastikan ekstensi file benar
            base, ext = os.path.splitext(save_path)
            clean_expected_ext = expected_ext.lstrip('.')
            if not ext or ext.lower().replace('.', '') != clean_expected_ext:
                 save_path = f"{base}.{clean_expected_ext}"

            with open(save_path, 'wb') as f:
                f.write(data)
            
            status_msg = f"File berhasil disimpan: {os.path.basename(save_path)}"
            
            # Tampilkan pesan sukses di layar yang aktif
            if self.manager.current == 'main':
                self.ids.status_label.text = status_msg
            elif result_screen:
                result_screen.ids.result_status_label.text = status_msg

        except Exception as e:
            status_msg = f"Gagal menyimpan file: {e}"
            if self.manager.current == 'main':
                self.ids.status_label.text = status_msg
            elif result_screen:
                result_screen.ids.result_status_label.text = status_msg
            import traceback
            traceback.print_exc()

# --- Kelas Aplikasi Utama ---

class PDFExtractApp(MDApp):
    def build(self):
        """Metode build utama aplikasi Kivy. Mengatur judul jendela dan tema."""
        self.title = "PDFExtract" # Atur nama judul jendela
        
        try:
            icon_path = os.path.join(application_path, 'pdfextract.ico')
            if os.path.exists(icon_path):
                self.icon = icon_path
            else:
                print(f"Peringatan: File ikon tidak ditemukan di {icon_path}")
        except Exception as e:
            print(f"Gagal mengatur ikon aplikasi: {e}")
            
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.accent_palette = "Green"
        self.theme_cls.theme_style = "Light"
        
        return RootScreenManager()

# --- Entry Point Aplikasi ---

if __name__ == '__main__':
    PDFExtractApp().run()