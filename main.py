import os
import threading
import io
import sys
import gc
import zipfile
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.clock import mainthread, Clock
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.factory import Factory
from kivy.graphics import Color, Rectangle, Line
from kivy.properties import ObjectProperty, StringProperty, NumericProperty
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


# Dapatkan direktori tempat skrip dijalankan
if getattr(sys, 'frozen', False):
    # Jika dijalankan sebagai aplikasi yang di-bundle
    application_path = os.path.dirname(sys.executable)
else:
    # Jika dijalankan sebagai skrip python biasa
    application_path = os.path.dirname(os.path.abspath(__file__))


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

class DownloadButtonCell(BoxLayout):
    file_index = NumericProperty(-1)

def preprocess_image_for_ocr(img_array):
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
    return text_based_table_detection(img_array, gray)

def text_based_table_detection(img_array, gray):
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
    return reconstruct_table(table_texts, w_table, h_table)

def reconstruct_table(texts, table_width, table_height):
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
            output = io.BytesIO() # Kembalikan bytes kosong jika gagal

    output.seek(0)
    return output.getvalue(), mime_type, file_ext

def process_image_for_tables(img_array):
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
    try:
        image = Image.open(image_path)
        img_array = np.array(image.convert('RGB'))
        tables = process_image_for_tables(img_array)
        # Hapus objek gambar dan panggil GC
        del image
        del img_array
        gc.collect()
        return tables
    except Exception as e:
        print(f"Gagal memproses gambar {image_path}: {e}")
        return []

# Definisi Kivy UI
Window.clearcolor = (1, 1, 1, 1)

KV = """
#:set text_color (0.1, 0.1, 0.1, 1)
#:set placeholder_text_color (0.4, 0.4, 0.4, 1)
#:set bg_color (1, 1, 1, 1)
#:set green_color (0.22, 0.6, 0.22, 1)
#:set blue_color (0.1, 0.6, 0.9, 1)
#:set light_gray_color (0.9, 0.9, 0.9, 1)
#:set mid_gray_color (0.8, 0.8, 0.8, 1)
#:set disabled_gray_color (0.7, 0.7, 0.7, 1)
#:set corner_radius 8
#:set green_success_color (0.22, 0.6, 0.22, 1)
#:set red_fail_color (0.8, 0.2, 0.2, 1)

<SpinnerOption>:
    background_normal: ''
    background_color: bg_color
    color: text_color
    height: dp(40)
    padding: dp(5)
    canvas.before:
        Color:
            rgba: light_gray_color if self.state == 'down' else bg_color
        Rectangle:
            pos: self.pos
            size: self.size

<GridCellLabel@Label>:
    color: text_color
    halign: 'left'
    valign: 'middle'
    padding_x: dp(5)
    shorten: True
    text_size: (self.width - dp(10), None) if self.width > dp(20) else (dp(10), None)
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

<HeaderCellLabel@GridCellLabel>:
    bold: True
    canvas.before:
        Color:
            rgba: light_gray_color
        Rectangle:
            pos: self.pos
            size: self.size

<HeaderCellLabelRight@GridCellLabelRight>:
    bold: True
    canvas.before:
        Color:
            rgba: light_gray_color
        Rectangle:
            pos: self.pos
            size: self.size

<DownloadButtonCell>:
    padding: dp(5)
    canvas.before:
        Color:
            rgba: 1, 1, 1, 1
        Rectangle:
            pos: self.pos
            size: self.size
    Button:
        text: 'Unduh'
        background_normal: ''
        background_disabled_normal: ''
        background_color: green_success_color
        color: (1,1,1,1)
        size_hint_y: 0.8
        pos_hint: {'center_y': 0.5}
        on_release: app.root.download_single_file(root.file_index)
        canvas.before:
            Color:
                rgba: self.background_color
            RoundedRectangle:
                pos: self.pos
                size: self.size
                radius: [app.root.corner_radius * 0.5]

<FailedStatusCell@BoxLayout>:
    padding: dp(5)
    canvas.before:
        Color:
            rgba: 1, 1, 1, 1
        Rectangle:
            pos: self.pos
            size: self.size
    Label:
        text: 'Gagal'
        color: red_fail_color
        bold: True
        halign: 'center'
        valign: 'middle'

<NoTablesStatusCell@BoxLayout>:
    padding: dp(5)
    canvas.before:
        Color:
            rgba: 1, 1, 1, 1
        Rectangle:
            pos: self.pos
            size: self.size
    Label:
        text: 'Tak Ada Tabel'
        color: placeholder_text_color
        halign: 'center'
        valign: 'middle'

<ProgressTextCell@GridCellLabelRight>:


<MainLayout>:
    orientation: 'vertical'
    padding: 20
    spacing: 15
    canvas.before:
        Color:
            rgba: bg_color
        Rectangle:
            pos: self.pos
            size: self.size

    BoxLayout:
        size_hint_y: None
        height: '48dp'
        Label:
            text: 'Konverter PDF ke Spreadsheet'
            font_size: '24sp'
            bold: True
            color: text_color

    Label:
        id: status_label
        text: 'Silakan pilih file (maks 3) dan klik Konversi.'
        size_hint_y: None
        height: '40dp'
        color: text_color
        font_size: '18sp'
        bold: True
        valign: 'middle'

    BoxLayout:
        size_hint_y: None
        height: '48dp'
        spacing: 10
        
        Button:
            id: select_button
            text: 'Pilih File'
            on_release: root.select_file()
            background_normal: ''
            background_disabled_normal: ''
            background_color: green_color
            color: (1, 1, 1, 1)
            size_hint_x: 0.4
            canvas.before:
                Color:
                    rgba: self.background_color
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [root.corner_radius]
            
        Spinner:
            id: format_spinner
            text: 'XLSX'
            values: ['XLSX', 'CSV', 'ODS']
            background_normal: ''
            background_disabled_normal: ''
            background_color: light_gray_color
            color: text_color
            option_cls: 'SpinnerOption'
            size_hint_x: 0.3
            canvas.before:
                Color:
                    rgba: self.background_color
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [root.corner_radius]
        
        Button:
            id: convert_button
            text: 'Konversi'
            on_release: root.start_conversion()
            background_normal: ''
            background_disabled_normal: ''
            background_color: blue_color
            color: (1, 1, 1, 1)
            size_hint_x: 0.3
            canvas.before:
                Color:
                    rgba: self.background_color
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [root.corner_radius]

    BoxLayout:
        orientation: 'vertical'
        size_hint_y: 1
        spacing: dp(1) 
        
        canvas.after:
            Color:
                rgba: mid_gray_color
            Line:
                width: dp(1)
                rounded_rectangle: (self.x, self.y, self.width, self.height, corner_radius)

        GridLayout:
            cols: 3
            size_hint_y: None
            height: '30dp'
            spacing: dp(1)
            canvas.before:
                Color:
                    rgba: mid_gray_color
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [corner_radius, corner_radius, 0, 0]
            
            HeaderCellLabel:
                text: "Nama File"
                size_hint_x: 0.6
            HeaderCellLabelRight:
                text: "Ukuran"
                size_hint_x: 0.15
            HeaderCellLabelRight:
                text: "Progres"
                size_hint_x: 0.25

        ScrollView:
            bar_width: dp(10)
            GridLayout:
                id: file_list_grid
                cols: 3
                size_hint_y: None
                height: self.minimum_height
                row_default_height: '40dp'
                row_force_default: True
                spacing: dp(1)
                canvas.before:
                    Color:
                        rgba: mid_gray_color
                    Rectangle:
                        pos: self.pos
                        size: self.size

    Button:
        id: save_button
        text: 'Unduh Semua'
        size_hint_y: None
        height: '48dp'
        disabled: True
        on_release: root.save_result()
        background_normal: ''
        background_disabled_normal: ''
        background_color: disabled_gray_color if self.disabled else blue_color
        color: (0.3, 0.3, 0.3, 1) if self.disabled else (1, 1, 1, 1)
        canvas.before:
            Color:
                rgba: self.background_color
            RoundedRectangle:
                pos: self.pos
                size: self.size
                radius: [root.corner_radius]
"""
Builder.load_string(KV)

# Logika Utama Aplikasi Kivy 
class MainLayout(BoxLayout):
    corner_radius = NumericProperty(dp(8))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.extracted_tables = {}
        self.sheet_names = {}
        self.output_data = {}
        self.file_list = []
        self.current_processing_index = -1
        self.is_processing = False
        self.text_color = (0.1, 0.1, 0.1, 1)
        self.placeholder_color = (0.4, 0.4, 0.4, 1)
        self.green_success_color = (0.22, 0.6, 0.22, 1)
        self.red_fail_color = (0.8, 0.2, 0.2, 1)

    def _clear_results(self):
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
        try:
            size_bytes = os.path.getsize(file_path)
            if size_bytes < 1024: return f"{size_bytes} B"
            elif size_bytes < 1024**2: return f"{size_bytes/1024:.1f} KB"
            elif size_bytes < 1024**3: return f"{size_bytes/1024**2:.1f} MB"
            else: return f"{size_bytes/1024**3:.1f} GB"
        except Exception: return "N/A"

    @mainthread
    def reset_ui_state(self):
        self.ids.status_label.text = "Silakan pilih file (maks 3) dan klik Konversi."
        self.ids.file_list_grid.clear_widgets()
        self.file_list = []
        self.ids.save_button.disabled = True
        self.ids.save_button.text = 'Unduh Semua'
        self.ids.select_button.disabled = False
        self.ids.convert_button.disabled = False
        self.is_processing = False
        self.current_processing_index = -1

    def select_file(self):
        if self.is_processing: return
        self._clear_results()
        self.reset_ui_state()
        filechooser.open_file(
            on_selection=self.handle_selection,
            filters=[("PDF dan Gambar", "*.pdf", "*.jpeg", "*.png")],
            multiple=True
        )

    def handle_selection(self, selection):
        if selection:
            selected_paths = selection[:3]
            status_text = f"{len(selected_paths)} file siap dikonversi."
            if len(selection) > 3:
                status_text = "Maksimal 3 file dipilih. " + status_text
                Clock.schedule_once(lambda dt: setattr(self.ids.status_label, 'text', f"{len(selected_paths)} file siap dikonversi."), 3)
            self.ids.status_label.text = status_text
            self._update_ui_after_selection(selected_paths)
        else:
             self.ids.status_label.text = "Tidak ada file yang dipilih."

    @mainthread
    def _update_ui_after_selection(self, selected_paths):
        self.ids.file_list_grid.clear_widgets()
        self.file_list = []

        if not selected_paths: return

        for index, path in enumerate(selected_paths):
            try:
                file_name = os.path.basename(path)
                file_size = self.get_file_size_str(path)

                label_name = Factory.GridCellLabel(text=file_name, size_hint_x=0.6)
                label_size = Factory.GridCellLabelRight(text=file_size, size_hint_x=0.15)
                progress_widget = Factory.ProgressTextCell(
                    text="Menunggu",
                    color=self.placeholder_color,
                    size_hint_x=0.25
                )

                self.ids.file_list_grid.add_widget(label_name)
                self.ids.file_list_grid.add_widget(label_size)
                self.ids.file_list_grid.add_widget(progress_widget)

                self.file_list.append({
                    'index': index,
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
                error_label = Factory.GridCellLabel(text=f"Error: {os.path.basename(path)}", size_hint_x=0.6, color=self.red_fail_color)
                na_label = Factory.GridCellLabelRight(text="N/A", size_hint_x=0.15)
                fail_label = Factory.ProgressTextCell(text="Gagal Muat", size_hint_x=0.25, color=self.red_fail_color)
                self.ids.file_list_grid.add_widget(error_label)
                self.ids.file_list_grid.add_widget(na_label)
                self.ids.file_list_grid.add_widget(fail_label)

        if self.file_list:
            self.ids.save_button.disabled = True

    def start_conversion(self):
        if self.is_processing:
            self.ids.status_label.text = "Proses konversi sedang berjalan."
            return
        if not self.file_list:
            self.ids.status_label.text = "Error: Tidak ada file yang dipilih!"
            return

        if not any(f['status'] == 'pending' for f in self.file_list):
            self.ids.status_label.text = "Semua file sudah diproses atau gagal dimuat."
            return

        selected_format = self.ids.format_spinner.text
        if selected_format not in ['XLSX', 'CSV', 'ODS']:
            self.ids.status_label.text = "Error: Format output tidak valid!"
            return

        self._clear_results()
        self.is_processing = True
        self.ids.select_button.disabled = True
        self.ids.convert_button.disabled = True
        self.ids.save_button.disabled = True
        self.ids.status_label.text = "Memulai konversi..."

        first_pending_index = -1
        for i, f_info in enumerate(self.file_list):
             if f_info['status'] != 'pending':
                  self.reset_progress_widget(i)
             if f_info['status'] == 'pending':
                if first_pending_index == -1:
                    first_pending_index = i

        if first_pending_index != -1:
            self.current_processing_index = first_pending_index
            self._process_next_file()
        else:
             self.is_processing = False
             self.ids.select_button.disabled = False
             self.ids.convert_button.disabled = False

    @mainthread
    def reset_progress_widget(self, file_index):
        if 0 <= file_index < len(self.file_list):
            file_info = self.file_list[file_index]
            grid = self.ids.file_list_grid

            current_widget = file_info.get('progress_widget')
            if not current_widget or current_widget not in grid.children: return

            widget_index = grid.children.index(current_widget)
            grid.remove_widget(current_widget)

            new_widget = Factory.ProgressTextCell(
                text="Menunggu",
                color=self.placeholder_color,
                size_hint_x=0.25
            )
            grid.add_widget(new_widget, index=widget_index)

            file_info['progress_widget'] = new_widget
            file_info['progress_container'] = new_widget
            file_info['status'] = 'pending'


    def _process_next_file(self):
        next_index_to_process = -1
        for i in range(self.current_processing_index, len(self.file_list)):
             if self.file_list[i]['status'] == 'pending':
                  next_index_to_process = i
                  break

        if next_index_to_process != -1:
            self.current_processing_index = next_index_to_process
            file_info = self.file_list[self.current_processing_index]
            file_info['status'] = 'processing'
            self.ids.status_label.text = f"Memproses file {self.current_processing_index + 1}/{len(self.file_list)}: {file_info['name']}..."

            progress_widget = file_info['progress_widget']
            if progress_widget and isinstance(progress_widget, Label):
                 progress_widget.text = "Memproses..."
                 progress_widget.color = self.text_color

            thread = threading.Thread(
                target=self._run_single_conversion_thread,
                args=(file_info['path'], file_info['index'])
            )
            thread.daemon = True
            thread.start()
        else: # tidak ada file ditunggu
            self.is_processing = False
            self.ids.select_button.disabled = False
            self.ids.convert_button.disabled = False
            any_success = any(f['status'] == 'success' for f in self.file_list)
            self.ids.save_button.disabled = not any_success
            if len([f for f in self.file_list if f['status'] == 'success']) > 1:
                self.ids.save_button.text = 'Unduh Semua (ZIP)'
            else:
                 self.ids.save_button.text = 'Unduh Semua'

            self.ids.status_label.text = "Semua file selesai diproses."
            self.current_processing_index = -1

    @mainthread
    def _update_progress_label(self, file_index, current_page, total_pages, error=False, message=""):
         if 0 <= file_index < len(self.file_list):
            file_info = self.file_list[file_index]
            if file_info['status'] != 'processing': return

            progress_widget = file_info['progress_widget']
            if progress_widget and isinstance(progress_widget, Label):
                 if error:
                      if message:
                           progress_widget.text = message
                           progress_widget.color = self.red_fail_color
                 elif file_info['path'].lower().endswith('.pdf'):
                      # Tampilkan teks halaman
                      progress_widget.text = f"Halaman {current_page}/{total_pages}"
                      progress_widget.color = self.text_color
                 else:
                      progress_widget.text = "Memproses..."
                      progress_widget.color = self.text_color


    def _run_single_conversion_thread(self, file_path, file_index):
        tables = None
        sheet_names = None
        error_occurred = False
        error_object = None

        try:
            file_extension = file_path.split('.')[-1].lower()
            if file_extension == 'pdf':
                progress_callback = lambda current, total, error=False, message="": self._update_progress_label(file_index, current, total, error, message)
                tables, sheet_names = process_pdf(file_path, progress_callback)
            else:
                 self._update_progress_label(file_index, 0, 1)
                 tables = process_image(file_path)
                 sheet_names = [f"Hal_1_Tabel_{i+1}" for i in range(len(tables or []))]

        except Exception as e:
            error_occurred = True
            error_object = e
            print(f"Gagal memproses file index {file_index}: {e}")
            self._update_progress_label(file_index, 0, 1, error=True, message="Gagal Konversi")

        Clock.schedule_once(lambda dt: self.on_single_conversion_result(file_index, tables, sheet_names, error_object))

    @mainthread
    def on_single_conversion_result(self, file_index, tables, sheet_names, error):
        if not (0 <= file_index < len(self.file_list)): return

        file_info = self.file_list[file_index]
        grid = self.ids.file_list_grid

        current_widget = file_info.get('progress_widget')
        if not current_widget or current_widget not in grid.children:
             print(f"Error: Tidak dapat menemukan widget progress untuk index {file_index}.")
             try:
                  widget_list_index = (len(grid.children) - 1) - (file_index * 3 + 2)
                  current_widget = grid.children[widget_list_index]
                  if not isinstance(current_widget, Label): current_widget = None
             except IndexError: current_widget = None

             if not current_widget:
                  print(f"Error: Fallback gagal. Tidak dapat update UI hasil untuk index {file_index}.")
                  self.current_processing_index = file_index + 1
                  self._process_next_file()
                  return

        widget_index = grid.children.index(current_widget)
        grid.remove_widget(current_widget)
        new_widget = None

        if error:
             print(f"Melaporkan kegagalan untuk index {file_index}: {error}")
             file_info['status'] = 'failed'
             new_widget = Factory.FailedStatusCell(size_hint_x=0.25)
        elif tables:
            file_info['tables'] = tables
            file_info['sheet_names'] = sheet_names
            selected_format = self.ids.format_spinner.text.lower()
            try:
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
                 new_widget = Factory.DownloadButtonCell(file_index=file_index, size_hint_x=0.25)
            else:
                 new_widget = Factory.FailedStatusCell(size_hint_x=0.25)
        else:
            file_info['status'] = 'no_tables'
            new_widget = Factory.NoTablesStatusCell(size_hint_x=0.25)

        if new_widget:
            grid.add_widget(new_widget, index=widget_index)
            file_info['progress_widget'] = new_widget

        self.current_processing_index = file_index + 1
        self._process_next_file()

    def download_single_file(self, file_index):
        if not (0 <= file_index < len(self.file_list)):
            self.ids.status_label.text = "Error: Index file tidak valid."
            return

        file_info = self.file_list[file_index]

        if file_info['status'] != 'success' or not file_info['output_data']:
            self.ids.status_label.text = f"Error: File '{file_info['name']}' belum selesai atau gagal."
            return

        selected_format = self.ids.format_spinner.text.lower()
        output_data = file_info['output_data']
        file_ext = file_info.get('file_ext', selected_format)

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

        filechooser.save_file(
            on_selection=lambda path: self.write_saved_file(path, output_data, expected_ext=file_ext, is_single_download=True),
            path=default_filename,
            title="Simpan File Hasil Konversi"
        )

    def save_result(self):
        successful_files = [f for f in self.file_list if f['status'] == 'success' and f.get('output_data')]

        if not successful_files:
            self.ids.status_label.text = "Tidak ada file yang berhasil dikonversi untuk disimpan."
            return

        selected_format = self.ids.format_spinner.text.lower()
        final_output_data = None
        final_file_ext = selected_format
        base_filename = "hasil_konversi"

        if len(successful_files) > 1:
            final_file_ext = 'zip'
            base_filename = "hasil_konversi_batch"
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_info in successful_files:
                    try:
                        # Regenerate data untuk memastikan format benar
                        output_data_single, _, file_ext_single = create_tables_export(
                            file_info['tables'], selected_format, file_info['sheet_names']
                        )
                        # Buat timestamp UNIK untuk setiap file di dalam zip
                        timestamp_individual = datetime.now().strftime("%H-%M-%S")

                        # Handle multiple CSVs from one PDF source inside zip
                        if selected_format == 'csv' and file_info['tables'] and len(file_info['tables']) > 1:
                             for i, df in enumerate(file_info['tables']):
                                 # Nama sheet/tabel
                                 sheet_name = file_info['sheet_names'][i].replace(" ", "_") if file_info['sheet_names'] and i < len(file_info['sheet_names']) else f"Tabel_{i+1}"
                                 # Nama file di dalam zip: namaasli_timestamp_namasheet.csv
                                 filename_in_zip = f"{file_info['original_name']}_{timestamp_individual}_{sheet_name}.csv"
                                 zf.writestr(filename_in_zip, df.to_csv(index=False, header=False, encoding='utf-8-sig'))
                        elif output_data_single:
                             # Nama file di dalam zip: namaasli_timestamp.ext
                             filename_in_zip = f"{file_info['original_name']}_{timestamp_individual}.{file_ext_single}"
                             zf.writestr(filename_in_zip, output_data_single)

                    except Exception as e:
                         print(f"Gagal menambahkan file {file_info['name']} ke zip: {e}")
                         continue

            final_output_data = zip_buffer.getvalue()

        else: # Hanya satu file sukses
            file_info = successful_files[0]
            try:
                final_output_data, _, final_file_ext = create_tables_export(
                     file_info['tables'], selected_format, file_info['sheet_names']
                )
            except Exception as e:
                 print(f"Gagal regenerasi ekspor untuk simpan {file_info['name']}: {e}")
                 self.ids.status_label.text = "Error saat menyiapkan file untuk disimpan."
                 return

            if selected_format == 'csv' and file_info['tables'] and len(file_info['tables']) > 1:
                 zip_buffer = io.BytesIO()
                 try:
                     with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                          timestamp_individual = datetime.now().strftime("%H-%M-%S") # Timestamp unik
                          for i, df in enumerate(file_info['tables']):
                               sheet_name = file_info['sheet_names'][i].replace(" ", "_") if file_info['sheet_names'] and i < len(file_info['sheet_names']) else f"Tabel_{i+1}"
                               filename_in_zip = f"{file_info['original_name']}_{timestamp_individual}_{sheet_name}.csv" # Nama file di zip
                               zf.writestr(filename_in_zip, df.to_csv(index=False, header=False, encoding='utf-8-sig'))
                     final_output_data = zip_buffer.getvalue()
                     final_file_ext = 'zip'
                 except Exception as e:
                      print(f"Gagal membuat zip untuk CSV multi-tabel {file_info['name']}: {e}")
                      self.ids.status_label.text = "Error saat membuat file ZIP."
                      return

            base_filename = file_info['original_name']

        if not final_output_data:
            self.ids.status_label.text = "Error: Tidak ada data hasil konversi yang valid untuk disimpan."
            return

        # Nama file ZIP luar tetap menggunakan timestamp lengkap
        timestamp_outer = datetime.now().strftime("%d %b %Y, %H-%M")
        default_filename = f"{base_filename}_{timestamp_outer}.{final_file_ext}"

        filechooser.save_file(
            on_selection=lambda path: self.write_saved_file(path, final_output_data, expected_ext=final_file_ext),
            path=default_filename,
            title="Simpan Semua Hasil"
        )

    def write_saved_file(self, path, data, expected_ext, is_single_download=False):
        if not path:
            return
        try:
            save_path = path[0] if isinstance(path, list) else path

            base, ext = os.path.splitext(save_path)
            clean_expected_ext = expected_ext.lstrip('.')
            if not ext or ext.lower().replace('.', '') != clean_expected_ext:
                 save_path = f"{base}.{clean_expected_ext}"


            with open(save_path, 'wb') as f:
                f.write(data)

            if not is_single_download:
                self._clear_results()
                self.reset_ui_state()
                self.ids.status_label.text = f"File berhasil disimpan di: {os.path.basename(save_path)}"
            else:
                 self.ids.status_label.text = f"File berhasil disimpan: {os.path.basename(save_path)}"


        except Exception as e:
            self.ids.status_label.text = f"Gagal menyimpan file: {e}"
            import traceback
            traceback.print_exc()

class PDFConverterApp(App):
    root = ObjectProperty(None)

    def build(self):
        self.root = MainLayout()
        return self.root

if __name__ == '__main__':
    PDFConverterApp().run()