import os
import nibabel as nib
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLineEdit, QGridLayout, QMessageBox,
    QRadioButton, QButtonGroup, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage


class NiiViewer(QWidget):
    def __init__(self, img_dir, seg_dir):
        super().__init__()
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.current_index = 0
        self.slice_index = 0
        self.saved_slices = {}
        self.patient_list = self.load_patient_ids()
        self.slice_record_path = os.path.join(os.path.dirname(__file__), "saved_slices.txt")
        self.min_required_slices = 30

        self.init_ui()
        self.load_patient_data()

    def init_ui(self):
        self.setWindowTitle("NII Viewer")

        # 下拉式選單取代病患ID Label
        self.patient_combo = QComboBox()
        self.patient_combo.addItems(self.patient_list)
        self.patient_combo.currentIndexChanged.connect(self.combo_patient_changed)

        self.min_slice_label = QLabel("最少切片資訊")

        top_info_layout = QHBoxLayout()
        top_info_layout.addWidget(QLabel("Patient ID:"))
        top_info_layout.addWidget(self.patient_combo)
        top_info_layout.addStretch()
        top_info_layout.addWidget(self.min_slice_label)

        self.image_labels = []
        self.image_titles = []

        for _ in range(4):
            title = QLabel()
            title.setAlignment(Qt.AlignCenter)
            img = QLabel()
            img.setFixedSize(256, 256)
            img.setStyleSheet("border: 1px solid black; background-color: black;")

            self.image_titles.append(title)
            self.image_labels.append(img)

        # 對齊選項
        self.align_yes = QRadioButton("對齊")
        self.align_no = QRadioButton("未對齊")
        self.align_group = QButtonGroup()
        self.align_group.addButton(self.align_yes, 1)
        self.align_group.addButton(self.align_no, 0)

        align_layout = QHBoxLayout()
        align_layout.addWidget(QLabel("是否對齊:"))
        align_layout.addWidget(self.align_yes)
        align_layout.addWidget(self.align_no)
        align_layout.addStretch()

        # 新增給未對齊選項的輸入欄位
        self.offset_widgets = []
        self.offset_layout = QHBoxLayout()
        self.offset_layout.addWidget(QLabel("Phase Offsets:"))
        for i in range(3):
            label = QLabel(f"Phase {i+1}:")
            line_edit = QLineEdit("0")
            line_edit.setFixedWidth(50)
            self.offset_layout.addWidget(label)
            self.offset_layout.addWidget(line_edit)
            self.offset_widgets.append((label, line_edit))
        
        self.align_no.toggled.connect(self.toggle_offset_widgets)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self.update_slice_from_slider)

        self.slice_input = QLineEdit()
        self.confirm_button = QPushButton("Confirm Slice")
        self.confirm_button.clicked.connect(self.confirm_slice)

        self.slice_hint_label = QLabel()
        self.slice_hint_label.setStyleSheet("color: red; font-weight: bold;")
        self.slice_hint_label.setAlignment(Qt.AlignCenter)

        self.prev_button = QPushButton("Previous Patient")
        self.next_button = QPushButton("Next Patient")
        self.prev_button.clicked.connect(self.prev_patient)
        self.next_button.clicked.connect(self.next_patient)

        control_layout = QVBoxLayout()
        control_layout.addLayout(align_layout)
        control_layout.addLayout(self.offset_layout)
        control_layout.addWidget(self.slice_slider)
        control_layout.addWidget(self.slice_input)
        control_layout.addWidget(self.slice_hint_label)
        control_layout.addWidget(self.confirm_button)

        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        control_layout.addLayout(nav_layout)

        grid = QGridLayout()
        grid.addWidget(self.image_titles[0], 0, 0)
        grid.addWidget(self.image_labels[0], 1, 0)
        grid.addWidget(self.image_titles[1], 0, 1)
        grid.addWidget(self.image_labels[1], 1, 1)
        grid.addWidget(self.image_titles[2], 2, 0)
        grid.addWidget(self.image_labels[2], 3, 0)
        grid.addWidget(self.image_titles[3], 2, 1)
        grid.addWidget(self.image_labels[3], 3, 1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_info_layout)
        main_layout.addLayout(grid)
        main_layout.addLayout(control_layout)

        self.setLayout(main_layout)
        self.toggle_offset_widgets(False)

    def toggle_offset_widgets(self, checked):
        for label, line_edit in self.offset_widgets:
            label.setVisible(checked)
            line_edit.setVisible(checked)

    def load_patient_ids(self):
        files = os.listdir(self.img_dir)
        patient_ids = set()
        for f in files:
            if f.endswith(".nii.gz"):
                parts = f.split("_")
                if len(parts) >= 3:
                    patient_ids.add(parts[1][:-2])
        return sorted(list(patient_ids))

    def combo_patient_changed(self, index):
        self.current_index = index
        self.slice_index = 0
        self.load_patient_data()

    def load_patient_data(self):
        patient_id = self.patient_list[self.current_index]
        self.patient_combo.setCurrentIndex(self.current_index)
        self.phase_images = []

        for phase in ["0001", "0002", "0003"]:
            phase_idx = int(phase)
            path = os.path.join(self.img_dir, f"HCC_{patient_id}P{phase_idx}_{phase}.nii.gz")
            if os.path.exists(path):
                nii = nib.load(path)
                self.phase_images.append(nii.get_fdata())
            else:
                self.phase_images.append(np.zeros((256, 256, 1)))

        seg_path = None
        for i in range(1, 5):  # Check for P1, P2, P3, P4
            potential_path = os.path.join(self.seg_dir, f"HCC_{patient_id}P{i}.nii.gz")
            if os.path.exists(potential_path):
                seg_path = potential_path
                break

        if seg_path:
            seg_img = nib.load(seg_path).get_fdata()
            self.seg_image = seg_img.astype(np.uint8)
            unique_vals = sorted(np.unique(self.seg_image))
            self.image_titles[0].setText(f"Segmentation (label: {', '.join(map(str, unique_vals))})")
        else:
            self.seg_image = np.zeros_like(self.phase_images[0], dtype=np.uint8)
            self.image_titles[0].setText("Segmentation (label: none)")

        max_slices = 0
        for img in self.phase_images:
            if img is not None and img.shape[2] > 1:
                max_slices = max(max_slices, img.shape[2])
        
        if max_slices > 0:
            self.slice_slider.setMaximum(max_slices - 1)

        self.load_saved_slice_if_exists(patient_id)
        self.update_images()

        min_file, min_slices = self.find_min_slice_volume()
        min_patient_id = min_file.split("_")[1] if min_file else "N/A"
        self.min_slice_label.setText(f"最少切片為 Patient ID {min_patient_id} 共 {min_slices} 張")

        self.update_hint_label()

    def update_images(self):
        slice_idx = self.slice_index
        
        seg_slice_idx = min(max(0, slice_idx), self.seg_image.shape[2] - 1)
        seg_slice = self.seg_image[:, :, seg_slice_idx]
        self.image_labels[0].setPixmap(self.seg_to_pixmap(seg_slice))

        offsets = [0, 0, 0]
        is_unaligned = self.align_no.isChecked()
        if is_unaligned:
            try:
                offsets = [int(w[1].text()) for w in self.offset_widgets]
            except ValueError:
                offsets = [0, 0, 0]

        for i in range(3):
            img_data = self.phase_images[i]
            
            base_title = f"Phase 000{i+1} ({img_data.shape[2]} 張)"

            if img_data.shape[2] <= 1:
                self.image_labels[i+1].clear()
                self.image_labels[i+1].setStyleSheet("border: 1px solid black; background-color: black;")
                self.image_titles[i+1].setText(base_title)
                continue

            offset_slice_idx = slice_idx + offsets[i]
            final_slice_idx = min(max(0, offset_slice_idx), img_data.shape[2] - 1)
            
            slice_img = img_data[:, :, final_slice_idx]
            self.image_labels[i+1].setPixmap(self.numpy_to_pixmap(slice_img))

            if is_unaligned:
                self.image_titles[i+1].setText(f"{base_title} - Slice: {final_slice_idx}")
            else:
                self.image_titles[i+1].setText(base_title)

    def numpy_to_pixmap(self, img):
        img = np.clip(img, 0, 255).astype(np.uint8)
        img_rgb = np.stack([img] * 3, axis=-1)
        img_rgb = np.rot90(img_rgb, k=-1)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg).scaled(256, 256, Qt.KeepAspectRatio)

    def seg_to_pixmap(self, seg_slice):
        color_map = {0: 0, 1: 127, 2: 255}
        gray_img = np.vectorize(color_map.get)(seg_slice)
        gray_rgb = np.stack([gray_img] * 3, axis=-1).astype(np.uint8)
        gray_rgb = np.rot90(gray_rgb, k=-1)
        h, w, ch = gray_rgb.shape
        qimg = QImage(gray_rgb.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg).scaled(256, 256, Qt.KeepAspectRatio)

    def update_slice_from_slider(self, value):
        self.slice_index = value
        self.slice_input.setText(str(value))
        self.update_images()

    def update_hint_label(self):
        valid_slice_counts = [img.shape[2] for img in self.phase_images if img.shape[2] > 1]
        if not valid_slice_counts:
            max_start = 0
        else:
            max_start = min(valid_slice_counts) - self.min_required_slices
        
        max_start = max(0, max_start)
        self.slice_hint_label.setText(f"最大允許起始 slice index：{max_start}")

    def confirm_slice(self):
        value = self.slice_input.text()
        if not value.isdigit():
            return

        slice_value = int(value)
        
        valid_slice_counts = [img.shape[2] for img in self.phase_images if img.shape[2] > 1]
        if not valid_slice_counts:
            max_start = 0
        else:
            max_start = min(valid_slice_counts) - self.min_required_slices
        max_start = max(0, max_start)

        if slice_value > max_start:
            QMessageBox.warning(self, "切片數不足",
                                f"最多只能從 index {max_start} 開始，以保證有 {self.min_required_slices} 張切片")
            return

        selected_button = self.align_group.checkedButton()
        if selected_button is None:
            QMessageBox.warning(self, "缺少選擇", "請選擇是否對齊")
            return

        alignment_value = self.align_group.id(selected_button)
        patient_id = self.patient_list[self.current_index]
        self.slice_index = slice_value
        self.slice_slider.setValue(slice_value)

        records = {}
        if os.path.exists(self.slice_record_path):
            with open(self.slice_record_path, "r") as f:
                for line in f:
                    if "-" in line:
                        pid, info = line.strip().split("-", 1)
                        records[pid.strip()] = info.strip()

        info_str = f"slice: {slice_value}, alignment: {alignment_value}"
        if alignment_value == 0:
            offsets = [w[1].text() for w in self.offset_widgets]
            for o in offsets:
                try:
                    int(o)
                except ValueError:
                    QMessageBox.warning(self, "輸入錯誤", f"Offset '{o}' 必須是整數。")
                    return
            offsets_str = ", ".join(offsets)
            info_str += f", offsets: [{offsets_str}]"

        confirm = True
        if patient_id in records:
            old_info = records[patient_id]
            msg = QMessageBox()
            msg.setWindowTitle("更新確認")
            msg.setText(f"病患 {patient_id} 已有記錄：{old_info}\n是否更新為新值: {info_str}？")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            result = msg.exec_()
            confirm = (result == QMessageBox.Yes)

        if confirm:
            records[patient_id] = info_str
            with open(self.slice_record_path, "w") as f:
                for pid, val in sorted(records.items()):
                    f.write(f"{pid} - {val}\n")

    def load_saved_slice_if_exists(self, patient_id):
        self.slice_index = 0
        self.align_group.setExclusive(False)
        self.align_yes.setChecked(False)
        self.align_no.setChecked(False)
        self.align_group.setExclusive(True)

        for _, line_edit in self.offset_widgets:
            line_edit.setText("0")
        self.toggle_offset_widgets(False)

        found = False
        if os.path.exists(self.slice_record_path):
            with open(self.slice_record_path, "r") as f:
                for line in f:
                    if "-" in line:
                        pid, info = line.strip().split("-", 1)
                        pid = pid.strip()
                        if pid == patient_id:
                            found = True
                            parts = info.split(",")
                            slice_val = int(parts[0].split(":")[1].strip())
                            align_val = int(parts[1].split(":")[1].strip())
                            self.slice_index = slice_val
                            if align_val == 1:
                                self.align_yes.setChecked(True)
                            elif align_val == 0:
                                self.align_no.setChecked(True)
                                self.toggle_offset_widgets(True)
                                if "offsets" in info:
                                    try:
                                        offsets_str = info.split("[")[1].split("]")[0]
                                        offsets = [o.strip() for o in offsets_str.split(",")]
                                        for i, offset in enumerate(offsets):
                                            if i < len(self.offset_widgets):
                                                self.offset_widgets[i][1].setText(offset)
                                    except IndexError:
                                        pass 
                            break
        
        self.slice_slider.setValue(self.slice_index)
        self.slice_input.setText(str(self.slice_index))

    def prev_patient(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.slice_index = 0
            self.load_patient_data()

    def next_patient(self):
        if self.current_index < len(self.patient_list) - 1:
            self.current_index += 1
            self.slice_index = 0
            self.load_patient_data()

    def find_min_slice_volume(self):
        min_slices = float('inf')
        min_file = None
        for fname in os.listdir(self.img_dir):
            if fname.endswith(".nii.gz"):
                path = os.path.join(self.img_dir, fname)
                try:
                    nii = nib.load(path)
                    shape = nii.shape
                    if len(shape) == 3 and shape[2] < min_slices and shape[2] > 1:
                        min_slices = shape[2]
                        min_file = fname
                except Exception as e:
                    print(f"Error reading {fname}: {e}")
        return min_file, min_slices


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    root_path = "../../data_folder/processed_data/first_version"
    viewer = NiiViewer(os.path.join(root_path, "imagesTr"), os.path.join(root_path, "labelsTr"))
    viewer.show()
    sys.exit(app.exec_())
