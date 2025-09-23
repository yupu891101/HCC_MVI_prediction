import sys
from queue import Queue
from pathlib import Path
from typing import Literal
from dataclasses import dataclass, field
from multiprocessing import Process
from multiprocessing import Queue as mQueue
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import SimpleITK as sitk
from skimage import exposure

if __package__ :
    from ..preprocessing_configs import SlicePreprocessingConfig as Config
    from ..preprocessing_configs import EqualizeConfig
    from ..utils.preprocess_utils import normalized_cross_correlation_fft, slice_resize
else:
    # import for multiprocessing and directly run
    sys.path.append("./src/CT_preprocessing/")
    from preprocessing_configs import SlicePreprocessingConfig as Config
    from preprocessing_configs import EqualizeConfig
    from utils.preprocess_utils import normalized_cross_correlation_fft, slice_resize

Confidence = float
Offset = int
PhaseNames = Literal["arterial_phase", "portal_venous_phase", "delayed_phase"] | str
LabelNames = Literal["arterial_label", "portal_venous_label", "delayed_label"] | str
@dataclass
class PatientDicom:
    patient_id: str
    slice_thickness: int
    value_range: tuple[int, int]
    arterial_phase: np.ndarray = field(repr=False)
    portal_venous_phase: np.ndarray = field(repr=False)
    delayed_phase: np.ndarray = field(repr=False)
    arterial_label: np.ndarray | None = field(default=None, repr=False)
    portal_venous_label: np.ndarray | None = field(default=None, repr=False)
    delayed_label: np.ndarray | None = field(default=None, repr=False)

    proc_arterial_phase: np.ndarray | None = field(default=None, init=False, repr=False)
    proc_portal_venous_phase: np.ndarray | None = field(default=None, init=False, repr=False)
    proc_delayed_phase: np.ndarray | None = field(default=None, init=False, repr=False)

    min_clip_val: int | float = field(default=-1024, init=False)
    max_clip_val: int | float = field(default=1024, init=False)
    align_result: dict[PhaseNames, int] | None = field(default=None, init=False)
    all_proc_phase: dict[PhaseNames, np.ndarray] = field(init=False, repr=False)
    all_phase: dict[PhaseNames, np.ndarray] = field(init=False, repr=False)
    all_label: dict[LabelNames, np.ndarray | None] = field(init=False, repr=False)
    ref_phase: PhaseNames | None = field(default=None, init=False)
    def __post_init__(self,):
        self.min_clip_val = min(self.value_range)
        self.max_clip_val = max(self.value_range)
        self.proc_arterial_phase = self._normalize(self.arterial_phase)
        self.proc_portal_venous_phase = self._normalize(self.portal_venous_phase)
        self.proc_delayed_phase = self._normalize(self.delayed_phase)
        self.all_proc_phase = {
            "arterial_phase": self.proc_arterial_phase,
            "portal_venous_phase": self.proc_portal_venous_phase,
            "delayed_phase": self.proc_delayed_phase
        }
        self.all_phase = {
            "arterial_phase": self.arterial_phase,
            "portal_venous_phase": self.portal_venous_phase,
            "delayed_phase": self.delayed_phase
        }
        self.all_label = {
            "arterial_label": self.arterial_label,
            "portal_venous_label": self.portal_venous_label,
            "delayed_label": self.delayed_label
        }

    def clip(self, dicom_array: np.ndarray, value_range: tuple[int, int] | None = None):
        min_val, max_val = value_range or (self.min_clip_val, self.max_clip_val)
        return dicom_array.clip(
            min=min_val, max=max_val,
            dtype=np.float32
        )

    def _normalize(self, dicom_array: np.ndarray):
        dicom_array = self.clip(dicom_array)
        dicom_array = (dicom_array - self.min_clip_val) / (self.max_clip_val - self.min_clip_val)
        return dicom_array

    def set_align_result(self, result: dict[str, int]):
        self.align_result = result

    def set_ref_phase(self, phase_name: PhaseNames):
        self.ref_phase = phase_name

class DataProducer:
    def __init__(self ,config: Config):
        self.config = config

    def read_all_dicom(self, phase_folder: Path) -> dict[str, np.ndarray | int]:
        print(phase_folder.name, "start")
        if "non_contrast" in phase_folder.name:
            return dict()
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(phase_folder))
        slice_instance = reader.Execute()
        dicom_array = sitk.GetArrayFromImage(slice_instance)
        slice_thickness = self.get_thickness(slice_instance)
        phase_name = "_".join(phase_folder.name.split("_")[1:])
        print(phase_folder.name, "done")
        return {phase_name: dicom_array, "slice_thickness": slice_thickness}

    @staticmethod
    def read_all_label(label_file: Path) -> dict[str, np.ndarray]:
        print(label_file.stem, "start")
        if "non_contrast" in label_file.stem:
            return dict()
        label_array = sitk.GetArrayFromImage(sitk.ReadImage(label_file))
        label_name = "_".join(label_file.stem.split("_")[2:]).replace("phase", "label")
        print(label_file.stem, "done")
        return {label_name: label_array}

    def get_thickness(self, slice_instance: sitk.Image) -> int:
        slice_size: tuple = slice_instance.GetSize()
        slice_thickness_index = slice_size.index(min(slice_size))
        return int(slice_instance.GetSpacing()[slice_thickness_index])

    def read_all_phase(self, patient_folder: Path) -> PatientDicom:
        patient_dicom = {"patient_id": patient_folder.name, "value_range": self.config.value_range}
        with ThreadPoolExecutor(max_workers=self.config.num_producer_thread) as excutor:
            dicom_results = excutor.map(self.read_all_dicom, patient_folder.glob("*[!.jpg]"))

        with ThreadPoolExecutor(max_workers=self.config.num_producer_thread) as excutor:
            label_results = excutor.map(
                self.read_all_label,
                self.config.label_data_path.glob(f"{patient_folder.name}*.nrrd")
            )

        for result in dicom_results:
            patient_dicom.update(result)
        for label in label_results:
            patient_dicom.update(label)
        return PatientDicom(**patient_dicom)

    def validate_label(self, patient_dicom: PatientDicom):
        if not (
            all(label is None for label in patient_dicom.all_label.values()) or
            all(label is not None for label in patient_dicom.all_label.values())
        ):
            missing_phase = [
                phase for phase, label in patient_dicom.all_label.items()
                if label is None
            ]
            raise FileNotFoundError(f"Missing {missing_phase} label file of patient: {patient_dicom.patient_id}")

    def run(self, folder_queue: Queue[Path], dicom_queue: Queue[PatientDicom], early_stop = False):
        print("Start to load file")
        while True:
            patient_folder = folder_queue.get()
            if patient_folder is None:
                break
            while not dicom_queue.qsize() < self.config.max_dicom_queue:
                pass
            print(f"start to load {patient_folder} dicom")
            patient_dicom = self.read_all_phase(patient_folder)
            self.validate_label(patient_dicom)
            print(f"load {patient_folder} dicom done")
            dicom_queue.put(patient_dicom)
            if early_stop:
                print("Early stop activate")
                break

class DataConsumer:
    def __init__(self, config: Config):
        self.config = config

    def slice_equalize(self, dicom_array: np.ndarray, config: EqualizeConfig):
        """dicom_array shape: (slice_num, H, W)"""
        for idx, dicom_slice in enumerate(dicom_array):
            dicom_array[idx, :, :] = exposure.equalize_adapthist(
                dicom_slice,
                kernel_size = int(dicom_slice.shape[0] * config.kernel_size_ratio),
                clip_limit = config.clip_limit,
                nbins = config.nbins
            )
        return dicom_array

    @staticmethod
    def get_confidence(scores: np.ndarray) -> float:
        max_score = scores.max()

        # 創建一個不包含最高分的 numpy 陣列
        other_scores = scores[scores<max_score]
        average_of_others = np.mean(other_scores)
        std_of_others = np.std(other_scores)

        # 計算標準化強度
        strength = (max_score - average_of_others) / std_of_others

        return strength

    @staticmethod
    def cal_similarity(ref_slice: np.ndarray, target_slice: np.ndarray) -> float:
        correlation_array = normalized_cross_correlation_fft(ref_slice, target_slice)
        score = correlation_array.max()
        return score

    def slice_augment(self, dicom_array: np.ndarray) -> np.ndarray:
        dicom_array = self.slice_equalize(dicom_array, self.config.equalize_config)
        dicom_array = slice_resize(dicom_array, self.config.target_slice_size)
        return dicom_array

    def _slice_align(self, slice_percentile: tuple[int, ...], ref_dicom: np.ndarray, target_dicom: np.ndarray) -> int:
        import time
        offset_dict: dict[Confidence, Offset] = dict()
        slice_indecies = [int(percentile / 100 * len(ref_dicom)) for percentile in slice_percentile]
        for slice_index in slice_indecies:
            ref_slice = ref_dicom[slice_index]
            score_list: list[float] = []
            for target_slice in target_dicom:
                score_list.append(self.cal_similarity(ref_slice, target_slice))
            np.save(f"./{int(time.time())}.npy", np.array(score_list))
            confidence = self.get_confidence(np.array(score_list))

            offset_dict[confidence] = score_list.index(max(score_list)) - slice_index
        offset_list = [v for _, v in sorted(offset_dict.items(), reverse=True)][:self.config.top_k_offset]
        return round(sum(offset_list) / len(offset_list))

    def slice_align(self, patient_dicom: PatientDicom):
        shortest_phase_name = min(patient_dicom.all_proc_phase, key=lambda k: len(patient_dicom.all_proc_phase[k]))
        ref_phase = patient_dicom.all_proc_phase[shortest_phase_name]
        patient_dicom.set_ref_phase(shortest_phase_name)
        offset_result = dict()
        for phase_name, target_pahse in patient_dicom.all_proc_phase.items():
            if phase_name == shortest_phase_name:
                offset_result[phase_name] = 0
                continue
            offset_result[phase_name] = self._slice_align(
                self.config.align_slice_percentile,
                ref_phase,
                target_pahse
            )
        patient_dicom.set_align_result(offset_result)

    def process_dicom(self, patient_dicom: PatientDicom) -> PatientDicom:
        for (phase_name, phase_dicom), (label_phase_name, phase_label) in zip(
            patient_dicom.all_proc_phase.items(),
            patient_dicom.all_label.items()
        ):
            patient_dicom.all_proc_phase[phase_name] = self.slice_augment(phase_dicom)
            patient_dicom.all_label[label_phase_name] = (
                slice_resize(phase_label, self.config.target_slice_size, "Nearest")
                if phase_label is not None else None
            )
            if phase_label is not None:
                if len(phase_dicom) != len(phase_label):
                    raise ValueError(
                        f"The lenth of CT from patient: {patient_dicom.patient_id}"
                        f" of phase: {phase_name} is different from its label: {label_phase_name}."
                        f" CT shape: {phase_dicom.shape}, label shape: {phase_label.shape}."
                    )
        return patient_dicom

    def alignment(
        self,
        input_align_result: dict[PhaseNames, int] | None,
        input_process_dict: dict[PhaseNames, np.ndarray] | dict[LabelNames, np.ndarray | None],
        input_ref_phase: PhaseNames | None,
        target_instance: Literal["phase", "label"]
    ):
        """
        根據多個 offset 對齊並用 padding_val 補齊 ref_list 和多個 tgt_list。

        Args:
            ref_list (List): 參考列表。
            targets (Dict[str, Tuple[List, int]]): 包含多個目標列表和其對應 offset 的字典。
                                                鍵為列表名稱，值為 (列表, offset)。
            padding_val (Any): 用來補齊的數值。預設為 -999。

        Returns:
            Dict[str, List]: 包含所有補齊後列表的字典。
        """
        if input_align_result is None:
            raise ValueError("align_result should not be None")
        if input_ref_phase is None:
            raise ValueError("ref_phase should not be None")
        if any(value is None for value in input_process_dict.values()):
            return None
        process_dict: dict[PhaseNames, np.ndarray] | dict[LabelNames, np.ndarray] = input_process_dict
        if target_instance == "label":
            ref_phase = input_ref_phase.replace("phase", "label")
            align_result = {
                phase.replace("phase", "label"): offset
                for phase, offset in input_align_result.items()
            }
        else:
            ref_phase = input_ref_phase
            align_result = input_align_result

        max_offset = max(0, *list(align_result.values()))
        # 1. 計算所有array需要的總長度
        # 每個列表的總長度 = 自己的長度 + 自己的 offset + 最大 offset
        # 取所有列表中的最大長度作為最終總長度
        for tgt_phase, offset in align_result.items():
            print(f"phase: {tgt_phase}", "phase_length:", process_dict[tgt_phase].shape[0], "offset:", offset)
        total_length = max(
            process_dict[ref_phase].shape[0] + max_offset,
            *[
                process_dict[tgt_phase].shape[0] + max_offset - offset
                for tgt_phase, offset in align_result.items()
            ]
        )
        for phase_name, phase_array in process_dict.items():
            front_padding = (
                max_offset if phase_name == ref_phase
                else max_offset - align_result[phase_name]
            )
            back_padding = total_length - phase_array.shape[0] - front_padding
            padding_width = ((front_padding, back_padding), (0, 0), (0, 0))
            process_dict[phase_name] = np.pad(
                phase_array, padding_width, mode="constant", constant_values=self.config.padding_value
            )

    def dicom_export_aligment(self,patient_dicom: PatientDicom):
        self.alignment(
            patient_dicom.align_result,
            patient_dicom.all_phase,
            patient_dicom.ref_phase,
            "phase"
        )
        self.alignment(
            patient_dicom.align_result,
            patient_dicom.all_label,
            patient_dicom.ref_phase,
            "label"
        )

    def _save_dicom_to_nii(
            self,
            phase_dicom: np.ndarray,
            phase_label: np.ndarray | None,
            save_category: str,
            patient_id: str,
            phase_id: int,
            need_split: bool
        ):
        if need_split:
            split_dicoms: list[np.ndarray] = [
                phase_dicom[i::self.config.split_num] for i in range(self.config.split_num)
            ]
            split_labels: list[np.ndarray | None] = [
                phase_label[i::self.config.split_num] for i in range(self.config.split_num)
            ] if phase_label is not None else [None] * self.config.split_num
        else:
            split_dicoms = [phase_dicom]
            split_labels = [phase_label]

        for idx, (dicom, label) in enumerate(zip(split_dicoms, split_labels)):
            save_name = f"{patient_id}G{idx:02d}P{phase_id}"
            dicom_save_path = self.config.save_data_path.joinpath(
                f"{save_category}/HCC_{save_name}_{phase_id:04d}.nii.gz"
            )
            label_save_path = self.config.save_data_path.joinpath(
                f"labelsTr/HCC_{save_name}.nii.gz"
            )
            sitk.WriteImage(sitk.GetImageFromArray(dicom), dicom_save_path)
            if label is not None:
                sitk.WriteImage(sitk.GetImageFromArray(label), label_save_path)

    def save_dicom_to_nii(self, patient_dicom: PatientDicom):
        save_category = "imagesTs" if patient_dicom.arterial_label is None else "imagesTr"
        need_split = patient_dicom.slice_thickness < 5

        for idx, (phase_name, phase_dicom) in enumerate(patient_dicom.all_phase.items()):
            phase_id = idx + 1
            self._save_dicom_to_nii(
                phase_dicom,
                patient_dicom.all_label[phase_name.replace("phase", "label")],
                save_category,
                patient_dicom.patient_id,
                phase_id,
                need_split
            )

    def export_dicom(self, patient_dicom: PatientDicom):
        for phase_name, phase_dicom in patient_dicom.all_phase.items():
            patient_dicom.all_phase[phase_name] = patient_dicom.clip(phase_dicom, self.config.export_value_range)
            patient_dicom.all_phase[phase_name] = slice_resize(
                patient_dicom.all_phase[phase_name], self.config.target_slice_size, "Nearest"
            )

        self.dicom_export_aligment(patient_dicom)
        self.save_dicom_to_nii(patient_dicom)

    def run(self, dicom_queue: Queue[PatientDicom]):
        while True:
            patient_dicom = dicom_queue.get()
            if patient_dicom is None:
                print("leave consumer")
                break
            print(f"Start to process {patient_dicom}")
            patient_dicom = self.process_dicom(patient_dicom)
            print(f"process {patient_dicom} done")
            print(f"Start to align {patient_dicom}")
            self.slice_align(patient_dicom)
            print(f"Align {patient_dicom} done")
            self.export_dicom(patient_dicom)
            print(f"export {patient_dicom} done")

    def align_patient(self, patient_dicom: PatientDicom) -> PatientDicom:
        print(f"Start to process {patient_dicom}")
        patient_dicom = self.process_dicom(patient_dicom)
        print(f"process {patient_dicom} done")
        print(f"Start to align {patient_dicom}")
        self.slice_align(patient_dicom)
        print(f"Align {patient_dicom} done")
        return patient_dicom

class SlicePreprocessingPipeline:
    def __init__(self):
        self.config = Config()

    def run(self):
        data_root = self.config.raw_data_path
        folder_queue = mQueue()
        dicom_queue = mQueue(maxsize=self.config.max_dicom_queue)
        for patient_folder in data_root.glob("*[!.json]"):
            folder_queue.put(patient_folder)
        for _ in range(self.config.num_producer):
            folder_queue.put(None)

        producer_processes = [
            Process(target=DataProducer(self.config).run, args=(folder_queue, dicom_queue))
            for _ in range(self.config.num_producer)
        ]
        consumer_processes = [
            Process(target=DataConsumer(self.config).run, args=(dicom_queue, ))
            for _ in range(self.config.num_consumer)
        ]
        for p in producer_processes:
            p.start()
        for p in consumer_processes:
            p.start()
        for p in producer_processes:
            p.join()
        for _ in range(self.config.num_consumer):
            dicom_queue.put(None)
        for p in consumer_processes:
            p.join()
        folder_queue.cancel_join_thread()
        dicom_queue.cancel_join_thread()

if __name__ == "__main__":
    pipeline = SlicePreprocessingPipeline()
    pipeline.run()
    print("all task done")
