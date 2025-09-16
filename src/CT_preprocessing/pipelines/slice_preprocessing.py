import os
import sys
import importlib
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
    sys.path.append("./src/CT_preprocessing/")
    preprocessing_configs = importlib.import_module("preprocessing_configs")
    preprocess_utils = importlib.import_module("utils.preprocess_utils")
    # 2. 從載入的模組中存取類別
    Config = preprocessing_configs.SlicePreprocessingConfig
    EqualizeConfig = preprocessing_configs.EqualizeConfig
    normalized_cross_correlation_fft = preprocess_utils.normalized_cross_correlation_fft
    slice_resize = preprocess_utils.slice_resize


# from ..preprocessing_configs import SlicePreprocessingConfig as Config
# from ..preprocessing_configs import EqualizeConfig
# from ..utils.preprocess_utils import normalized_cross_correlation_fft, slice_resize

@dataclass
class PatientDicom:
    patient_id: str
    value_range: tuple[int, int]
    arterial_phase: np.ndarray = field(repr=False)
    portal_venous_phase: np.ndarray = field(repr=False)
    delayed_phase: np.ndarray = field(repr=False)

    proc_arterial_phase: np.ndarray | None = field(default=None, init=False, repr=False)
    proc_portal_venous_phase: np.ndarray | None = field(default=None, init=False, repr=False)
    proc_delayed_phase: np.ndarray | None = field(default=None, init=False, repr=False)

    min_clip_val: int | float | None = field(default=-1024, init=False)
    max_clip_val: int | float | None = field(default=1024, init=False)
    align_result: dict[str, int] | None = field(default=None, init=False)
    all_proc_phase: dict[str, np.ndarray] = field(init=False, repr=False)
    all_phase: dict[str, np.ndarray] = field(init=False, repr=False)
    ref_phase: Literal[
        "arterial_phase", "portal_venous_phase", "delayed_phase"
    ] | None = field(default=None, init=False)
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

    def clip(self, dicom_array: np.ndarray):
        return np.clip(
            dicom_array,
            min=self.min_clip_val, max=self.max_clip_val,
            dtype=np.float32
        )

    def _normalize(self, dicom_array: np.ndarray):
        dicom_array = self.clip(dicom_array)
        dicom_array = (dicom_array - self.min_clip_val) / (self.max_clip_val - self.min_clip_val)
        return dicom_array

    def set_align_result(self, result: dict[str, int]):
        self.align_result = result

    def set_ref_phase(self, phase_name: str):
        self.ref_phase = phase_name


class DataProducer:
    def __init__(self ,config: Config):
        self.config = config

    @staticmethod
    def read_all_dicom(phase_folder: Path) -> dict[str, np.ndarray]:
        print(phase_folder.name, "start")
        if "non_contrast" in phase_folder.name:
            return dict()
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(phase_folder))
        dicom_array = sitk.GetArrayFromImage(reader.Execute())
        phase_name = "_".join(phase_folder.name.split("_")[1:])
        print(phase_folder.name, "done")
        return {phase_name: dicom_array}

    def read_all_phase(self, patient_folder: Path) -> PatientDicom:
        patient_dicom = {"patient_id": patient_folder.name, "value_range": self.config.value_range}
        with ThreadPoolExecutor(max_workers=self.config.num_producer_thread) as excutor:
            dicom_results = excutor.map(self.read_all_dicom, patient_folder.glob("*[!.jpg]"))
        for result in dicom_results:
            patient_dicom.update(result)
        return PatientDicom(**patient_dicom)

    def run(self, folder_queue: Queue[Path], dicom_queue: Queue[PatientDicom], early_stop = False):
        print("Start to load file")
        while True:
            patient_folder = folder_queue.get()
            if patient_folder is None:
                break
            print(f"start to load {patient_folder} dicom")
            patient_dicom = self.read_all_phase(patient_folder)
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

    def cal_similarity(self, ref_slice: np.ndarray, target_slice: np.ndarray) -> float:
        correlation_array = normalized_cross_correlation_fft(ref_slice, target_slice)
        score = correlation_array.max()
        return score

    def slice_augment(self, dicom_array: np.ndarray) -> PatientDicom:
        dicom_array = self.slice_equalize(dicom_array, self.config.equalize_config)
        dicom_array = slice_resize(dicom_array, self.config.target_slice_size)
        return dicom_array

    def _slice_align(self, slice_indecies: tuple[int], ref_dicom: np.ndarray, target_dicom: np.ndarray) -> int:
        offset_index_list: list[int] = []
        for slice_index in slice_indecies:
            ref_slice = ref_dicom[slice_index]
            score_list: list[float] = []
            for target_slice in target_dicom:
                score_list.append(self.cal_similarity(ref_slice, target_slice))
            offset_index_list.append(score_list.index(max(score_list)) - slice_index)
        return round(sum(offset_index_list) / len(offset_index_list))

    def slice_align(self, patient_dicom: PatientDicom) -> PatientDicom:
        shortest_phase_name = min(patient_dicom.all_proc_phase, key=lambda k: len(patient_dicom.all_proc_phase[k]))
        ref_phase = patient_dicom.all_proc_phase[shortest_phase_name]
        patient_dicom.set_ref_phase(shortest_phase_name)
        offset_result = dict()
        for phase_name, target_pahse in patient_dicom.all_proc_phase.items():
            if phase_name == shortest_phase_name:
                continue
            offset_result[phase_name] = self._slice_align(
                self.config.align_slice_index,
                ref_phase,
                target_pahse
            )
        patient_dicom.set_align_result(offset_result)

    def process_dicom(self, patient_dicom: PatientDicom) -> PatientDicom:
        for phase_dicom in patient_dicom.all_proc_phase.values():
            phase_dicom = self.slice_augment(phase_dicom)
        return patient_dicom

    def dicom_export_aligment(
        self,
        patient_dicom: PatientDicom
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
        # 1. 計算所有列表需要的總長度

        # 計算所有列表需要的最長前端 padding
        # 這由最大的 offset 決定（因為 ref_list 向右偏移）
        ref_phase = patient_dicom.ref_phase
        max_offset = max(0, *list(patient_dicom.align_result.values()))

        # 每個列表的總長度 = 自己的長度 + 自己的 offset + 最大 offset
        # 取所有列表中的最大長度作為最終總長度
        total_length = max(
            patient_dicom.all_phase[ref_phase].shape[0] + max_offset,
            *[
                patient_dicom.all_phase[tgt_phase].shape[0] + max_offset - offset
                for tgt_phase, offset in patient_dicom.align_result.items()
            ]
        )

        for phase_name, phase_dicom in patient_dicom.all_phase.items():
            front_padding = (
                max_offset if phase_name == ref_phase
                else max_offset - patient_dicom.align_result[phase_name]
            )
            back_padding = total_length - phase_dicom.shape[0] - front_padding
            padding_width = ((front_padding, back_padding), (0, 0), (0, 0))
            phase_dicom = np.pad(
                phase_dicom, padding_width, mode="constant", constant_values=self.config.padding_value
            )

    def save_dicom_to_nii(self, patient_dicom: PatientDicom):
        for idx, (phase_name, phase_dicom) in enumerate(patient_dicom.all_phase.items()):
            phase_id = str(idx + 1).zfill(4)
            print(f"Saving {phase_name} of patient {patient_dicom.patient_id} with id: {phase_id}")
            save_path = self.config.save_data_path.joinpath(
                f"imagesTr/HCC_{patient_dicom.patient_id}_{phase_id}.nii.gz"
            )
            print("saving path:", os.path.abspath(save_path))
            sitk.WriteImage(sitk.GetImageFromArray(phase_dicom), save_path)

    def export_dicom(self, patient_dicom: PatientDicom):
        for phase_dicom in patient_dicom.all_phase.values():
            phase_dicom = patient_dicom.clip(phase_dicom)
            phase_dicom = slice_resize(phase_dicom, self.config.target_slice_size, "Nearest")

        self.dicom_export_aligment(patient_dicom)
        self.save_dicom_to_nii(patient_dicom)

    def run(self, dicom_queue: Queue[PatientDicom]):
        while True:
            print(dicom_queue)
            patient_dicom = dicom_queue.get()
            if patient_dicom is None:
                print("leave consumer")
                break
            print(f"Start to process {patient_dicom}")
            patient_dicom = self.process_dicom(patient_dicom)
            print(f"process {patient_dicom} done")
            self.slice_align(patient_dicom)
            self.export_dicom(patient_dicom)
            print(f"export {patient_dicom} done")

class SlicePreprocessingPipeline():
    def __init__(self):
        self.config = Config()

    def run(self):
        data_root = self.config.raw_data_path
        folder_queue = mQueue()
        dicom_queue = mQueue()
        for patient_folder in data_root.glob("*[!.json]"):
            folder_queue.put(patient_folder)
        for _ in range(self.config.num_producer):
            folder_queue.put(None)

        producer_processes = [
            Process(target=DataProducer(self.config).run, args=(folder_queue, dicom_queue, True))
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
            print("producer joined")
        for _ in range(self.config.num_consumer):
            print("put none to queue")
            dicom_queue.put(None)
            print("put none to done")
        for p in consumer_processes:
            p.join()
            print("consumer joined")
        folder_queue.cancel_join_thread()
        dicom_queue.cancel_join_thread()

if __name__ == "__main__":
    print(os.getcwd())
    pipeline = SlicePreprocessingPipeline()
    pipeline.run()
    print("all task done")
