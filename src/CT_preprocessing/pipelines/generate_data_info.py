import os
import sys
import json
from pathlib import Path

import pydicom
from tqdm.auto import tqdm


if __package__ :
    from ..preprocessing_configs import DataInfoPipelineConfig as Config
    from ..utils.path_modify_utils import rename_patient_folder, rename_phase_folder, rename_label_file
else:
    # import for multiprocessing and directly run
    sys.path.append("./src/CT_preprocessing/")
    from preprocessing_configs import DataInfoPipelineConfig as Config
    from utils.path_modify_utils import rename_patient_folder, rename_phase_folder, rename_label_file

class DataInfoPipeline:
    def __init__(self, config: Config):
        self.config = config

    def parsing_dicom_info(self, patient_folder: Path):
        dicom_info = {}
        for phase_pholder in sorted(patient_folder.glob("*[!.jpg]")):
            phase_name = phase_pholder.name
            dicom_info[phase_name] = {}
            dicom_file_path = next(phase_pholder.glob("*.dcm"))
            dicom = pydicom.dcmread(dicom_file_path)
            dicom_info[phase_name]['Manufacturer'] = getattr(dicom, 'Manufacturer', 'Unknown')
            dicom_info[phase_name]['Slope'] = getattr(dicom, 'RescaleSlope', 1)
            dicom_info[phase_name]['Intercept'] = getattr(dicom, 'RescaleIntercept', 0)
        return dicom_info

    def gen_data_info(self, data_root: Path):
        data_info = {}
        all_patients_folder = sorted(data_root.glob("*[!.json]"))
        print("Parsing all patient's CT info ...")
        for patient_folder in tqdm(all_patients_folder):
            data_info[patient_folder.name] = self.parsing_dicom_info(patient_folder)
        print("All parsing Done")
        return data_info

    def run(self,):
        if not os.path.exists(self.config.raw_data_path):
            raise FileNotFoundError(
                f"The root path of your data: {self.config.raw_data_path} is not exist."
            )
        print("Rename all patient folder ...", end="")
        rename_patient_folder(self.config.raw_data_path)
        print("\rRename all patient folder ... Done")
        print("Rename all phase folder ...", end="")
        rename_phase_folder(self.config.raw_data_path)
        print("\rRename all phase folder ... Done")
        print("Rename all label file ...", end="")
        rename_label_file(self.config.label_data_path)
        print("\rRename all label file ... Done")
        data_info = self.gen_data_info(self.config.raw_data_path)
        with open(self.config.data_info_json_path, "w", encoding="utf-8") as j:
            json.dump(data_info, j, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    data_pipeline = DataInfoPipeline(Config())
    data_pipeline.run()
