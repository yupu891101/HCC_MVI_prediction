import shutil
from pathlib import Path


def rename_patient_folder(data_root: str) -> None:
    """ rename all patient in raw_data/MVI from "1 (1mm)" to "001"
    """
    data_root_path = Path(data_root)
    mvi_data_path_list = data_root_path.glob("*")
    for data_folder in mvi_data_path_list:
        name_splits = data_folder.name.split(" ")

        folder_dir = data_folder.parent
        new_folder_name = name_splits[0].zfill(3)
        new_path = folder_dir.joinpath(new_folder_name)
        try:
            shutil.move(data_folder, new_path)
        except FileNotFoundError as e:
            print(e)

def rename_phase_folder(data_root: str) -> None:
    data_root_path = Path(data_root)
    patient_folders = data_root_path.glob("*[!.json]")
    for single_patient in patient_folders:
        phase_folders = sorted(single_patient.glob("*[!.jpg]"))
        phase_names = [
            "0_non_contrast_phase",
            "1_arterial_phase",
            "2_protal_venous_phase",
            "3_delayed_phase"
        ]

        phase_names = phase_names if len(phase_folders) == 4 else phase_names[1:]
        if len(phase_folders) < 3:
            raise FileNotFoundError(
                f"The CT PHASE num in folder: {single_patient} dose not contains enough PHASE (at least 3)."
            )
        for new_name, old_path in zip(phase_names, phase_folders):
            new_path = old_path.parent.joinpath(new_name)
            shutil.move(old_path, new_path)
