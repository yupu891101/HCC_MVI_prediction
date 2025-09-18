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
            "2_portal_venous_phase",
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

def grouping_label_path(label_root: Path) -> dict[int, list[str]]:
    label_grouping_dict: dict[int, list[str]] = dict()
    for label_path in label_root.glob("*.nrrd"):
        label_file_name = label_path.name
        patient_id = int(label_file_name.split("_")[0])
        if patient_id not in label_grouping_dict:
            label_grouping_dict[patient_id] = []
        label_grouping_dict[patient_id].append(label_path.as_posix())
    return dict(sorted(label_grouping_dict.items(), key=lambda item: item[0]))

def rename_label_file(label_root: Path):
    label_dict = grouping_label_path(label_root)
    for patient_id, label_files in label_dict.items():
        label_files.sort()
        phase_names = [
            "0_non_contrast_phase",
            "1_arterial_phase",
            "2_portal_venous_phase",
            "3_delayed_phase"
        ]
        if len(label_files) < 4:
            phase_names.pop(0)
        for label_path, phase_name in zip(label_files, phase_names):
            label_path = Path(label_path).absolute()
            new_name = f"{patient_id:03d}_{phase_name}.nrrd"
            new_path = label_path.parent.joinpath(new_name).absolute()
            shutil.move(label_path, new_path)
