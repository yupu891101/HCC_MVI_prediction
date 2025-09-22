import os
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class DataInfoPipelineConfig:
    raw_data_path: Path = Path("./data_folder/raw_data/MVI/").absolute()
    label_data_path: Path = Path("./data_folder/raw_data/label/").absolute()
    data_info_json_path: Path = raw_data_path.joinpath("data_info.json")

@dataclass
class EqualizeConfig:
    _value_range: tuple[int, int]
    kernel_size_ratio: float = 1/16
    clip_limit: float = 0.005
    nbins: int | None = field(init=False, default=None)
    def __post_init__(self,):
        self.nbins = max(self._value_range) - min(self._value_range)

@dataclass
class SlicePreprocessingConfig:
    raw_data_path: Path = DataInfoPipelineConfig.raw_data_path
    label_data_path: Path = DataInfoPipelineConfig.label_data_path
    save_data_path: Path = Path("./data_folder/processed_data/first_version").absolute()

    num_producer: int = 1
    num_producer_thread: int = 4
    num_consumer: int = 1
    max_dicom_queue: int = 1

    target_slice_size: tuple[int, int] = (512, 512)
    value_range: tuple[int, int] = (100, 1024)
    export_value_range: tuple[int, int] = (-1024, 1024)
    align_slice_percentile: tuple[int] = tuple({5, 25, 50, 75, 95})
    padding_value: int = 0
    split_num: int = 5
    top_k_offset: int = 2
    _equalize_config = EqualizeConfig(value_range)
    
    def __post_init__(self,):
        image_tr_path = self.save_data_path.joinpath("imagesTr")
        image_ts_path = self.save_data_path.joinpath("imagesTs")
        label_path = self.save_data_path.joinpath("labelsTr")
        os.makedirs(image_tr_path, exist_ok=True)
        os.makedirs(image_ts_path, exist_ok=True)
        os.makedirs(label_path, exist_ok=True)
        if self.top_k_offset > len(self.align_slice_percentile):
            raise ValueError(
                f"top_k_offset should be greater than len(align_slice_percentile): {len(self.align_slice_percentile)}"
            )

    @property
    def equalize_config(self,):
        return self._equalize_config

    def get_value_range(self,) -> tuple[int, int]:
        return self.value_range
