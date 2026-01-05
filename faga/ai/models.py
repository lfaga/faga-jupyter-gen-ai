from enum import Enum
from dataclasses import dataclass


class Architecture(Enum):
  CHECKPOINT = "checkpoint"
  MODULAR = "modular"


class TensorFileType(Enum):
  checkpoint = "checkpoint"
  unet = "unet"
  text_encoder = "text_encoder"
  clip_l = "clip_l"
  vae = "vae"


class ClipType(Enum):
  STABLE_DIFFUSION = "STABLE_DIFFUSION"
  STABLE_CASCADE = "STABLE_CASCADE"
  SD3 = "SD3"
  STABLE_AUDIO = "STABLE_AUDIO"
  HUNYUAN_DIT = "HUNYUAN_DIT"
  FLUX = "FLUX"
  MOCHI = "MOCHI"
  LTXV = "LTXV"
  HUNYUAN_VIDEO = "HUNYUAN_VIDEO"
  PIXART = "PIXART"
  COSMOS = "COSMOS"
  LUMINA2 = "LUMINA2"
  WAN = "WAN"
  HIDREAM = "HIDREAM"
  CHROMA = "CHROMA"
  ACE = "ACE"
  OMNIGEN2 = "OMNIGEN2"
  QWEN_IMAGE = "QWEN_IMAGE"
  HUNYUAN_IMAGE = "HUNYUAN_IMAGE"
  HUNYUAN_VIDEO_15 = "HUNYUAN_VIDEO_15"
  OVIS = "OVIS"
  KANDINSKY5 = "KANDINSKY5"
  KANDINSKY5_IMAGE = "KANDINSKY5_IMAGE"
  NEWBIE = "NEWBIE"


class ModelType(Enum):
  SD15 = "SD15"
  FLUX1D = "FLUX1D"
  FLUX1S = "FLUX1S"
  SDXL = "SDXL"
  LUMINA2 = "LUMINA2"
  HUNYUAN_DIT = "HUNYUAN_DIT"


class WeightType(Enum):
  default = "default"
  fp8_e4m3fn = "fp8_e4m3fn"
  fp8_e4m3fn_fast = "fp8_e4m3fn_fast"
  fp8_e5m2 = "fp8_e5m2"


# ----------------------------------


@dataclass
class TensorFile:
  name: str
  url: str
  type: ClipType | None = None
  is_gguf: bool = False
  weight_type: WeightType | None = None


@dataclass
class Model:
  architecture: Architecture
  base_size_px: int
  type: ModelType
  checkpoint: TensorFile | None = None
  unet: TensorFile | None = None
  text_encoder: TensorFile | None = None
  clip_l: TensorFile | None = None
  vae: TensorFile | None = None
