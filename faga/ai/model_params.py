from dataclasses import dataclass
from typing import Any, Self
from faga.ai.models import ModelType


@dataclass
class LoRAParams:
  name: str
  strength: float

  @classmethod
  def from_dict(cls, data: Any) -> Self:
    if isinstance(data, cls):
      return data
    if isinstance(data, dict):
      return cls(**data)
    raise TypeError("Can't convert to LoraParams")


@dataclass
class EmbeddingParams:
  name: str

  @classmethod
  def from_dict(cls, data: Any) -> Self:
    if isinstance(data, cls):
      return data
    if isinstance(data, dict):
      return cls(**data)
    raise TypeError("Can't convert to EmbeddingParams")


@dataclass
class ModelParams:
  model: str
  width: int
  height: int
  steps: int
  sampler: str
  scheduler: str
  batch_size: int
  seed: int
  prompt: str
  cfg: float
  clip_skip: int
  hires_fix: bool
  upscale_model: str

  @property
  def model_type(self) -> ModelType:
    raise NotImplementedError()

  @classmethod
  def create_new(cls, model_name: str) -> Self:
    raise NotImplementedError()

  @classmethod
  def create_from_values(cls, model_name: str, data: dict[str, Any]) -> Self:
    raise NotImplementedError()


@dataclass
class MP_NegativePrompt:
  negative: str


@dataclass
class MP_TagPrompt:
  tag_prompt: str
  guidance: float


@dataclass
class MP_LoRAs:
  loras: list[LoRAParams]


@dataclass
class MP_Embeddings:
  embeddings: list[EmbeddingParams]
