from dataclasses import dataclass, replace
from typing import Self
from faga.ai.model_params import ModelType


@dataclass
class LoRA:
  type: ModelType
  name: str
  url: str
  strength: float

  def clone(self) -> Self:
    return replace(self)


@dataclass
class Embedding:
  type: ModelType
  name: str
  url: str

  def clone(self) -> Self:
    return replace(self)
