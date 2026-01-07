from typing import Any, Self
from dataclasses import dataclass
from faga.ai.models import ModelType
from faga.ai.model_params import (
  ModelParams,
  LoRAParams,
  EmbeddingParams,
  MP_NegativePrompt,
  MP_LoRAs,
  MP_Embeddings,
)


@dataclass
class T2I_SD15_Params(ModelParams, MP_NegativePrompt, MP_LoRAs, MP_Embeddings):

  @property
  def model_type(self) -> ModelType:
    return ModelType.SD15

  @classmethod
  def create_new(cls, model_name: str) -> Self:
    return cls(
      model=model_name,
      width=512,
      height=512,
      steps=25,
      cfg=6.0,
      sampler="euler",
      scheduler="simple",
      clip_skip=1,
      seed=0,
      batch_size=1,
      hires_fix=False,
      upscale_model="",
      loras=[],
      prompt="",
      negative="",
      embeddings=[]
    )

  @classmethod
  def create_from_values(cls, model_name: str, data: dict[str, Any]) -> Self:
    cdata = data.copy()
    cdata["model"] = model_name
    cdata["embeddings"] = [EmbeddingParams.from_dict(e) for e in cdata.get("embeddings", {})]
    cdata["loras"] = [LoRAParams.from_dict(l) for l in cdata.get("loras", {})]
    return cls(**cdata)
