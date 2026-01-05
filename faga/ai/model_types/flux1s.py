from typing import Any, Self
from dataclasses import dataclass
from faga.ai.models import ModelType
from faga.ai.model_params import (
  ModelParams,
  LoRAParams,
  MP_TagPrompt,
  MP_LoRAs,
)


@dataclass
class T2I_Flux1s_Params(ModelParams, MP_TagPrompt, MP_LoRAs):

  def get_type(self) -> ModelType:
    return ModelType.FLUX1S

  @classmethod
  def create_new(cls, model_name: str) -> Self:
    return cls(
      model=model_name,
      width=1024,
      height=1024,
      steps=25,
      cfg=1.0,
      sampler="euler",
      scheduler="simple",
      clip_skip=1,
      seed=0,
      batch_size=1,
      hires_fix=False,
      upscale_model="",
      loras=[],
      prompt="",
      tag_prompt="",
      guidance=1.0
    )

  @classmethod
  def create_from_values(cls, model_name: str, data: dict[str, Any]) -> Self:
    cdata = data.copy()
    cdata["model"] = model_name
    cdata["loras"] = [LoRAParams.from_dict(l) for l in cdata.get("loras", {})]
    return cls(**cdata)
