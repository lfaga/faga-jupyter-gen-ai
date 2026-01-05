from typing import Any
from faga.ai.model_params import ModelType


class ModelParamsFactory:

  @staticmethod
  def create_new(model_type: ModelType, model_name: str):

    match model_type:
      case ModelType.SD15:
        from faga.ai.model_types.sd15 import T2I_SD15_Params
        return T2I_SD15_Params.create_new(model_name=model_name)
      case ModelType.HUNYUAN_DIT:
        from faga.ai.model_types.hunyuan_dit import T2I_HunyuanDit_Params
        return T2I_HunyuanDit_Params.create_new(model_name=model_name)
      case ModelType.LUMINA2:
        from faga.ai.model_types.lumina2 import T2I_Lumina2_Params
        return T2I_Lumina2_Params.create_new(model_name=model_name)
      case ModelType.FLUX1S:
        from faga.ai.model_types.flux1s import T2I_Flux1s_Params
        return T2I_Flux1s_Params.create_new(model_name=model_name)
      case _:
        return None

  @staticmethod
  def create_from_values(model_type: ModelType, model_name: str, data: dict[str, Any]):

    match model_type:
      case ModelType.SD15:
        from faga.ai.model_types.sd15 import T2I_SD15_Params
        return T2I_SD15_Params.create_from_values(model_name=model_name, data=data)
      case ModelType.HUNYUAN_DIT:
        from faga.ai.model_types.hunyuan_dit import T2I_HunyuanDit_Params
        return T2I_HunyuanDit_Params.create_from_values(model_name=model_name, data=data)
      case ModelType.LUMINA2:
        from faga.ai.model_types.lumina2 import T2I_Lumina2_Params
        return T2I_Lumina2_Params.create_from_values(model_name=model_name, data=data)
      case ModelType.FLUX1S:
        from faga.ai.model_types.flux1s import T2I_Flux1s_Params
        return T2I_Flux1s_Params.create_from_values(model_name=model_name, data=data)
      case _:
        return None
