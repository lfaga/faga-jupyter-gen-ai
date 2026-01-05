import torch, platform
from faga.ai.models import Model, ModelType
from faga.ai.model_params import ModelParams
from faga.ai.extras import LoRA, Embedding
from faga.ai.file_manager import FileManager
from faga.ai.download_manager import DownloadManager


class Workflow:

  @staticmethod
  def generate(file_manager: FileManager, model_id: str, params: ModelParams, dest_path: str) -> torch.Tensor | None:
    if platform.system() == "Windows":
      print("Not supported on Windows")
      raise Exception("Do not run locally!")

    if not (model := file_manager.get_model(model_id)):
      raise Exception(f"Model {model_id} not found.")

    dm = DownloadManager(dest_path)
    if not dm.download_model(model):
      raise Exception("Cannot download model")

    match model.type:
      case ModelType.SD15:
        from faga.ai.workflows.sd15_workflow import SD15_Workflow
        from faga.ai.model_types.sd15 import T2I_SD15_Params

        if isinstance(params, T2I_SD15_Params):
          loras: list[LoRA] = []
          for lp in params.loras:
            if lora := file_manager.get_lora_from_param(lp):
              loras.append(lora)
            else:
              raise Exception(f"The LoRAs {lp.name} is no longer available in extras.json")

          embeddings: list[Embedding] = []
          for emb in params.embeddings:
            if embedding := file_manager.get_embedding_from_param(emb):
              embeddings.append(embedding)
            else:
              raise Exception(f"The Embedding {emb.name} is no longer available in extras.json")

          if dm.download_loras(loras) and dm.download_embeddings(embeddings):
            return SD15_Workflow.generate(file_manager, params)
          else:
            #figure out later how to give more information
            raise Exception("Cannot download all the associated files")
      case _:
        return None
    return None
