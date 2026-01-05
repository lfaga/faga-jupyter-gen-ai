import os, json
from typing import Any
from dataclasses import asdict
from faga.ai.models import (
  Architecture,
  Model,
  ModelType,
  TensorFile,
  TensorFileType,
  ClipType,
  WeightType,
)
from faga.ai.model_params import (
  ModelParams,
  LoRAParams,
  EmbeddingParams,
)
from faga.ai.factory import ModelParamsFactory
from faga.ai.extras import (
  LoRA,
  Embedding,
)


class FileManager:

  def __init__(self, base_folder: str):

    self._base_folder: str = base_folder
    self._models: dict[str, Model] = {}
    self._loras: dict[str, LoRA] = {}
    self._embeddings: dict[str, Embedding] = {}
    self._presets: dict[str, ModelParams] = {}

    with os.scandir(os.path.join(self._base_folder, "models")) as models:
      for entry in models:
        if entry.is_file() and entry.name.endswith(".json"):

          with open(entry.path, "r") as file:
            data = json.load(file)
          if isinstance(data, dict):
            if model_data := self._json_to_model(data):
              name, model = model_data
              self._models[name] = model

    with open(os.path.join(self._base_folder, "extras.json"), "r") as file:
      extras = json.load(file)
    if isinstance(extras, dict):
      ll = extras.get("loras")
      if isinstance(ll, list):
        for entry in ll:
          if isinstance(entry, dict) and (lora := self._json_to_lora(entry)):
            self._loras[lora.name] = lora

      embs = extras.get("embeddings")
      if isinstance(embs, list):
        for entry in embs:
          if isinstance(entry, dict) and (emb := self._json_to_embedding(entry)):
            self._embeddings[emb.name] = emb

    with os.scandir(os.path.join(self._base_folder, "presets")) as presets:
      for entry in presets:
        if entry.is_file() and entry.name.endswith(".json"):
          with open(entry.path, "r") as file:
            data = json.load(file)
          if isinstance(data, dict) and (preset := self._json_to_preset(data)) and isinstance(preset, ModelParams):
            p_name, _ = os.path.splitext(entry.name)
            self._presets[p_name] = preset

  def _json_to_model(self, json_data: dict[str, Any]) -> tuple[str, Model] | None:
    for name, properties in json_data.items():
      checkpoint: TensorFile | None = self._json_to_tensor_file(properties.get(TensorFileType.checkpoint.value))
      unet: TensorFile | None = self._json_to_tensor_file(properties.get(TensorFileType.unet.value))
      text_encoder: TensorFile | None = self._json_to_tensor_file(properties.get(TensorFileType.text_encoder.value))
      clip_l: TensorFile | None = self._json_to_tensor_file(properties.get(TensorFileType.clip_l.value))
      vae: TensorFile | None = self._json_to_tensor_file(properties.get(TensorFileType.vae.value))
      base_size_px = int(properties.get("base_size_px", 512))

      try:
        return name, Model(
          architecture=Architecture(properties.get("architecture")),
          type=ModelType(properties.get("type")),
          checkpoint=checkpoint,
          unet=unet,
          text_encoder=text_encoder,
          clip_l=clip_l,
          vae=vae,
          base_size_px=base_size_px
        )

      except ValueError as e:
        print(f"Error: Invalid value: {e}")
    return None

  def _json_to_tensor_file(self, properties: dict[str, Any] | None) -> TensorFile | None:
    if properties:
      ct = properties.get("type")
      wt = properties.get("weight_type")

      return TensorFile(
        name=properties.get("name", ""),
        url=properties.get("url", ""),
        type=ClipType(ct) if ct else None,
        is_gguf=bool(properties.get("is_gguf", False)),
        weight_type=WeightType(wt) if wt else None,
      )
    return None

  def _json_to_lora(self, json: dict[str, Any]) -> LoRA | None:
    try:
      return LoRA(
        type=ModelType(json.get("type")),
        name=str(json.get("name", "")),
        url=str(json.get("url", "")),
        strength=float(json.get("strength", 1.0)),
      )
    except Exception:
      return None

  def _json_to_embedding(self, json: dict[str, Any]) -> Embedding | None:
    try:
      return Embedding(
        type=ModelType(json.get("type")),
        name=str(json.get("name", "")),
        url=str(json.get("url", "")),
      )
    except Exception:
      return None

  def _json_to_preset(self, json_data: dict[str, Any]) -> ModelParams | None:
    m = json_data.get("model")
    if m and m in self._models:
      model = self._models[m]
      return ModelParamsFactory.create_from_values(model.type, m, json_data)
    return None

  def get_models(self) -> dict[str, Model]:
    return self._models

  def get_model(self, model_id: str) -> Model | None:
    return self._models.get(model_id)

  def is_valid_model_id(self, model_id: str) -> bool:
    return model_id in self._models

  def get_loras(self, model_type: ModelType | None) -> dict[str, LoRA]:
    return {k: l for k, l in self._loras.items() if not model_type or l.type == model_type}

  def get_lora(self, name: str) -> LoRA | None:
    return self._loras.get(name)

  def get_lora_from_param(self, param: LoRAParams) -> LoRA | None:
    if lora := self._loras.get(param.name):
      clora = lora.clone()
      clora.strength = param.strength
      return clora
    return None

  def get_embeddings(self, model_type: ModelType | None) -> dict[str, Embedding]:
    return {k: e for k, e in self._embeddings.items() if not model_type or e.type == model_type}

  def get_embedding(self, name: str) -> Embedding | None:
    return self._embeddings.get(name)

  def get_embedding_from_param(self, param: EmbeddingParams) -> Embedding | None:
    if emb := self._embeddings.get(param.name):
      return emb.clone()
    return None

  def get_presets(self, model_id: str | None = None) -> dict[str, ModelParams]:
    return {n: p for n, p in self._presets.items() if not model_id or p.model == model_id}

  def get_preset(self, name: str) -> ModelParams | None:
    return self._presets.get(name)

  def is_valid_preset_name(self, name: str) -> bool:
    return name in self._presets

  def is_valid_preset_for_model(self, model_id: str, name: str) -> bool:
    return name in self.get_presets(model_id)

  def save_preset(self, name: str, preset: ModelParams) -> bool:

    try:
      new_name = name
      p_folder = os.path.join(self._base_folder, "presets")
      out_file = os.path.join(p_folder, f"{new_name}.json")
      idx: int = 1

      while os.path.exists(out_file):
        new_name = f"{name}_{idx}"
        out_file = os.path.join(p_folder, f"{new_name}.json")
        idx += 1

      with open(out_file, 'w') as f:
        json.dump(asdict(preset), f, indent=2)

      self._presets[new_name] = preset
      return True
    except Exception:
      print(f"save preset '{name}' failed.")
      return False
