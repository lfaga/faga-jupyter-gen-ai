import sys, math
import torch
from typing import Any
from faga.ai.file_manager import FileManager
from faga.ai.model_types.sd15 import T2I_SD15_Params
from faga.ai.models import Architecture, ClipType, WeightType

comfypath = "/content/ComfyUI"
if not comfypath in sys.path:
  sys.path.insert(0, comfypath)

from nodes import (
  CheckpointLoaderSimple,
  VAELoader,
  KSampler,
  LatentUpscale,
  LoraLoader,
  EmptyLatentImage,
  VAEDecode,
  CLIPTextEncode,
  UNETLoader,
  CLIPLoader,
)
from comfy_extras.nodes_upscale_model import (
  UpscaleModelLoader,
  ImageUpscaleWithModel,
)

from custom_nodes.ComfyUI_GGUF.nodes import (
  UnetLoaderGGUF,
  CLIPLoaderGGUF,
)


class SD15_Workflow:

  @staticmethod
  def generate(fileManager: FileManager, params: T2I_SD15_Params) -> torch.Tensor:

    model_config = fileManager.get_model(params.model)

    if not model_config:
      raise ValueError(f"Model not found: The key '{params.model}' does not exist in the models dictionary.")

    with torch.inference_mode():

      text_encoder: Any
      vae: Any = None

      if model_config.architecture == Architecture.CHECKPOINT and model_config.checkpoint:
        model, text_encoder, vae = CheckpointLoaderSimple().load_checkpoint(model_config.checkpoint.name)

      elif model_config.architecture == Architecture.MODULAR:
        if not model_config.unet or not model_config.text_encoder:
          raise Exception("Parts of the model are not loaded.")

        if model_config.unet.is_gguf:
          model, = UnetLoaderGGUF().load_unet(model_config.unet.name)

        else:
          model, = UNETLoader().load_unet(
            model_config.unet.name, (model_config.unet.weight_type or WeightType.fp8_e4m3fn).value
          )

        if model_config.text_encoder.is_gguf:
          text_encoder, = CLIPLoaderGGUF().load_clip(
            model_config.text_encoder.name, (model_config.text_encoder.type or ClipType.STABLE_DIFFUSION).value
          )

        else:
          text_encoder, = CLIPLoader().load_clip(
            model_config.text_encoder.name, (model_config.text_encoder.type or ClipType.STABLE_DIFFUSION).value
          )

      else:
        raise ValueError(f"No valid architecture selected.")

      text_encoder.clip_layer(-params.clip_skip)

      if model_config.vae:
        vae, = VAELoader().load_vae(model_config.vae.name)

      if not vae:
        raise Exception("Required VAE not provided.")

      patched_clip: Any = None
      patched_model: Any = None

      for lora_config in params.loras:
        if lora_config.strength > 0.0:
          patched_model, patched_clip = LoraLoader().load_lora(
            model,
            text_encoder,
            lora_config.name,
            strength_model=lora_config.strength,
            strength_clip=lora_config.strength  #TODO: i might want to have a separate value
          )
          #load_lora clones model and clip when strenght > 0.0
          del model, text_encoder
          model = patched_model
          text_encoder = patched_clip

      positive_cond, = CLIPTextEncode().encode(text_encoder, params.prompt)
      negative_cond, = CLIPTextEncode().encode(text_encoder, params.negative)
      del text_encoder

      final_width = params.width
      final_height = params.height
      final_steps = params.steps
      final_sampler = params.sampler
      final_scheduler = params.scheduler

      if params.hires_fix:
        aspect = final_width / final_height
        area = model_config.base_size_px**2
        raw_height = math.sqrt(area / aspect)
        raw_width = raw_height * aspect
        first_height = round(raw_height / 8.0) * 8
        first_width = round(raw_width / 8.0) * 8

        first_steps = round(final_steps * 0.4)
        final_steps = final_steps - first_steps
        first_sampler = "dpmpp_sde_gpu"
        first_scheduler = "normal"

      else:
        first_width = final_width
        first_height = final_height
        first_steps = final_steps
        first_sampler = final_sampler
        first_scheduler = final_scheduler

      empty_latent_img, = EmptyLatentImage().generate(first_width, first_height, params.batch_size)

      #print(f"First KSampler pass with empty latent at W:{first_width}xH:{first_height}")
      #print(f"with steps:{first_steps}")

      sampled_latent, = KSampler().sample(
        model,
        params.seed,
        first_steps,
        params.cfg,
        first_sampler,
        first_scheduler,
        positive_cond,
        negative_cond,
        empty_latent_img,
        denoise=1.0
      )

      if params.hires_fix:
        upscaled_latent, = LatentUpscale().upscale(
          sampled_latent, "nearest-exact", final_width, final_height, "disabled"
        )

        #print(f"Second KSampler pass with upscaled latent at W:{final_width}xH:{final_height}")
        #print(f"with steps:{final_steps}")

        final_latent, = KSampler().sample(
          model,
          params.seed,
          final_steps,
          params.cfg,
          final_sampler,
          final_scheduler,
          positive_cond,
          negative_cond,
          upscaled_latent,
          denoise=0.5
        )
        del upscaled_latent

      else:
        final_latent = sampled_latent

      del empty_latent_img
      del model, positive_cond, negative_cond
      del patched_model, patched_clip

      images, = VAEDecode().decode(vae, final_latent)

      del vae, sampled_latent, final_latent

      if params.upscale_model:
        #print(f"Upscaling with model: {upscale_model_name}")
        upscale_model, = UpscaleModelLoader().execute(params.upscale_model)
        upscaled_images, = ImageUpscaleWithModel().execute(upscale_model, images)
        del images, upscale_model
        images = upscaled_images

      return images
