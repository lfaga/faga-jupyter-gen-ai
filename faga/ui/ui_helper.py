import sys, os, platform
import gc, torch
import ipywidgets as widgets
from contextlib import nullcontext
from shutil import copy2
from datetime import datetime
from typing import Any, ContextManager
from dataclasses import asdict
from IPython.core.display import Image as IpdImage

fake_gdrive_path: str = "D:\\Projects\\Python\\fake-gdrive"
if platform.system() == "Windows" and not fake_gdrive_path in sys.path:
  sys.path.insert(0, fake_gdrive_path)

from google.colab import drive


class UiHelper:

  def __init__(
    self, tensor_images: Any, generation_params: Any, msg_out: widgets.Output | ContextManager[Any] = nullcontext()
  ):
    self._saved_images: dict[str, str] = {}
    self._generation_params = asdict(generation_params)
    self._msg_out: widgets.Output | ContextManager[Any] = msg_out
    self._tensor_images_to_images(tensor_images)

  def _tensor_images_to_images(self, tensor_images: torch.Tensor):
    if platform.system() == "Windows":
      raise Exception("Not Implemented in Windows")

    comfypath = "/content/ComfyUI"
    if not comfypath in sys.path:
      sys.path.insert(0, comfypath)
    from nodes import SaveImage

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-3]
    prefix = f"{timestamp}-"

    with torch.inference_mode():

      cpu_images = tensor_images.detach().cpu()
      saved_images = SaveImage().save_images(cpu_images, prefix, extra_pnginfo=self._generation_params)
      if ui := saved_images.get("ui"):
        if results := ui.get("images"):
          for result in results:
            r_path = os.path.join("/content/ComfyUI/output", result.get("subfolder"), result.get("filename"))
            self._saved_images[result.get("filename")] = r_path
      del cpu_images, tensor_images

  def copy_to_gdrive(self, gdrive_output_path: str) -> bool:
    try:
      if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')

      if not os.path.exists(gdrive_output_path):
        os.makedirs(gdrive_output_path, exist_ok=True)

      for key, img_path in self._saved_images.items():
        with self._msg_out:
          print(f"Copying image {img_path} to {gdrive_output_path}")
        copy2(img_path, os.path.join(gdrive_output_path, key))
      return True

    except:
      return False

  def get_images(self) -> list[IpdImage]:

    result: list[IpdImage] = []

    for saved_image in self._saved_images.values():
      result.append(IpdImage(filename=saved_image))

    return result

  def get_generation_params(self) -> dict[str, Any]:
    return self._generation_params

  @staticmethod
  def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
      torch.cuda.ipc_collect()
    for obj in list(globals().values()):
      if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
        del obj
    gc.collect()

  @staticmethod
  def force_reimports():
    to_delete = [m for m in sys.modules if m.startswith('faga')]
    for m in to_delete:
      del sys.modules[m]
    print(f"Cleaned {len(to_delete)} modules. Next import will read from Drive.")
