import os, sys, time
import platform, subprocess
import ipywidgets as widgets
from typing import ContextManager, Any
from contextlib import nullcontext
from IPython.display import display
from faga.ai.models import (
  Model,
  Architecture,
  TensorFile,
)
from faga.ai.extras import (
  LoRA,
  Embedding,
)

fake_gdrive_path: str = "D:\\Projects\\Python\\fake-gdrive"
if platform.system() == "Windows" and not fake_gdrive_path in sys.path:
  sys.path.insert(0, fake_gdrive_path)

from google.colab import drive


class DownloadManager:

  def __init__(self, comfyui_root: str, msg_output: widgets.Output | ContextManager[Any] = nullcontext()):
    self._is_windows = platform.system() == "Windows"
    self._aria_exe = "aria2c.exe" if self._is_windows else "aria2c"
    self._comfyui_root = comfyui_root
    self._msg_out: widgets.Output | ContextManager[Any] = msg_output

  def download_model(self, model: Model) -> bool:
    if not os.path.exists(self._comfyui_root):
      with self._msg_out:
        print(f"Error: comfyui_root ('{self._comfyui_root}') doesn't exist")
      return False

    if model.architecture == Architecture.CHECKPOINT:
      if not self._download_tensorfile(model.checkpoint, "checkpoints"):
        return False

    elif model.architecture == Architecture.MODULAR:

      if model.unet:
        if not self._download_tensorfile(model.unet, "unet"):
          return False

      if model.text_encoder:
        if not self._download_tensorfile(model.text_encoder, "text_encoders"):
          return False

      if model.clip_l:
        if not self._download_tensorfile(model.clip_l, "text_encoders"):
          return False

    else:
      with self._msg_out:
        print(f"No valid architecture selected.")
      return False

    if model.vae:
      if not self._download_tensorfile(model.vae, "vae"):
        return False

    return True

  def download_loras(self, loras: list[LoRA]) -> bool:

    for lora in loras:
      if not self._download_file(lora.name, lora.url, "loras"):
        with self._msg_out:
          print(f"Failed to get LoRA {lora.name} from {lora.url}")
        return False
    return True

  def download_embeddings(self, embeddings: list[Embedding]) -> bool:

    for embedding in embeddings:
      if not self._download_file(embedding.name, embedding.url, "embeddings"):
        with self._msg_out:
          print(f"Failed to get embedding {embedding.name} from {embedding.url}")
        return False
    return True

  def _download_tensorfile(self, file: TensorFile | None, dest_folder: str) -> bool:
    if file:
      return self._download_file(file.name, file.url, dest_folder)
    return False

  def _download_file(self, file_name: str, file_url: str, dest_folder: str) -> bool:
    if self._is_windows:
      with self._msg_out:
        print("Not supported on Windows")
      return False

    dest_path = os.path.join(self._comfyui_root, "models", dest_folder)
    file_full_path = os.path.join(dest_path, file_name)
    exe: str = ""

    if not os.path.exists(dest_path):
      os.makedirs(dest_path, exist_ok=True)

    msg: str

    if file_url.startswith("/content/drive/MyDrive/"):
      if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')

      exe = "ln"
      cmd = [exe, "-sf", file_url, file_full_path]

      with self._msg_out:
        print(f"Linking {file_url}")

    else:

      if os.path.exists(file_full_path) and not os.path.exists(f"{file_full_path}.aria2"):
        return True

      exe = self._aria_exe
      cmd = [
        exe, "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "-q", "-d", dest_path, "-o",
        file_name, file_url
      ]
      with self._msg_out:
        print(f"Downloading {file_url}")

    try:
      process = subprocess.Popen(cmd)
      idx: int = 0
      anim: str = "🕐🕒🕔🕖🕘🕚"
      lbl: widgets.Label = widgets.Label()
      with self._msg_out:
        display(lbl)
        while (rc := process.poll()) is None:
          if isinstance(self._msg_out, widgets.Output):
            lbl.value = anim[idx % 6]
          else:
            print(f"{anim[idx % 6]}", end="")
          idx += 1
          time.sleep(0.5)
        print()
        lbl.close()

      if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)

    except FileNotFoundError:
      print(f"\n{exe} not found in system PATH")
      return False
    except subprocess.CalledProcessError as e:
      print(f"\n{exe} {file_name} failed, exit code: {e.returncode}")
      return False

    return os.path.exists(file_full_path)

    #try:
    #  with self._msg_out:
    #    subprocess.run(cmd, check=True, capture_output=True)
    #except FileNotFoundError:
    #  with self._msg_out:
    #    print(f"{exe} not found in system PATH")
    #  return False
    #except subprocess.CalledProcessError as e:
    #  with self._msg_out:
    #    print(f"{exe} {file_name} failed, exit code: {e.returncode}")
