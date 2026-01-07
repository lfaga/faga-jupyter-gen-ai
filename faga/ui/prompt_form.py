import re
import ipywidgets as widgets
from IPython.core.display import HTML
from typing import Any, Callable, ContextManager
from dataclasses import asdict
from contextlib import nullcontext
from faga.ai.file_manager import FileManager
from faga.ai.model_params import ModelParams
from faga.ai.factory import ModelParamsFactory
from faga.ai.models import Model, ModelType
from faga.ui.loras_selector import LoraSelector
from faga.ui.embed_selector import EmbeddingSelector
from faga.ui.custom_combo import CustomCombo


class PromptForm:

  def __init__(
    self,
    file_manager: FileManager,
    callback_generate: None | Callable[[str, ModelParams], None],
    message_output: widgets.Output | ContextManager[Any] = nullcontext()
  ):
    self._fm: FileManager = file_manager
    self._callback_generate: None | Callable[[str, ModelParams], None] = callback_generate
    self._model_id: str = ""
    self._preset_name: str = ""
    self._model_params: ModelParams | None = None
    self._header_controls: dict[str, Any] = {}
    self._body_controls: list = []
    self._msg_out: widgets.Output | ContextManager[Any] = message_output
    self._header_container: widgets.VBox = widgets.VBox()
    self._body_container: widgets.VBox = widgets.VBox()
    self._header_container.add_class("box_container")
    self._body_container.add_class("box_container")
    self._is_refreshing: bool = False
    self._generate_form()
    self._refresh_form()

  @property
  def model_id(self) -> str:
    return self._model_id

  @model_id.setter
  def model_id(self, value: str):
    new_value = value or ""
    if self._model_id != new_value and self._fm.is_valid_model_id(new_value):
      was_refeshing: bool = self._is_refreshing
      self._is_refreshing = True

      prev_model = self._fm.get_model(self.model_id)
      self._model_id = new_value
      model = self._fm.get_model(self.model_id)
      if model and (not prev_model or prev_model.type != model.type):
        self._model_params = ModelParamsFactory.create_new(model.type, self._model_id)
      if not was_refeshing:
        self._refresh_form()
        self._is_refreshing = False

  @property
  def preset_name(self) -> str:
    return self._preset_name

  @preset_name.setter
  def preset_name(self, value: str):
    new_value = value or ""
    if self._preset_name != new_value and self._fm.is_valid_preset_name(new_value):
      was_refeshing: bool = self._is_refreshing
      self._is_refreshing = True
      self._preset_name = new_value
      preset = self._fm.get_preset(self._preset_name)
      if preset and self._fm.is_valid_model_id(preset.model):
        sm: Model | None = self._fm.get_model(self.model_id)
        if not sm or sm.type != preset.model_type:
          self.model_id = preset.model
        self._model_params = preset
      if not was_refeshing:
        self._refresh_form()
        self._is_refreshing = False

  def on_generate(self, callback_generate: Callable[[str, ModelParams], None]):
    self._callback_generate = callback_generate

  def _on_selected_model_change(self, change: dict[str, Any]):
    ctl: widgets.Dropdown = self._header_controls["model"]
    ctl.unobserve(self._on_selected_model_change, type="change", names="value")
    if isinstance(change['new'], str):
      self.model_id = str(change['new'])
    ctl.observe(self._on_selected_model_change, type="change", names="value")

  def _on_preset_change(self, change: dict[str, Any]):
    ctl: CustomCombo = self._header_controls["preset"]
    ctl.on_change(None)
    if isinstance(change['new'], str):
      self.preset_name = str(change['new'])
    ctl.on_change(self._on_preset_change)

  def _harvest_data(self) -> ModelParams | None:
    if self._fm.is_valid_model_id(self.model_id):
      values = {}
      for control in self._body_controls:
        if hasattr(control, "data_param_name") and not getattr(control, "data_param_ignore", True):
          values[getattr(control, "data_param_name")] = control.value

      if model := self._fm.get_model(self.model_id):
        return ModelParamsFactory.create_from_values(model.type, self.model_id, values)
    return None

  def _btn_execute_on_click(self, b: widgets.Button):
    if self._callback_generate:
      mp = self._harvest_data()
      if mp:
        self._callback_generate(self.model_id, mp)
      else:
        raise ValueError("Execute: Cannot harvest data")

  def _btn_defaults_on_click(self, b: widgets.Button):
    if self._fm.is_valid_model_id(self.model_id):
      if (model := self._fm.get_model(self.model_id)):
        self._model_params = ModelParamsFactory.create_new(model.type, self.model_id)
        self._refresh_form()

  def _btn_save_preset_on_click(self, b: widgets.Button):

    mp = self._harvest_data()
    if mp:
      p_name: str = self._header_controls["preset"].value
      if p_name and not re.search(r"[^\w\-\.]", p_name):
        self._fm.save_preset(p_name, mp)
        self._header_controls["preset"].options = self._get_select_preset_options()
      else:
        with self._msg_out:
          print("Save preset: Invalid preset name. (Allowed characters: a-z A-Z 0-9 - _ .)")
    else:
      with self._msg_out:
        print("Save preset: Cannot harvest data")

  def _append_control(self, controls: list, param_name: str, widget: Any, full_row: bool = False):
    widget.data_param_name = param_name
    widget.data_full_row = full_row
    controls.append(widget)
    return controls, widget

  def _generate_form(self):

    self._header_controls["model"] = widgets.Dropdown(
      description='Model:',
      disabled=False,
      continuous_update=False,
      layout=widgets.Layout(flex="1 0 auto", width="auto")
    )
    self._header_controls["model"].observe(self._on_selected_model_change, type="change", names="value")

    self._header_controls["preset"] = CustomCombo(
      placeholder="[Select a preset or type a new name]",
      description='Presets:',
      ensure_option=False,
      disabled=False,
      continuous_update=False,
      hidden_values=["_hidden_"],
      layout=widgets.Layout(flex="1 0 auto", width="auto")
    )
    self._header_controls["preset"].on_change(self._on_preset_change)

    self._header_controls["save_preset"] = widgets.Button(description='Save Preset', disabled=False)
    self._header_controls["save_preset"].add_class("control")
    self._header_controls["save_preset"].on_click(self._btn_save_preset_on_click)

    self._header_controls["defaults"] = widgets.Button(description='Defaults', disabled=False)
    self._header_controls["defaults"].add_class("control")
    self._header_controls["defaults"].on_click(self._btn_defaults_on_click)

    self._header_controls["execute"] = widgets.Button(description='Generate', disabled=False)
    self._header_controls["execute"].add_class("control")
    self._header_controls["execute"].on_click(self._btn_execute_on_click)

    buttons_box: widgets.HBox = widgets.HBox(
      [self._header_controls["save_preset"], self._header_controls["defaults"], self._header_controls["execute"]],
      layout=widgets.Layout(flex="0 0 auto", align_self="flex-end")
    )
    header_box: widgets.VBox = widgets.VBox(
      [self._header_controls["model"], self._header_controls["preset"], buttons_box]
    )
    header_box.add_class("box")
    self._header_container.children = [header_box]

    body_controls: list = []
    body_controls, _ = self._append_control(
      body_controls, "prompt",
      widgets.Textarea(description='Prompt:', disabled=False, rows=5, layout=widgets.Layout(flex="1 0 auto"))
    )
    body_controls, _ = self._append_control(
      body_controls, "negative",
      widgets.Textarea(description='Negative:', disabled=False, rows=5, layout=widgets.Layout(flex="1 0 auto"))
    )
    body_controls, _ = self._append_control(
      body_controls, "tag_prompt",
      widgets.Textarea(description='Tag prompt:', disabled=False, rows=5, layout=widgets.Layout(flex="1 0 auto"))
    )
    body_controls, _ = self._append_control(
      body_controls, "width",
      widgets.IntSlider(
        min=512,
        max=2048,
        step=64,
        description="Width:",
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        layout=widgets.Layout(flex="1 0 auto")
      )
    )
    body_controls, _ = self._append_control(
      body_controls, "height",
      widgets.IntSlider(
        min=512,
        max=2048,
        step=64,
        description="Height:",
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        layout=widgets.Layout(flex="1 0 auto")
      )
    )
    body_controls, _ = self._append_control(
      body_controls, "steps",
      widgets.IntSlider(
        min=1,
        max=100,
        step=1,
        description="Steps:",
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        layout=widgets.Layout(flex="1 0 auto")
      )
    )
    body_controls, _ = self._append_control(
      body_controls, "guidance",
      widgets.FloatSlider(
        min=1.0,
        max=10.0,
        step=0.1,
        description="Guidance:",
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        layout=widgets.Layout(flex="1 0 auto")
      )
    )
    body_controls, _ = self._append_control(
      body_controls, "cfg",
      widgets.FloatSlider(
        min=1.0,
        max=10.0,
        step=0.1,
        description="Cfg:",
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        layout=widgets.Layout(flex="1 0 auto")
      )
    )
    body_controls, _ = self._append_control(
      body_controls, "sampler",
      widgets.Dropdown(
        options=[
          "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2", "exp_heun_2_x0",
          "exp_heun_2_x0_sde", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral",
          "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde",
          "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm",
          "lcm", "ipndm", "ipndm_v", "deis", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral",
          "res_multistep_ancestral_cfg_pp", "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde", "seeds_2",
          "seeds_3", "sa_solver", "sa_solver_pece"
        ],
        description='Sampler:',
        disabled=False,
        layout=widgets.Layout(flex="1 0 auto")
      )
    )
    body_controls, _ = self._append_control(
      body_controls, "scheduler",
      widgets.Dropdown(
        options=[
          "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal", "linear_quadratic",
          "kl_optimal"
        ],
        description='Scheduler:',
        disabled=False,
        layout=widgets.Layout(flex="1 0 auto")
      )
    )
    body_controls, _ = self._append_control(
      body_controls, "hires_fix",
      widgets.Checkbox(description='Hi-Res fix', disabled=False, indent=True, layout=widgets.Layout(flex="1 0 auto"))
    )
    #need to make a selector for different upscale models
    body_controls, _ = self._append_control(
      body_controls, "upscale_model",
      widgets.Text(description='Upscale:', disabled=False, layout=widgets.Layout(flex="1 0 auto"))
    )
    body_controls, _ = self._append_control(
      body_controls, "batch_size",
      widgets.IntText(description="Batch size", disabled=False, layout=widgets.Layout(flex="1 0 auto"))
    )
    body_controls, _ = self._append_control(
      body_controls, "seed",
      widgets.BoundedIntText(
        description="Seed:", min=0, max=65535, step=1, disabled=False, layout=widgets.Layout(flex="1 0 auto")
      )
    )
    body_controls, _ = self._append_control(
      body_controls, "clip_skip",
      widgets.BoundedIntText(
        description="Clip Skip:", min=1, max=24, step=1, disabled=False, layout=widgets.Layout(flex="1 0 auto")
      )
    )
    body_controls, _ = self._append_control(
      body_controls, "loras", LoraSelector(fileManager=self._fm, layout=widgets.Layout(flex="1 0 auto", padding="2px")),
      True
    )
    body_controls, _ = self._append_control(
      body_controls, "embeddings",
      EmbeddingSelector(fileManager=self._fm, layout=widgets.Layout(flex="1 0 auto", padding="2px")), True
    )
    self._body_controls = body_controls

  def _get_select_preset_options(self) -> list[tuple[str, str]]:
    m: Model | None = self._fm.get_model(self.model_id)
    values: list[tuple[str, str]] = [(n, n) for n in self._fm.get_presets(model_type=m.type if m else None)]
    values.insert(0, ("_hidden_", "_hidden_"))
    return values

  def _refresh_form(self):
    for ctl_name, control in self._header_controls.items():
      match ctl_name:
        case "model":
          model_options = [
            (m[1].checkpoint.name if m[1].checkpoint else (m[1].unet.name if m[1].unet else "No name"), m[0])
            for m in self._fm.get_models().items()
          ]
          model_options.insert(0, ("[Select a model]", " "))
          control.options = model_options
          control.value = self.model_id if self._fm.is_valid_model_id(self.model_id) else " "

        case "preset":
          control.options = self._get_select_preset_options()
          mt: ModelType | None = m.type if (m := self._fm.get_model(self.model_id)) else None
          control.value = self.preset_name if mt and self._fm.is_valid_preset_for_type(
            mt, self.preset_name
          ) else "_hidden_"

        case "execute" | "save_preset":
          control.layout = widgets.Layout(
            visibility="visible" if self._fm.is_valid_model_id(self.model_id) else "hidden"
          )

    dict_params = asdict(self._model_params) if self._model_params else {}
    visible_controls = [
      c for c in self._body_controls
      if getattr(c, "data_include", False) or getattr(c, "data_param_name", "") in dict_params
    ]
    for control in self._body_controls:
      control.data_param_ignore = not control in visible_controls

    for control in visible_controls:
      param_name = getattr(control, "data_param_name", "")
      if hasattr(control, "model_type") and self._fm.is_valid_model_id(self.model_id):
        if model := self._fm.get_model(self.model_id):
          setattr(control, "model_type", model.type)
      value = dict_params.get(param_name)
      if control.value != value:
        control.value = value

    rows: list[widgets.HBox] = []
    children = []
    vccnt = len(visible_controls)
    for i in range(0, vccnt, 1):
      children.append(visible_controls[i])
      if len(children) == 2 or getattr(
        visible_controls[i], "data_full_row", False
      ) or (i < vccnt - 1 and getattr(visible_controls[i + 1], "data_full_row", False)) or (i == vccnt - 1):
        row: widgets.HBox = widgets.HBox()
        row.add_class("box")
        row.children = children
        rows.append(row)
        children = []

    self._body_container.children = tuple(rows)

  def get_header_control(self):
    return self._header_container

  def get_body_control(self):
    return self._body_container

  def get_style(self):
    return HTML(
      """<style>
:root {
  --darker-color: #202020;
  --dark-color: #303030;
  --medium-color: #646464;
  --light-color: #B0B0B0;
  --lighter-color: #E0E0E0;
  --accent-color: #60D0C0;
  --title-color: #B0D0C0;
  --icon-color: #2060A0;
  --font-family: Verdana, Geneva, Tahoma, sans-serif;
  --font-size: 11pt;
}

html,
body,
.box_container,
.box {
  background: var(--darker-color) !important;
  border: none !important;
  color: var(--light-color);
  font-family: var(--font-family);
  font-size: var(--font-size);
  margin: 0;
  padding: 0;
}

.box_container {
  display: block !important;
}

.box {
  display: flex !important;
  border: 1px solid var(--medium-color) !important;
  margin: 0;
  padding: 4px !important;
  flex: 0 1 auto;
}

div:has(.box_container) {
  display: block !important;
  background: var(--darker-color) !important;
  border: none !important;
}

label,
div,
span {
  color: var(--light-color) !important;
}

input,
button,
select,
textarea,
.control {
  font-family: var(--font-family);
  background: var(--dark-color) !important;
  color: var(--light-color) !important;
  resize: none;
  border: 1px solid var(--medium-color) !important;
}

.control:hover,
button:hover {
  border: 1px solid var(--accent-color) !important;
  color: var(--accent-color) !important;
}

option[value='_hidden_'] {
  display: none !important;
}
</style>"""
    )
