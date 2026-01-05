import ipywidgets as widgets
from faga.ai.file_manager import FileManager
from faga.ai.models import ModelType
from faga.ai.model_params import LoRAParams
from faga.ui.selectors import Selector


class LoraSelector(widgets.HBox):

  def __init__(self, fileManager: FileManager, **kwargs):
    super().__init__(**kwargs)
    self._fileManager: FileManager = fileManager
    self._loras_available: dict[str, LoRAParams] = {}
    self._loras_selected: dict[str, LoRAParams] = {}
    self._model_type: ModelType | None = None
    self._available_rows: list = []
    self._selected_rows: list = []
    self._available_rows_ctrl: widgets.VBox = widgets.VBox(
      layout=widgets.
      Layout(height="100%", flex="1 0", align_items="flex-end", border="1px solid silver", padding="8px")
    )
    self._selected_rows_ctrl: widgets.VBox = widgets.VBox(
      layout=widgets.
      Layout(height="100%", flex="1 0", align_items="flex-start", border="1px solid silver", padding="8px")
    )

  @property
  def value(self) -> list[LoRAParams]:
    return [l for _, l in self._loras_selected.items()]

  @value.setter
  def value(self, value: list[LoRAParams]):
    self._loras_selected = {}
    if value:
      for lp in value:
        if isinstance(lp, dict):
          lp = LoRAParams(**lp)

        if lp.name in self._loras_available:
          self._loras_selected[lp.name] = self._loras_available.pop(lp.name)
          self._loras_selected[lp.name].strength = lp.strength
        else:
          print(f"The LoRA {lp.name} is no longer available.")
    self._refresh_control()

  @property
  def model_type(self) -> ModelType | None:
    return self._model_type

  @model_type.setter
  def model_type(self, value: ModelType | None):
    self._model_type = value
    self._loras_available = {
      k: LoRAParams(l.name, l.strength)
      for k, l in self._fileManager.get_loras(self._model_type).items()
    }
    self._loras_selected = {}
    self._refresh_control()

  def _fxt_on_change(self, change, lora_name):
    if isinstance(change["new"], (float, int)):
      if lora_name in self._loras_available:
        self._loras_available[lora_name].strength = float(change["new"])
      elif lora_name in self._loras_selected:
        self._loras_selected[lora_name].strength = float(change["new"])

  def _btn_move_handler(self, lora_name, direction: Selector.Direction, strength_control, row_control):
    if isinstance(strength_control, widgets.BoundedFloatText) and isinstance(row_control, widgets.HBox):

      origin: dict[str, LoRAParams]
      origin_ctrls: list
      origin_container: widgets.VBox
      destination: dict[str, LoRAParams]
      dest_ctrls: list
      dest_container: widgets.VBox

      if direction == Selector.Direction.Add:
        origin = self._loras_available
        origin_ctrls = self._available_rows
        origin_container = self._available_rows_ctrl
        destination = self._loras_selected
        dest_ctrls = self._selected_rows
        dest_container = self._selected_rows_ctrl
      else:
        origin = self._loras_selected
        origin_ctrls = self._selected_rows
        origin_container = self._selected_rows_ctrl
        destination = self._loras_available
        dest_ctrls = self._available_rows
        dest_container = self._available_rows_ctrl

      lora = origin.pop(lora_name)
      destination[lora_name] = lora

      dest_ctrls.append(
        self._get_row(
          lora_name, Selector.Direction.Add if direction == Selector.Direction.Remove else Selector.Direction.Remove
        )
      )

      dest_container.children = dest_ctrls

      origin_ctrls.remove(row_control)
      origin_container.children = origin_ctrls

  def _get_row(self, sel_name: str, direction: Selector.Direction) -> widgets.HBox:

    loras: dict[str, LoRAParams]
    if direction == Selector.Direction.Add:
      loras = self._loras_available
    else:
      loras = self._loras_selected

    lbl = widgets.Label(value=sel_name)
    fxt: widgets.BoundedFloatText = widgets.BoundedFloatText(
      value=loras[sel_name].strength, min=0, max=5.0, step=0.01, disabled=False, layout=widgets.Layout(width="60px")
    )
    btn: widgets.Button = widgets.Button(
      description=direction.value, disabled=False, layout=widgets.Layout(width="40px")
    )

    row: widgets.HBox
    if direction == Selector.Direction.Add:
      row = widgets.HBox([lbl, fxt, btn])
    else:
      row = widgets.HBox([btn, lbl, fxt])

    fxt.observe(
      (lambda change, lora_name=sel_name: self._fxt_on_change(change, lora_name)), type="change", names="value"
    )

    btn.on_click(
      lambda button, lora_name=sel_name, strength_control=fxt, row_control=row: self.
      _btn_move_handler(lora_name, direction, strength_control, row_control)
    )
    return row

  def _refresh_control(self):
    self._available_rows = []
    for l in self._loras_available:
      self._available_rows.append(self._get_row(l, Selector.Direction.Add))

    self._available_rows_ctrl.children = self._available_rows

    self._selected_rows = []
    for l in self._loras_selected:
      self._selected_rows.append(self._get_row(l, Selector.Direction.Remove))

    self._selected_rows_ctrl.children = self._selected_rows

    self.children = [self._available_rows_ctrl, self._selected_rows_ctrl]
