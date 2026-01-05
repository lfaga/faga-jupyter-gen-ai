import ipywidgets as widgets
from faga.ai.file_manager import FileManager
from faga.ai.models import ModelType
from faga.ai.model_params import EmbeddingParams
from faga.ui.selectors import Selector


class EmbeddingSelector(widgets.HBox):

  def __init__(self, fileManager: FileManager, **kwargs):
    super().__init__(**kwargs)
    self._fileManager: FileManager = fileManager
    self._embeds_available: dict[str, EmbeddingParams] = {}
    self._embeds_selected: dict[str, EmbeddingParams] = {}
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
  def value(self) -> list[EmbeddingParams]:
    return [l for _, l in self._embeds_selected.items()]

  @value.setter
  def value(self, value: list[EmbeddingParams]):
    self._embeds_selected = {}
    if value:
      for ep in value:
        if isinstance(ep, dict):
          ep = EmbeddingParams(**ep)

        if ep.name in self._embeds_available:
          self._embeds_selected[ep.name] = self._embeds_available.pop(ep.name)
        else:
          print(f"The Embedding {ep.name} is no longer available.")
    self._refresh_control()

  @property
  def model_type(self) -> ModelType | None:
    return self._model_type

  @model_type.setter
  def model_type(self, value: ModelType | None):
    self._model_type = value
    self._embeds_available = {
      k: EmbeddingParams(e.name)
      for k, e in self._fileManager.get_embeddings(self._model_type).items()
    }
    self._embeds_selected = {}
    self._refresh_control()

  def _btn_move_handler(self, embed_name, direction: Selector.Direction, row_control):
    if isinstance(row_control, widgets.HBox):

      origin: dict[str, EmbeddingParams]
      origin_ctrls: list
      origin_container: widgets.VBox
      destination: dict[str, EmbeddingParams]
      dest_ctrls: list
      dest_container: widgets.VBox

      if direction == Selector.Direction.Add:
        origin = self._embeds_available
        origin_ctrls = self._available_rows
        origin_container = self._available_rows_ctrl
        destination = self._embeds_selected
        dest_ctrls = self._selected_rows
        dest_container = self._selected_rows_ctrl
      else:
        origin = self._embeds_selected
        origin_ctrls = self._selected_rows
        origin_container = self._selected_rows_ctrl
        destination = self._embeds_available
        dest_ctrls = self._available_rows
        dest_container = self._available_rows_ctrl

      embed = origin.pop(embed_name)
      destination[embed_name] = embed

      dest_ctrls.append(
        self._get_row(
          embed_name, Selector.Direction.Add if direction == Selector.Direction.Remove else Selector.Direction.Remove
        )
      )
      dest_container.children = dest_ctrls

      origin_ctrls.remove(row_control)
      origin_container.children = origin_ctrls

  def _get_row(self, sel_name: str, direction: Selector.Direction) -> widgets.HBox:

    lbl = widgets.Label(value=sel_name)
    btn = widgets.Button(description=direction.value, disabled=False, layout=widgets.Layout(width="40px"))
    row: widgets.HBox
    if direction == Selector.Direction.Add:
      row = widgets.HBox([lbl, btn])
    else:
      row = widgets.HBox([btn, lbl])

    btn.on_click(
      lambda button, embed_name=sel_name, row_control=row: self._btn_move_handler(embed_name, direction, row_control)
    )
    return row

  def _refresh_control(self):
    self._available_rows = []
    for l in self._embeds_available:
      self._available_rows.append(self._get_row(l, Selector.Direction.Add))

    if self._available_rows_ctrl:
      self._available_rows_ctrl.children = self._available_rows

    self._selected_rows = []
    for l in self._embeds_selected:
      self._selected_rows.append(self._get_row(l, Selector.Direction.Remove))

    self._selected_rows_ctrl.children = self._selected_rows

    self.children = [self._available_rows_ctrl, self._selected_rows_ctrl]
