import ipywidgets as widgets
from typing import Any, Callable


class CustomCombo(widgets.VBox):

  def __init__(
    self,
    value: str = "",
    options: list[Any] = [],
    placeholder: str = "",
    on_change: None | Callable[[dict[str, Any]], None] = None,
    **kwargs
  ):
    super().__init__(**kwargs)

    self._value: str = value or ""
    self._options: list[Any] | None = options or []

    self._txtctl: widgets.Text = widgets.Text(
      placeholder=placeholder, layout=widgets.Layout(flex="1 0 auto", width="auto")
    )
    self._selctl: widgets.Select = widgets.Select(
      disabled=False, continuous_update=False, layout=widgets.Layout(flex="1 0 auto", width="auto")
    )
    self._selctl.value = None

    self.children = [self._txtctl, self._selctl]
    self.layout = widgets.Layout(width="auto", box_sizing="border-box", display="flex")

    self._txtctl.observe(self._on_txtctl_change, type="change", names="value")
    self._selctl.observe(self._on_selctl_change, type="change", names="value")
    self._on_change_callback: None | Callable[[dict[str, Any]], None] = on_change

    self._is_refreshing: bool = False

  def on_change(self, callback_function: None | Callable[[dict[str, Any]], None]):
    self._on_change_callback = callback_function

  @property
  def value(self) -> str:
    return self._value

  @value.setter
  def value(self, value: str):
    new_value = value or ""
    if self._value != new_value:
      was_refeshing: bool = self._is_refreshing
      self._is_refreshing = True

      self._value = new_value

      if not was_refeshing:
        self._refresh_controls()
        self._is_refreshing = False
      if self._on_change_callback:
        self._on_change_callback({"new": self._value})

  @property
  def options(self) -> list[Any] | None:
    return self._options

  @options.setter
  def options(self, value: list[Any] | None):
    new_options = value or []
    if self._options != new_options:
      was_refeshing: bool = self._is_refreshing
      self._is_refreshing = True

      self._options = new_options

      if not was_refeshing:
        self._refresh_controls()
        self._is_refreshing = False

  def _refresh_controls(self):
    if self._txtctl.value != self.value:
      self._txtctl.value = self.value
    self._selctl.value = None
    self._selctl.options = self._options or []
    if self.value in self._selctl.options:
      self._selctl.value = self.value or None

  def _on_txtctl_change(self, change: dict[str, Any]):
    self._txtctl.unobserve(self._on_txtctl_change, type="change", names="value")
    if isinstance(change['new'], str):
      new_value: str = str(change['new']) or ""
      if self.value != new_value:
        self.value = new_value
    self._txtctl.observe(self._on_txtctl_change, type="change", names="value")

  def _on_selctl_change(self, change: dict[str, Any]):
    self._selctl.unobserve(self._on_selctl_change, type="change", names="value")
    if isinstance(change['new'], str):
      new_value: str = str(change['new']) or ""
      if self.value != new_value:
        self.value = new_value
    self._selctl.observe(self._on_selctl_change, type="change", names="value")
