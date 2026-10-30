"""Matplotlib controls that cannot interact while their panel is hidden."""
from contextlib import contextmanager
from matplotlib import widgets


class _Visible:
    def ignore(self, event):
        return not self.ax.get_visible() or super().ignore(event)


class Button(_Visible, widgets.Button):
    pass


class Slider(_Visible, widgets.Slider):
    pass


class CheckButtons(_Visible, widgets.CheckButtons):
    pass


class RadioButtons(_Visible, widgets.RadioButtons):
    pass


class TextBox(_Visible, widgets.TextBox):
    pass


def release_input(widget):
    if isinstance(widget, widgets.TextBox):
        widget.stop_typing()
    if isinstance(widget, widgets.Slider):
        widget.drag_active = False
    canvas = widget.canvas
    if canvas.mouse_grabber is widget.ax:
        canvas.release_mouse(widget.ax)


def disconnect_widgets(controls):
    for widget in controls:
        release_input(widget)
        widget.active = False
        widget.disconnect_events()


@contextmanager
def paused_widgets(controls):
    states = [(w, w.get_active()) for w in controls]
    for widget, _ in states:
        release_input(widget)
        widget.active = False
    try:
        yield
    finally:
        for widget, active in states:
            widget.active = active
