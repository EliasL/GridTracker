"""One image-space length calibration per video, edited in the existing figure."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from session_data import write_json

UNITS = {'mm': 1e-3, 'cm': 1e-2, 'm': 1., 'µm': 1e-6, 'in': .0254}


def length_calibration(points, length, unit, image_size):
    points = np.asarray(points, dtype=float)
    length = float(length)
    if unit not in UNITS:
        raise ValueError(f'Unsupported length unit: {unit}')
    if points.shape != (2, 2) or not np.isfinite(points).all() or not np.isfinite(length) or length <= 0:
        raise ValueError('Choose two finite endpoints and a positive known length')
    if np.any(points < 0) or np.any(points > np.asarray(image_size)-1):
        raise ValueError('Calibration endpoints must lie inside the image')
    pixels = float(np.linalg.norm(points[1]-points[0]))
    if pixels <= 1e-6:
        raise ValueError('Calibration line has zero pixel length')
    return dict(points=points.tolist(), reference_length=length, unit=unit,
                image_size=list(image_size), pixel_length=pixels,
                units_per_pixel=length/pixels, meters_per_pixel=length*UNITS[unit]/pixels)


def load_calibration(output):
    path = Path(output)/'calibration.json'
    if not path.exists():
        return None
    document = json.loads(path.read_text())
    expected = length_calibration(document['points'], document['reference_length'], document['unit'], document['image_size'])
    if any(not np.isclose(expected[key], document[key], rtol=1e-12, atol=0) for key in ('units_per_pixel', 'meters_per_pixel', 'pixel_length')):
        raise ValueError('Saved calibration scale disagrees with its measured line')
    return document


class UnitDropdown:
    """A small Matplotlib dropdown, also testable with a noninteractive backend."""
    def __init__(self, fig, rectangle, value='mm'):
        from plot_widgets import Button, RadioButtons
        self.value = value
        self.button = Button(fig.add_axes(rectangle), value+' ▾')
        x, y, w, h = rectangle
        self.menu_ax = fig.add_axes([x, y+h, w, .20], zorder=20)
        self.menu = RadioButtons(self.menu_ax, list(UNITS), active=list(UNITS).index(value))
        self.menu_ax.set_visible(False)
        self.button.on_clicked(self.toggle)
        self.menu.on_clicked(self.select)
        self.fig = fig

    def toggle(self, event):
        self.menu_ax.set_visible(not self.menu_ax.get_visible())
        self.fig.canvas.draw_idle()

    def select(self, value):
        self.value = value
        self.button.label.set_text(value+' ▾')
        self.menu_ax.set_visible(False)
        self.fig.canvas.draw_idle()


class CalibrationPanel:
    def __init__(self, fig, output, on_done):
        from plot_widgets import Button, TextBox
        from message_panel import MessagePanel
        self.fig, self.output, self.on_done = fig, Path(output), on_done
        self.old = [(a, a.get_visible()) for a in list(fig.axes)+list(fig.texts)]
        for artist, _ in self.old:
            artist.set_visible(False)
        self.ax = fig.add_axes([.06, .24, .88, .68])
        self.ax.set_axis_off()
        self.messages = MessagePanel(fig, [.05, .005, .90, .03])
        self.status = self.messages.channel('message')
        self.status.set_text('Load a calibration image at the same magnification as the video. Drag endpoints or the whole line.')
        self.length = TextBox(fig.add_axes([.20, .12, .10, .04]), 'Known length ', initial='10')
        self.units = UnitDropdown(fig, [.34, .12, .10, .04])
        self.buttons = []
        for text, x, callback in (('Load image', .05, self.choose), ('Cancel', .66, self.cancel), ('Apply to video', .81, self.apply)):
            button = Button(fig.add_axes([x, .045, .14, .045]), text)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.points, self.image, self.image_path, self.drag = None, None, None, None
        self.cids = [fig.canvas.mpl_connect(event, callback) for event, callback in
                     (('button_press_event', self.press), ('motion_notify_event', self.move),
                      ('button_release_event', self.release), ('close_event', self.window_closed))]
        existing = load_calibration(output)
        initial = Path(existing['image']) if existing is not None else self.output/'calibration_input.png'
        if initial.exists():
            self.load_image(initial)
            if existing is not None:
                self.points = np.array(existing['points'])
                self.length.set_val(str(existing['reference_length']))
                self.units.select(existing['unit'])
                self.redraw()
        else:
            self.ax.text(.5, .5, 'Load an image, then mark a known distance', transform=self.ax.transAxes, ha='center')
        self.fig.canvas.draw_idle()

    def load_image(self, path):
        with Image.open(path) as source:
            self.image = ImageOps.exif_transpose(source).convert('RGB')
        self.image_path = Path(path)
        width, height = self.image.size
        self.points = np.array([[.3*(width-1), .5*(height-1)], [.7*(width-1), .5*(height-1)]])
        self.ax.clear()
        self.ax.imshow(self.image)
        self.ax.set_axis_off()
        self.line, = self.ax.plot(*self.points.T, '-o', color='yellow', lw=2, markersize=10)
        self.redraw()

    def choose(self, event=None):
        from tkinter.filedialog import askopenfilename
        path = askopenfilename(title='Choose a length-calibration image', filetypes=[('Images', '*.png *.jpg *.jpeg *.tif *.tiff *.bmp'), ('All files', '*')])
        if path:
            self.load_image(path)

    def redraw(self):
        self.line.set_data(*self.points.T)
        self.ax.set_title(f'Calibration: {np.linalg.norm(self.points[1]-self.points[0]):.2f} pixels')
        self.fig.canvas.draw_idle()

    def press(self, event):
        if self.points is None or event.inaxes != self.ax or event.button != 1 or getattr(self.fig.canvas.toolbar, 'mode', ''):
            return
        display = self.ax.transData.transform(self.points)
        cursor = np.array([event.x, event.y])
        distance = np.linalg.norm(display-cursor, axis=1)
        if distance.min() < 15:
            self.drag = (int(distance.argmin()), np.array([event.xdata, event.ydata]), self.points.copy())
        else:
            delta = display[1]-display[0]
            u = np.clip((cursor-display[0])@delta/max(delta@delta, 1e-12), 0, 1)
            if np.linalg.norm(cursor-(display[0]+u*delta)) < 10:
                self.drag = ('line', np.array([event.xdata, event.ydata]), self.points.copy())

    def move(self, event):
        if self.drag is None or event.inaxes != self.ax:
            return
        mode, start, original = self.drag
        xy = np.array([event.xdata, event.ydata])
        extent = np.array(self.image.size)-1
        if mode == 'line':
            shift = np.clip(xy-start, -original.min(axis=0), extent-original.max(axis=0))
            self.points = original+shift
        else:
            self.points[mode] = np.clip(xy, 0, extent)
        self.redraw()

    def release(self, event):
        self.drag = None

    def apply(self, event=None):
        if self.image is None:
            self.status.set_text('Load an image before applying calibration.')
            self.fig.canvas.draw_idle()
            return
        try:
            document = length_calibration(self.points, self.length.text, self.units.value, self.image.size)
        except ValueError as error:
            self.status.set_text(str(error))
            self.fig.canvas.draw_idle()
            return
        self.output.mkdir(parents=True, exist_ok=True)
        image_path = self.output/'calibration_image.png'
        self.image.save(image_path)
        document.update(image=str(image_path.resolve()), source_image=str(self.image_path.resolve()),
                        image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                        applied_at=datetime.now(timezone.utc).isoformat())
        write_json(self.output/'calibration.json', document)
        self.finish(document)

    def cancel(self, event=None):
        self.finish(None)

    def window_closed(self, event):
        self.finish(None, closing=True)

    def finish(self, document, closing=False):
        from plot_widgets import disconnect_widgets
        for cid in self.cids:
            self.fig.canvas.mpl_disconnect(cid)
        disconnect_widgets(self.buttons+[self.length, self.units.button, self.units.menu])
        old_axes = {a for a, _ in self.old if a in self.fig.axes}
        for axes in list(self.fig.axes):
            if axes not in old_axes:
                axes.remove()
        self.messages.close()
        for artist, visible in self.old:
            artist.set_visible(visible)
        if not closing:
            self.on_done(document)
            self.fig.canvas.draw_idle()
