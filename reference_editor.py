"""Human approval of the first frame; importing this module never opens a GUI."""
from node_editor import NodeEditor
from PIL import Image, ImageEnhance, ImageOps
from session_data import parse_fps, save_reference, save_ui_state


def configure_backend(name):
    if name.lower() == 'tkagg':
        from gui_support import configure_tk
        configure_tk()
    import matplotlib
    matplotlib.use(name)


class ReferenceEditor(NodeEditor):
    def __init__(self, result, path, poll=None, *, nodes=None, roster=None, next_id=None,
                 fps=None, on_approve=None, validate_callback=None, frame=0,
                 figure=None, axes=None, on_cancel=None, on_export=None, on_export_video=None, output=None, keep_open=False):
        import matplotlib.pyplot as plt
        from plot_widgets import Button, TextBox
        from help_panel import HelpPanel
        self.embedded, self.keep_open = figure is not None, keep_open
        self.on_cancel, self.on_export = on_cancel, on_export
        self.on_export_video = on_export_video
        self.on_approve, self.validate_callback = on_approve, validate_callback
        self.calibration_panel = None
        fig, ax = (figure, axes) if self.embedded else plt.subplots(figsize=(15, 8))
        self.previous = [(a, a.get_visible()) for a in list(fig.axes)+list(fig.texts) if a is not ax]
        self.previous_layout = {k: getattr(fig.subplotpars, k) for k in ('left', 'right', 'top', 'bottom', 'wspace', 'hspace')}
        for artist, _ in self.previous:
            artist.set_visible(False)
        ax.clear()
        fig.canvas.manager.set_window_title(f'GridTracker — verify first frame')
        fig.subplots_adjust(bottom=.17, top=.90)
        with Image.open(result['source']) as source_image:
            image = ImageOps.exif_transpose(source_image).convert('RGB')
        ax.imshow(ImageEnhance.Brightness(image).enhance(3))
        ax.set_axis_off()
        self.status = fig.text(.02, .075, 'Edits autosave. Confirm the first frame to initialize tracking.', color='#222222')
        fps_box = TextBox(fig.add_axes([.13, .01, .10, .035]), 'Frames / second ', initial='' if fps is None else str(fps))
        self.help = HelpPanel(fig)
        self.help_button = Button(fig.add_axes([.04, .115, .08, .035]), 'Help')
        self.help_button.on_clicked(self.toggle_help)
        self.calibration_button = Button(fig.add_axes([.28, .01, .12, .04]), 'Calibrate length')
        self.calibration_button.on_clicked(self.calibrate)
        self.export_button = Button(fig.add_axes([.42, .01, .11, .04]), 'Export CSV')
        self.export_button.on_clicked(self.export)
        self.video_button = Button(fig.add_axes([.55, .01, .11, .04]), 'Export MP4')
        self.video_button.on_clicked(self.export_video)
        self.cancel_button = Button(fig.add_axes([.68, .01, .10, .04]), 'Cancel')
        self.cancel_button.on_clicked(self.cancel)
        self.approve_button = Button(fig.add_axes([.80, .01, .18, .04]), 'Approve and track' if on_approve is None else 'Confirm frame')
        self.approve_button.on_clicked(self.approve)
        self.own_axes = [a for a in fig.axes if a is not ax and a not in [p[0] for p in self.previous]]
        self.own_texts = [t for t in fig.texts if t not in [p[0] for p in self.previous]]
        super().__init__(result, path, fig, ax, fps_box, nodes=nodes, roster=roster, next_id=next_id,
                         fps=fps, frame=frame, output=output, poll=poll)

    def toggle_help(self, event=None):
        self.help.toggle()
        self.active = not self.help.visible

    def approve(self, event):
        import matplotlib.pyplot as plt
        if not self.active:
            return
        try:
            self.fps = parse_fps(self.fps_box.text)
            if self.validate_callback is not None:
                self.validate_callback(self.nodes)
            document = save_reference(self.path, self.result, self.nodes, approved=True, fps=self.fps)
        except ValueError as error:
            self.status.set_text(str(error))
            self.fig.canvas.draw_idle()
            return
        self.approved = document
        save_reference(self.draft_path, self.result, self.nodes, approved=False, fps=self.fps)
        save_ui_state(self.output, self.frame, 'review')
        self.dispose()
        if self.on_approve is not None:
            self.on_approve(document)
        if self.keep_open:
            self.fig.canvas.stop_event_loop()
        elif not self.embedded:
            plt.close(self.fig)

    def dispose(self):
        from plot_widgets import disconnect_widgets
        super().dispose()
        disconnect_widgets((self.approve_button, self.cancel_button, self.export_button, self.video_button, self.calibration_button, self.help_button, self.fps_box))
        for artist in self.own_axes+self.own_texts:
            artist.remove()
        for artist, visible in self.previous:
            artist.set_visible(visible)
        self.fig.subplots_adjust(**self.previous_layout)

    def calibrate(self, event=None):
        if not self.active:
            return
        from calibration import CalibrationPanel
        self.fps_box.stop_typing()
        self.active = False
        def done(document):
            self.active, self.calibration_panel = True, None
            if document is not None:
                self.status.set_text(f"Calibration saved: {document['units_per_pixel']:.6g} {document['unit']}/pixel")
            self.fig.canvas.draw_idle()
        self.calibration_panel = CalibrationPanel(self.fig, self.output, done)

    def export(self, event=None):
        if not self.active or self.on_export is None:
            return
        try:
            self.fps = parse_fps(self.fps_box.text)
            paths = self.on_export(self.nodes, self.fps)
            self.status.set_text(f'CSV exported to {paths[0].parent}')
        except ValueError as error:
            self.status.set_text(str(error))
        self.fig.canvas.draw_idle()

    def export_video(self, event=None):
        if not self.active or self.on_export_video is None:
            return
        try:
            self.fps = parse_fps(self.fps_box.text)
            self.autosave()
            path = self.on_export_video(self.nodes, self.fps)
            self.status.set_text(f'MP4 exported from completed frames to {path} (30 fps playback)')
        except ValueError as error:
            self.status.set_text(str(error))
        self.fig.canvas.draw_idle()

    def cancel(self, event=None):
        import matplotlib.pyplot as plt
        self.close(None)
        self.dispose()
        if self.on_cancel is not None:
            self.on_cancel()
        save_ui_state(self.output, self.frame, 'review')
        if self.keep_open:
            self.fig.canvas.stop_event_loop()
        elif not self.embedded:
            plt.close(self.fig)

    def close(self, event):
        self.timer.stop()
        if self.approved is None:
            self.autosave()
        if self.keep_open:
            self.fig.canvas.stop_event_loop()

    def show(self, block=True):
        import matplotlib.pyplot as plt
        if self.keep_open and block:
            plt.show(block=False)
            self.fig.canvas.start_event_loop(0)
        else:
            plt.show(block=block)
        if block:
            self.timer.stop()
        return self.approved
