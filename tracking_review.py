"""Compact tracking overview and an interactive frame/identity inspector."""
import numpy as np
import matplotlib
from PIL import Image, ImageEnhance, ImageOps

from fit_bezier_curves import bezier_basis, node_available
from mesh_hints import missing_crossings

from node_editor import NODE_COLORS, NodeEditor
from session_data import node_geometry, load_ui_state, parse_fps, save_reference

COLORS = NODE_COLORS


def background(path):
    with Image.open(path) as source:
        return ImageEnhance.Brightness(ImageOps.exif_transpose(source).convert('RGB')).enhance(3)


def plot_frame(ax, frame, *, candidates=False, predictions=True, curves=True, labels=False, missing=True):
    ax.imshow(background(frame['source']))
    if curves:
        basis = bezier_basis(np.linspace(0, 1, 300))
        for curve in frame['curves']:
            points = basis@np.array(curve['control_points'])
            ax.plot(points[:, 0], points[:, 1], color='magenta' if curve['family'] == 'vertical' else 'cyan', lw=.7, alpha=.7)
    for status, color in COLORS.items():
        nodes = [n for n in frame['nodes'] if n['status'] == status and (predictions or node_available(n))]
        points = np.array([[n['x'], n['y']] for n in nodes]).reshape(-1, 2)
        ax.scatter(points[:, 0], points[:, 1], marker='o' if status == 'held' else '+' if status in ('verified', 'observed', 'recovered') else 'x', c=color, s=28, clip_on=True)
        if labels:
            for n in nodes:
                ax.text(n['x']+3, n['y']-3, str(n['id']), color=color, fontsize=7)
    if candidates:
        points = np.array([[n['x'], n['y']] for n in frame['unclaimed_detections']]).reshape(-1, 2)
        ax.scatter(points[:, 0], points[:, 1], marker='o', facecolors='none', edgecolors='violet', s=40)
    if missing:
        hints = missing_crossings([n for n in frame['nodes'] if node_available(n)], frame['image_size'], frame.get('width', 8))
        points = np.array([[h['x'], h['y']] for h in hints]).reshape(-1, 2)
        ax.scatter(points[:, 0], points[:, 1], s=650, facecolors='none', edgecolors='yellow', linewidths=2)
    ax.set_title(f"Frame {frame['frame']}: {sum(n['measured_xy'] is not None for n in frame['nodes'])}/{len(frame['nodes'])} observed")
    ax.set_axis_off()


@matplotlib.rc_context({'text.usetex': False, 'font.family': 'DejaVu Sans'})
def save_overview(report, path):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    frames = report['frames']
    sampled = [frames[i] for i in np.unique(np.linspace(0, len(frames)-1, min(20, len(frames)), dtype=int))]
    columns = min(5, len(sampled))
    rows = int(np.ceil(len(sampled)/columns))
    fig = Figure(figsize=(4*columns, 2.5*rows+2), layout='constrained')
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(rows+1, columns, height_ratios=[1]*rows+[.7])
    for i, frame in enumerate(sampled):
        plot_frame(fig.add_subplot(grid[i//columns, i % columns]), frame)
    ax = fig.add_subplot(grid[-1, :])
    for status, color in COLORS.items():
        ax.plot([f['frame'] for f in frames], [f['counts'].get(status, 0) for f in frames], '-o', label=status, color='gray' if status == 'verified' else color)
    ax.set(xlabel='Frame', ylabel='Number of nodes', title='Missing interiors are excluded from curve fits; boundary endpoints hold their last identified positions')
    ax.legend(ncol=6)
    fig.savefig(path, dpi=130)


class TrackingReview:
    OPTIONS = dict(Candidates='candidates', **{'Held / lost': 'predictions'},
                   Curves='curves', Labels='labels', Missing='missing')

    def __init__(self, report, session=None, figure=None):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from plot_widgets import Button, CheckButtons, Slider, TextBox
        from help_panel import HelpPanel
        self.session, self._report, self.editor = session, report, None
        self.mode, self.calibration_panel = 'review', None
        saved_ui = load_ui_state(session.output, len(report['frames'])) if session is not None else dict(frame=0, mode='review')
        self.state = dict(frame=saved_ui['frame'], selected=None, selected_candidate=None,
                          candidates=False, predictions=True, curves=True, labels=False, missing=True)
        self.fig = figure if figure is not None else plt.figure(figsize=(16, 9))
        self.fig.clear()
        self.fig.set_facecolor('white')
        self.ax = self.fig.subplots()
        self.fig.canvas.manager.set_window_title('GridTracker — edit and review frames')
        self.fig.subplots_adjust(left=.04, bottom=.24, right=.79, top=.90)
        self.info = self.fig.text(.025, .025, 'Select a node to inspect it. Click Help for controls and explanations.', fontsize=9, color='#222222')
        self.message = self.fig.text(.30, .095, '', fontsize=9, color='#222222')
        self.slider = Slider(self.fig.add_axes([.12, .17, .62, .03]), 'Frame', 0,
                             max(1, len(report['frames'])-1), valinit=saved_ui['frame'], valstep=1)
        self.fps_box = TextBox(self.fig.add_axes([.13, .085, .10, .035]), 'Frames / second ',
                               initial='' if report.get('fps') is None else str(report['fps']))
        self.fps_box.label.set_color('#222222')
        self.fps_box.text_disp.set_color('#222222')
        legend_ax = self.fig.add_axes([.805, .57, .19, .37], facecolor='#171717')
        legend_ax.set(xticks=[], yticks=[])
        names = dict(verified='Human verified', observed='Observed', recovered='Recovered',
                     held='Held edge', predicted='Held interior', lost='Lost', outside='Outside image', draft='Edited draft')
        handles = [Line2D([], [], color=color, marker='o' if status == 'held' else '+' if status in ('verified', 'observed', 'recovered', 'draft') else 'x',
                          linestyle='none', markersize=10, label=names[status]) for status, color in COLORS.items()]
        handles += [Line2D([], [], color=color, marker='o', fillstyle='none', linestyle='none', markersize=12, label=label)
                    for color, label in (('yellow', 'Possibly missing'), ('violet', 'Unclaimed candidate'), ('red', 'Conflicting identity'))]
        self.legend = legend_ax.legend(handles=handles, loc='center left', frameon=False, labelcolor='linecolor', fontsize=10)
        self.checks = CheckButtons(self.fig.add_axes([.815, .37, .18, .18], facecolor='#f5f5f5'),
                                  list(self.OPTIONS), [False, True, True, False, True])
        for label in self.checks.labels:
            label.set_color('#222222')
        for widget in (self.slider.label, self.slider.valtext):
            widget.set_color('#222222')
        self.buttons = {}
        for name, y, callback in (('Help', .315, self.toggle_help), ('Confirm frame', .265, self.confirm),
                                  ('Reanalyze', .215, self.reanalyze), ('Reanalyze next 10', .165, self.reanalyze_next), ('Export CSV', .115, self.export),
                                  ('Calibrate length', .065, self.calibrate), ('Unmark frame', .015, self.unmark)):
            button = Button(self.fig.add_axes([.825, y, .16, .04]), name)
            if name == 'Reanalyze':
                button.label.set_text('Reanalyze all')
            button.label.set_color('#222222')
            button.on_clicked(callback)
            self.buttons[name] = button
        self.help = HelpPanel(self.fig)
        self.curve_artists, self._fit_geometry = [], None
        self.slider.on_changed(self.draw)
        self.checks.on_clicked(self.toggle)
        self.fig.canvas.mpl_connect('button_press_event', self.click)
        self.fig.canvas.mpl_connect('key_press_event', self.key)
        self.draw(restore_fps=True)

    @property
    def report(self):
        return self.session.report if self.session is not None else self._report

    def say(self, text):
        self.message.set_text(text)
        self.fig.canvas.draw_idle()

    def detach_editor(self):
        if self.editor is not None:
            self.editor.close()
            self.editor.dispose()
            self.editor = None

    def draw(self, value=None, *, restore_fps=False):
        if self.mode != 'review':
            return
        index = min(int(self.slider.val), len(self.report['frames'])-1)
        if self.editor is not None and self.editor.frame == index:
            self.apply_options()
            return
        limits = (self.ax.get_xlim(), self.ax.get_ylim()) if self.ax.images else None
        self.detach_editor()
        self.state['frame'] = index
        self.ax.clear()
        self.ax.imshow(background(self.report['frames'][index]['source']))
        self.ax.set_axis_off()
        self.curve_artists, self._fit_geometry = [], None
        if self.session is not None:
            roster = self.session.roster()
            self.editor = NodeEditor(self.session.results[index], self.session.output/'reviews'/f'frame_{index:02d}.json',
                                     self.fig, self.ax, self.fps_box,
                                     nodes=self.report['frames'][index]['nodes'], roster=roster,
                                     next_id=max([n['id'] for n in roster]+[0])+1, fps=self.session.fps,
                                     frame=index, output=self.session.output, review=True, restore_fps=restore_fps)
            self.editor.on_redraw = self.refresh
            self.apply_options()
        else:
            plot_frame(self.ax, self.report['frames'][index], **{k: self.state[k] for k in ('candidates', 'predictions', 'curves', 'labels', 'missing')})
        if limits is not None:
            self.ax.set_xlim(limits[0])
            self.ax.set_ylim(limits[1])
        self.fig.canvas.draw_idle()

    def apply_options(self):
        if self.editor is None:
            return
        for key in ('candidates', 'predictions', 'labels', 'missing'):
            setattr(self.editor, 'show_'+key, self.state[key])
        self.editor.active = not self.help.visible
        self.editor.redraw()

    def refresh(self, editor):
        from fit_bezier_curves import fit_labeled_curves
        frame = self.report['frames'][editor.frame]
        geometry = node_geometry(editor.nodes)
        if self._fit_geometry != geometry:
            if editor.modified:
                self.preview_curves, self.preview_ignored = fit_labeled_curves(editor.nodes, frame['image_size'], frame['width'])
            else:
                self.preview_curves, self.preview_ignored = frame['curves'], frame['ignored_curves']
            self._fit_geometry = geometry
        for artist in self.curve_artists:
            artist.remove()
        self.curve_artists = []
        if self.state['curves']:
            basis = bezier_basis(np.linspace(0, 1, 300))
            for curve in self.preview_curves:
                points = basis@np.array(curve['control_points'])
                self.curve_artists += self.ax.plot(points[:, 0], points[:, 1], color='magenta' if curve['family'] == 'vertical' else 'cyan', lw=.8, alpha=.7)
        confirmed = editor.frame in self.session.anchors
        title = 'UNCONFIRMED EDITS (autosaved)' if editor.modified else 'HUMAN CONFIRMED' if confirmed else 'Needs review'
        self.ax.set_title(self.ax.get_title()+' | '+title, color='#222222')
        count = len(self.session.anchors)
        self.fig.suptitle(f'{count}/{len(self.report["frames"])} frames confirmed'
                          +(' | Corrections need reanalysis' if self.session.dirty else ''), color='#222222')
        if editor.selected is not None:
            node = editor.nodes[editor.selected]
            self.state['selected'], self.state['selected_candidate'] = node['id'], None
            self.info.set_text(f"ID {node['id']} | row {node['row']}, column {node['column']} | {node.get('status')} | source: {node.get('measurement_source')}\n"
                               f"Velocity ({self.report['velocity_unit']}): {node.get('velocity')} | acceleration ({self.report['acceleration_unit']}): {node.get('acceleration')}")
        else:
            self.state['selected'] = None
            self.info.set_text('Nodes are always editable. Drafts autosave; Confirm frame fixes a human reference.')
        for name, enabled in (('Export CSV', True), ('Unmark frame', editor.frame != 0 and confirmed)):
            self.buttons[name].active = enabled
            self.buttons[name].label.set_color('#222222' if enabled else 'gray')

    def toggle(self, label):
        if self.mode != 'review':
            return
        key = self.OPTIONS[label]
        self.state[key] = not self.state[key]
        self.apply_options()

    def toggle_help(self, event=None):
        if self.mode != 'review':
            return
        self.fps_box.stop_typing()
        self.help.toggle()
        self.buttons['Help'].label.set_text('Close help' if self.help.visible else 'Help')
        if self.editor is not None:
            self.editor.active = not self.help.visible

    def advance(self):
        index = self.state['frame']
        remaining = sorted(set(range(len(self.report['frames'])))-self.session.anchors.keys())
        if remaining:
            self.slider.set_val(next((i for i in remaining if i > index), remaining[0]))
        self.draw()

    def confirm(self, event=None):
        if self.session is None or self.mode != 'review':
            return
        index = self.state['frame']
        nodes = [n for n in self.editor.nodes if node_available(n)]
        try:
            fps = parse_fps(self.fps_box.text)
            self.session.validate_edit(index, nodes)
            document = save_reference(self.session.output/'reviews'/f'frame_{index:02d}.json', self.session.results[index], nodes, True, fps)
            self.session.confirm(index, document)
        except ValueError as error:
            self.say(str(error))
            return
        save_reference(self.editor.draft_path, self.session.results[index], document['nodes'], False, fps,
                       editor_state=dict(modified=False, fps_text=self.fps_box.text))
        self.editor.dispose()
        self.editor = None
        self.say(f'Frame {index} confirmed. Reanalyze to propagate corrections.')
        self.advance()

    def reanalyze_next(self, event=None):
        first = self.state['frame']+1
        last = min(first+9, len(self.report['frames'])-1)
        if first > last:
            self.say('There are no frames after the selected frame.')
            return
        self.reanalyze(frame_range=(first, last))

    def reanalyze(self, event=None, *, frame_range=None):
        from plot_widgets import paused_widgets
        if self.session is None or self.mode != 'review':
            return
        fps = parse_fps(self.fps_box.text)
        if frame_range is not None and fps != self.report.get('fps'):
            self.say('Use Reanalyze all after changing the frame rate.')
            return
        self.detach_editor()
        self.session.fps = fps
        self.say('Reanalyzing unconfirmed frames from fixed human anchors…')
        self.mode = 'processing'
        def progress(done, total):
            if done % 10 == 0 or done == total:
                self.say(f'Reanalyzing frames: {done}/{total}')
                self.fig.canvas.flush_events()
        try:
            with paused_widgets([self.slider, self.checks, self.fps_box, *self.buttons.values()]):
                self.fig.canvas.flush_events()
                self.session.reanalyze(progress=progress, **({'frame_range': frame_range} if frame_range is not None else {}))
        except Exception as error:
            self.say(f'Reanalysis failed: {error}')
            raise
        finally:
            self.mode = 'review'
            self.draw()
        self.say('Reanalysis complete. Confirmed coordinates stay fixed.' if frame_range is None else
                 f'Reanalyzed frames {frame_range[0]}–{frame_range[1]}; all other frames unchanged.')

    def unmark(self, event=None):
        if self.session is None or self.mode != 'review':
            return
        index = self.state['frame']
        self.detach_editor()
        try:
            changed = self.session.unmark(index)
        except ValueError as error:
            self.say(str(error))
            self.draw()
            return
        if changed:
            self.reanalyze()
            self.say(f'Frame {index} is automatic again; former edits are in reviews/history.')
        else:
            self.draw()

    def export(self, event=None):
        if self.session is None or self.mode != 'review':
            return
        self.editor.autosave()
        try:
            paths = self.session.export_csv(fps=parse_fps(self.fps_box.text), drafts=self.session.saved_drafts())
        except ValueError as error:
            self.say(str(error))
            return
        self.say(f'Exported fitted crossings, curves, and candidates to {paths[0].parent}')

    def calibrate(self, event=None):
        if self.session is None or self.mode != 'review':
            return
        from calibration import CalibrationPanel, load_calibration
        self.editor.autosave()
        self.editor.active = False
        self.fps_box.stop_typing()
        self.mode = 'calibration'
        def done(document):
            self.mode, self.calibration_panel = 'review', None
            self.editor.active = not self.help.visible
            self.session.calibration = load_calibration(self.session.output)
            self.session.save()
            if document is not None:
                self.say(f"Calibration saved: {document['units_per_pixel']:.6g} {document['unit']}/pixel")
            self.draw()
        self.calibration_panel = CalibrationPanel(self.fig, self.session.output, done)

    def click(self, event):
        if self.mode != 'review' or event.inaxes != self.ax or not self.state['candidates'] or self.editor is None:
            return
        if self.editor.nearest(event) is not None:
            return
        candidates = self.report['frames'][self.state['frame']]['unclaimed_detections']
        if candidates:
            distance = np.linalg.norm(self.ax.transData.transform([[n['x'], n['y']] for n in candidates])-[event.x, event.y], axis=1)
            if distance.min() < 15:
                node = candidates[int(np.argmin(distance))]
                self.state['selected_candidate'] = node.get('candidate_id')
                self.say(f"Unclaimed candidate {node.get('candidate_id', node['id'])}: {node.get('reason')}. Add it manually to accept it.")

    def key(self, event):
        if self.mode != 'review' or self.fps_box.capturekeystrokes:
            return
        if event.key in ('left', 'right'):
            self.slider.set_val(np.clip(self.slider.val+(-1 if event.key == 'left' else 1), 0, len(self.report['frames'])-1))
        elif event.key in ('f1', '?'):
            self.toggle_help()
        elif event.key in ('i', 'h'):
            self.checks.set_active(list(self.OPTIONS).index('Labels' if event.key == 'i' else 'Candidates'))


def show_review(report, backend='TkAgg', session=None, figure=None):
    from reference_editor import configure_backend
    configure_backend(backend)
    import matplotlib.pyplot as plt
    plt.rcParams.update({'text.usetex': False, 'font.family': 'DejaVu Sans'})
    viewer = TrackingReview(report, session, figure=figure)
    plt.show()
    return viewer


