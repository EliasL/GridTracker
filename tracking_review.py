"""Compact tracking overview and an interactive frame/identity inspector."""
import numpy as np
import matplotlib
from PIL import Image, ImageEnhance, ImageOps

from fit_bezier_curves import bezier_basis, node_available, grid_crossings
from mesh_hints import missing_crossings

from node_editor import NODE_COLORS, NodeEditor, node_marker
from session_data import node_geometry, load_ui_state, parse_fps
from message_panel import MessagePanel

COLORS = NODE_COLORS


def background(path):
    with Image.open(path) as source:
        return ImageEnhance.Brightness(ImageOps.exif_transpose(source).convert('RGB')).enhance(3)


def plot_frame(ax, frame, *, candidates=False, predictions=True, curves=True, labels=False, missing=True, guides=True, crossings=False):
    ax.imshow(background(frame['source']))
    if curves:
        basis = bezier_basis(np.linspace(0, 1, 300))
        for curve in frame['curves']:
            points = basis@np.array(curve['control_points'])
            ax.plot(points[:, 0], points[:, 1], color='magenta' if curve['family'] == 'vertical' else 'cyan', lw=.7, alpha=.7)
    for status, color in COLORS.items():
        nodes = [n for n in frame['nodes'] if guides and n['status'] == status and (predictions or node_available(n))]
        points = np.array([[n['x'], n['y']] for n in nodes]).reshape(-1, 2)
        ax.scatter(points[:, 0], points[:, 1], marker=node_marker(dict(status=status)), c=color, s=28, clip_on=True)
        if labels:
            for n in nodes:
                ax.text(n['x']+3, n['y']-3, str(n['id']), color=color, fontsize=7)
    if crossings:
        plot_crossings(ax, grid_crossings(frame['curves'], frame['image_size']), labels)
    if candidates:
        points = np.array([[n['x'], n['y']] for n in frame['unclaimed_detections']]).reshape(-1, 2)
        ax.scatter(points[:, 0], points[:, 1], marker='o', facecolors='none', edgecolors='yellow', s=40)
    if missing:
        hints = missing_crossings([n for n in frame['nodes'] if node_available(n)], frame['image_size'], frame.get('width', 8))
        points = np.array([[h['x'], h['y']] for h in hints]).reshape(-1, 2)
        ax.scatter(points[:, 0], points[:, 1], s=650, facecolors='none', edgecolors='yellow', linewidths=2)
    ax.set_title(f"Frame {frame['frame']}: {sum(n['measured_xy'] is not None for n in frame['nodes'])}/{len(frame['nodes'])} observed")
    ax.set_axis_off()


def plot_crossings(ax, nodes, labels=False):
    nodes = [n for n in nodes if n['status'] == 'curve_crossing']
    points = np.array([[n['x'], n['y']] for n in nodes]).reshape(-1, 2)
    artists = [ax.scatter(points[:, 0], points[:, 1], s=28, c='white', edgecolors='black', linewidths=.5, zorder=6)]
    if labels:
        artists += [ax.text(n['x']+3, n['y']-3, f"{n['row']},{n['column']}", color='white', fontsize=7) for n in nodes]
    return artists


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
    for name, statuses, color in (('Tracked / confirmed', ('verified', 'observed', 'recovered'), 'lime'),
                                   ('Edited', ('draft',), 'deepskyblue'),
                                   ('Held / missing', ('held', 'predicted', 'lost', 'outside'), 'orange')):
        ax.plot([f['frame'] for f in frames], [sum(f['counts'].get(s, 0) for s in statuses) for f in frames],
                '-o', label=name, color=color)
    ax.set(xlabel='Frame', ylabel='Number of nodes', title='Missing interiors are excluded from curve fits; boundary endpoints hold their last identified positions')
    ax.legend(ncol=6)
    fig.savefig(path, dpi=130)


class TrackingReview:
    OPTIONS = {'Guide nodes': 'guides', 'Grid crossings': 'crossings', 'Candidates': 'candidates',
               'Held / missing': 'predictions', 'Curves': 'curves', 'Labels': 'labels', 'Missing': 'missing'}

    def __init__(self, report, session=None, figure=None):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from plot_widgets import Button, CheckButtons, Slider, TextBox
        from help_panel import HelpPanel
        self.session, self._report, self.editor = session, report, None
        self.mode, self.calibration_panel = 'review', None
        saved_ui = load_ui_state(session.output, len(report['frames'])) if session is not None else dict(frame=0, mode='review')
        self.state = dict(frame=saved_ui['frame'], selected=None, selected_candidate=None,
                          candidates=False, predictions=True, curves=True, labels=False, missing=True, guides=True, crossings=False)
        self.fig = figure if figure is not None else plt.figure(figsize=(16, 9))
        self.fig.clear()
        self.fig.set_facecolor('white')
        self.ax = self.fig.subplots()
        self.fig.canvas.manager.set_window_title('GridTracker — edit and review frames')
        self.fig.subplots_adjust(left=.04, bottom=.24, right=.79, top=.90)
        self.messages = MessagePanel(self.fig, [.025, .005, .75, .075])
        self.info, self.message = self.messages.channel('info'), self.messages.channel('message')
        self.info.set_text('Select a guide node to inspect it. Status text can be selected and copied.')
        self.slider = Slider(self.fig.add_axes([.12, .17, .62, .03]), 'Frame', 0,
                             max(1, len(report['frames'])-1), valinit=saved_ui['frame'], valstep=1)
        self.fps_box = TextBox(self.fig.add_axes([.13, .085, .10, .035]), 'Frames / second ',
                               initial='' if report.get('fps') is None else str(report['fps']))
        self.fps_box.label.set_color('#222222')
        self.fps_box.text_disp.set_color('#222222')
        legend_ax = self.fig.add_axes([.805, .71, .19, .25], facecolor='#171717')
        legend_ax.set(xticks=[], yticks=[])
        handles = [Line2D([], [], color=color, marker=marker, linestyle='none', markersize=9,
                          markerfacecolor='none' if label in ('Suggestions', 'Conflicts') else color, label=label)
                   for color, marker, label in (('lime', '+', 'Tracked / confirmed'), ('deepskyblue', '+', 'Edited'),
                                                ('orange', 'x', 'Held / missing'), ('white', 'o', 'Grid crossings'),
                                                ('yellow', 'o', 'Suggestions'), ('red', 'o', 'Conflicts'))]
        self.legend = legend_ax.legend(handles=handles, loc='center left', frameon=False, labelcolor='linecolor', fontsize=10)
        self.checks = CheckButtons(self.fig.add_axes([.815, .405, .18, .28], facecolor='#f5f5f5'),
                                  list(self.OPTIONS), [self.state[key] for key in self.OPTIONS.values()])
        for label in self.checks.labels:
            label.set_color('#222222')
        for widget in (self.slider.label, self.slider.valtext):
            widget.set_color('#222222')
        self.buttons = {}
        for name, y, callback in (('Help', .355, self.toggle_help), ('Reanalyze', .255, self.reanalyze), ('Reanalyze next 10', .205, self.reanalyze_next),
                                  ('Export CSV', .155, self.export), ('Export MP4', .105, self.export_video),
                                  ('Calibrate length', .055, self.calibrate), ('Reset edits', .005, self.unmark)):
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

    def say(self, text, path=None):
        self.messages.set(text, path=path)
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
        self.ax.set_axis_off()
        self.curve_artists, self._fit_geometry = [], None
        if self.session is not None:
            self.ax.imshow(background(self.report['frames'][index]['source']))
            roster = self.session.roster()
            self.editor = NodeEditor(self.session.results[index], self.session.output/'reviews'/f'frame_{index:02d}.json',
                                     self.fig, self.ax, self.fps_box,
                                     nodes=self.report['frames'][index]['nodes'], roster=roster,
                                     next_id=max([n['id'] for n in roster]+[0])+1, fps=self.session.fps,
                                     frame=index, output=self.session.output, review=True, restore_fps=restore_fps, on_edit=self.edited)
            self.editor.on_redraw = self.refresh
            self.editor.commit_edits()
            self.apply_options()
        else:
            plot_frame(self.ax, self.report['frames'][index], **{k: self.state[k] for k in tuple(self.OPTIONS.values())})
        if limits is not None:
            self.ax.set_xlim(limits[0])
            self.ax.set_ylim(limits[1])
        self.fig.canvas.draw_idle()

    def apply_options(self):
        if self.editor is None:
            return
        for key in ('candidates', 'predictions', 'labels', 'missing', 'guides'):
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
            self.preview_crossings = grid_crossings(self.preview_curves, frame['image_size'])
            self._fit_geometry = geometry
        for artist in self.curve_artists:
            artist.remove()
        self.curve_artists = []
        if self.state['curves']:
            basis = bezier_basis(np.linspace(0, 1, 300))
            for curve in self.preview_curves:
                points = basis@np.array(curve['control_points'])
                self.curve_artists += self.ax.plot(points[:, 0], points[:, 1], color='magenta' if curve['family'] == 'vertical' else 'cyan', lw=.8, alpha=.7)
        if self.state['crossings']:
            self.curve_artists += plot_crossings(self.ax, self.preview_crossings, self.state['labels'])
        confirmed = editor.frame in self.session.anchors or editor.frame in self.session.edits
        title = f"{sum(bool(n.get('human_edited')) for n in editor.nodes)} human-edited nodes" if confirmed else 'Automatic tracking'
        self.ax.set_title(self.ax.get_title()+' | '+title, color='#222222')
        count = len(set(self.session.anchors) | set(self.session.edits))
        self.fig.suptitle(f'{count}/{len(self.report["frames"])} frames with human references'
                          +(' | Corrections need reanalysis' if self.session.dirty else ''), color='#222222')
        if editor.selected is not None and self.state['guides']:
            node = editor.nodes[editor.selected]
            self.state['selected'], self.state['selected_candidate'] = node['id'], None
            self.info.set_text(f"ID {node['id']} | row {node['row']}, column {node['column']} | {node.get('status')} | source: {node.get('measurement_source')}\n"
                               f"Velocity ({self.report['velocity_unit']}): {node.get('velocity')} | acceleration ({self.report['acceleration_unit']}): {node.get('acceleration')}")
        else:
            self.state['selected'] = None
            self.info.set_text('Moved and added nodes are human references. Held/missing positions update automatically.' if self.state['guides'] else
                               'Enable Guide nodes to edit. Grid crossings show the fitted intersections used for export.')
        for name, enabled in (('Export CSV', True), ('Reset edits', editor.frame != 0 and confirmed)):
            self.buttons[name].active = enabled
            self.buttons[name].label.set_color('#222222' if enabled else 'gray')

    def toggle(self, label):
        if self.mode != 'review':
            return
        key = self.OPTIONS[label]
        self.state[key] = not self.state[key]
        if self.editor is None:
            self.draw()
        else:
            self.apply_options()

    def toggle_help(self, event=None):
        if self.mode != 'review':
            return
        self.fps_box.stop_typing()
        self.help.toggle()
        self.buttons['Help'].label.set_text('Close help' if self.help.visible else 'Help')
        if self.editor is not None:
            self.editor.active = not self.help.visible

    def edited(self, editor, before, nodes):
        try:
            fps = parse_fps(self.fps_box.text)
        except ValueError:
            fps = self.session.fps  # Incomplete time text must not block node edits or undo.
        try:
            nodes = self.session.apply_edit(editor.frame, before, nodes, fps=fps)
        except ValueError as error:
            self.say(f'Edit could not be applied: {error}')
            return before
        self.say('Human edit saved. Later held/missing nodes were updated; observations were kept.')
        return nodes

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
        self.say('Reanalyzing with fixed human-edited nodes…')
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
        self.say('Reanalysis complete. Human-edited coordinates stay fixed.' if frame_range is None else
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
            self.say(f'Frame {index} edits reset; previous edits are in reviews/history.')
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
        self.say(f'Exported fitted crossings, curves, and candidates to {paths[0].parent}', path=paths[0].parent)

    def export_video(self, event=None):
        from plot_widgets import paused_widgets
        if self.session is None or self.mode != 'review':
            return
        self.editor.autosave()
        self.editor.active = False
        self.mode = 'processing'
        self.say('Exporting fitted grid crossings to MP4 (30 fps playback)…')
        def progress(done, total):
            if done % 10 == 0 or done == total:
                self.say(f'Exporting MP4: {done}/{total} frames')
                self.fig.canvas.flush_events()
        try:
            with paused_widgets([self.slider, self.checks, self.fps_box, *self.buttons.values()]):
                path = self.session.export_video(progress=progress)
        except Exception as error:
            self.say(f'MP4 export failed: {error}')
            raise
        finally:
            self.mode = 'review'
            self.apply_options()
        self.say(f'Exported {path.name} (30 fps playback) to {path.parent}', path=path.parent)

    def calibrate(self, event=None):
        if self.session is None or self.mode != 'review':
            return
        from calibration import CalibrationPanel, load_calibration
        self.editor.autosave()
        self.editor.active = False
        self.fps_box.stop_typing()
        self.mode = 'calibration'
        self.messages.set_visible(False)
        def done(document):
            self.mode, self.calibration_panel = 'review', None
            self.messages.set_visible(True)
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


