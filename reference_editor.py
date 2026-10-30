"""Human approval of the first frame; importing this module never opens a GUI."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


def configure_backend(name):
    if name.lower() == 'tkagg':
        from gui_support import configure_tk
        configure_tk()
    import matplotlib
    matplotlib.use(name)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analysis_hash(result):
    fields = {k: result[k] for k in ('final_nodes', 'maxima', 'width')}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def label_node(node, result):
    node = deepcopy(node)
    edge = node.get('kind') == 'edge'
    for axis, name in (('x', 'column'), ('y', 'row')):
        eligible = not edge or axis in node['scan_axes']
        peaks = np.asarray(result['maxima'][axis]['clean'])
        node[name] = int(np.argmin(abs(peaks-node[axis]))) if eligible else None
        if eligible and len(peaks) >= 2:
            pitch = float(np.median(np.diff(peaks)))
            end = result['image_size'][0 if axis == 'x' else 1]-1
            border = max(2*result['width'], .3*pitch)
            if node[axis] > peaks[-1]+pitch/2 and end-node[axis] < border:
                node[name] = len(peaks)  # A partially visible extra string.
            elif node[axis] < peaks[0]-pitch/2 and node[axis] < border:
                node[name] = len(peaks)+1
    return node


def validate_nodes(nodes, size, require_crossing=True):
    if require_crossing and (not nodes or not any(n['kind'] == 'interior' for n in nodes)):
        raise ValueError('The reference must contain at least one crossing')
    seen, ids = set(), set()
    for n in nodes:
        if n['id'] in ids:
            raise ValueError(f"Duplicate node ID {n['id']}")
        ids.add(n['id'])
        if not np.isfinite([n['x'], n['y']]).all() or not (0 <= n['x'] < size[0] and 0 <= n['y'] < size[1]):
            raise ValueError(f"Node {n['id']} is outside the image")
        if n['kind'] == 'interior':
            key = ('crossing', n['row'], n['column'])
            values = (n['row'], n['column'])
        elif n['kind'] == 'edge':
            side = n['side']
            if side not in ('top', 'bottom', 'left', 'right'):
                raise ValueError(f'Unknown edge {side}')
            axis = 'x' if side in ('top', 'bottom') else 'y'
            label = 'column' if axis == 'x' else 'row'
            normal = 'y' if axis == 'x' else 'x'
            boundary = 0 if side in ('top', 'left') else size[1 if normal == 'y' else 0]-1
            if n[normal] != boundary or n['scan_axes'] != [axis] or n['row' if label == 'column' else 'column'] is not None:
                raise ValueError(f"Edge node {n['id']} has inconsistent boundary metadata")
            key, values = (side, n[label]), (n[label],)
        else:
            raise ValueError(f"Unknown node kind {n['kind']}")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in values):
            raise ValueError(f"Node {n['id']} needs nonnegative integer row/column labels")
        if key in seen:
            raise ValueError(f'Duplicate grid identity {key}; move, relabel, or delete a node')
        seen.add(key)
    interior = [n for n in nodes if n['kind'] == 'interior']
    for label, order, coordinate in (('row', 'column', 'x'), ('column', 'row', 'y')):
        directions = {}
        for group in {n[label] for n in interior}:
            line = sorted((n for n in interior if n[label] == group), key=lambda n: n[order])
            for i, a in enumerate(line):
                for b in line[i+1:]:
                    pair = (a[order], b[order])
                    direction = np.sign(b[coordinate]-a[coordinate])
                    if direction == 0 or (pair in directions and directions[pair] != direction):
                        raise ValueError(f'Crossed identities in {label} {group}; correct row/column labels')
                    directions[pair] = direction


def save_reference(path, result, nodes, approved, fps=None, *, editor_state=None):
    if approved:
        validate_nodes(nodes, result['image_size'], require_crossing=result.get('frame', 0) == 0)
        nodes = deepcopy(nodes)
        for node in nodes:
            if 'measured_xy' in node:
                node.update(measured_xy=[node['x'], node['y']], velocity=None, acceleration=None,
                            status='verified', measurement_source='human')
    document = dict(source=result['source'], source_sha256=file_hash(result['source']),
                    analysis_sha256=analysis_hash(result), image_size=result['image_size'],
                    approved=approved, approved_at=datetime.now(timezone.utc).isoformat() if approved else None,
                    nodes=nodes, fps=parse_fps(fps), frame=result.get('frame', 0))
    if editor_state is not None:
        document['editor_state'] = editor_state
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(document, indent=2)+'\n')
    temporary.replace(path)
    return document


def validate_reference(document, result):
    if document.get('approved') is not True or not document.get('approved_at'):
        raise ValueError('The first frame has not been explicitly approved')
    if document['source_sha256'] != file_hash(result['source']) or document['analysis_sha256'] != analysis_hash(result):
        raise ValueError('The approved reference belongs to different image pixels or detection results')
    if document['image_size'] != result['image_size']:
        raise ValueError('Reference image dimensions changed')
    validate_nodes(document['nodes'], result['image_size'], require_crossing=result.get('frame', 0) == 0)
    return document


def load_reference(path, result):
    return validate_reference(json.loads(Path(path).read_text()), result)


def save_ui_state(output, frame, mode):
    from run_temporal import write_json
    write_json(Path(output)/'ui_state.json', dict(frame=frame, mode=mode))


def load_ui_state(output, frame_count):
    path = Path(output)/'ui_state.json'
    state = dict(frame=0, mode='review')
    if path.exists():
        state = json.loads(path.read_text())
    else:
        # Older versions saved drafts but no cursor. Recover the newest
        # unfinished edit, or the most recently confirmed frame.
        fields = ('id', 'x', 'y', 'row', 'column', 'kind', 'side', 'scan_axes')
        def signature(document):
            return [{k: n.get(k) for k in fields} for n in document['nodes']], document.get('fps')
        for draft_path in sorted((Path(output)/'reviews').glob('frame_*.draft.json'), key=lambda p: p.stat().st_mtime, reverse=True):
            draft = json.loads(draft_path.read_text())
            approved_path = draft_path.with_name(draft_path.name.replace('.draft.json', '.json'))
            if not approved_path.exists() or signature(draft) != signature(json.loads(approved_path.read_text())):
                state = dict(frame=draft['frame'], mode='edit')
                break
        else:
            review = Path(output)/'review.json'
            if review.exists():
                anchors = json.loads(review.read_text())['anchors']
                latest = max(anchors, key=lambda i: anchors[i]['approved_at'])
                state['frame'] = int(latest)
    if type(state['frame']) is not int or not 0 <= state['frame'] < frame_count or state['mode'] not in ('review', 'edit'):
        raise ValueError('Saved interface state does not match this frame sequence')
    return state


def parse_fps(value):
    if value is None or str(value).strip() == '':
        return None
    fps = float(value)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Frame sampling rate must be positive and finite, or blank')
    return fps


from node_editor import NodeEditor


class ReferenceEditor(NodeEditor):
    def __init__(self, result, path, poll=None, *, nodes=None, roster=None, next_id=None,
                 fps=None, on_approve=None, validate_callback=None, frame=0,
                 figure=None, axes=None, on_cancel=None, on_export=None, output=None, keep_open=False):
        import matplotlib.pyplot as plt
        from plot_widgets import Button, TextBox
        from help_panel import HelpPanel
        self.embedded, self.keep_open = figure is not None, keep_open
        self.on_cancel, self.on_export = on_cancel, on_export
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
        self.calibration_button = Button(fig.add_axes([.29, .01, .13, .04]), 'Calibrate length')
        self.calibration_button.on_clicked(self.calibrate)
        self.export_button = Button(fig.add_axes([.44, .01, .13, .04]), 'Export CSV')
        self.export_button.on_clicked(self.export)
        self.cancel_button = Button(fig.add_axes([.59, .01, .13, .04]), 'Cancel')
        self.cancel_button.on_clicked(self.cancel)
        self.approve_button = Button(fig.add_axes([.76, .01, .22, .04]), 'Approve and track' if on_approve is None else 'Confirm frame')
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
        disconnect_widgets((self.approve_button, self.cancel_button, self.export_button, self.calibration_button, self.help_button, self.fps_box))
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
