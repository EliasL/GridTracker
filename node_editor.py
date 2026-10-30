"""Shared node editing, overlays, and durable drafts for both application views."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from matplotlib.markers import MarkerStyle

NODE_COLORS = dict(verified='white', observed='lime', recovered='cyan', held='gold',
                   predicted='orange', lost='red', outside='red', draft='deepskyblue')
GEOMETRY_FIELDS = ('x', 'y', 'row', 'column', 'kind', 'side', 'scan_axes')


def node_geometry(nodes):
    return {n['id']: {k: n.get(k) for k in GEOMETRY_FIELDS} for n in nodes}


def load_draft(path, result):
    from reference_editor import file_hash, analysis_hash
    document = json.loads(Path(path).read_text())
    if document['source_sha256'] != file_hash(result['source']) or document['analysis_sha256'] != analysis_hash(result):
        raise ValueError('Saved editing draft belongs to different image pixels or detection results')
    return document


class NodeEditor:
    def __init__(self, result, path, fig, ax, fps_box, *, nodes=None, roster=None,
                 next_id=None, fps=None, frame=0, output=None, poll=None,
                 on_redraw=None, review=False, restore_fps=True):
        from reference_editor import label_node
        self.result, self.path = result, Path(path)
        self.fig, self.ax, self.fps_box = fig, ax, fps_box
        self.frame, self.fps, self.active = frame, fps, True
        self.review, self.on_redraw = review, on_redraw
        self.output = Path(output) if output is not None else self.path.parent
        self.nodes = deepcopy(nodes) if nodes is not None else [label_node(n, result) for n in result['final_nodes']]
        self.baseline = deepcopy(self.nodes)
        self.roster = deepcopy(roster if roster is not None else self.nodes)
        self.draft_path = self.path.with_suffix('.draft.json')
        editor_state = {}
        if self.draft_path.exists():
            draft = load_draft(self.draft_path, result)
            editor_state = draft.get('editor_state', {})
            if not review or editor_state.get('modified', node_geometry(draft['nodes']) != node_geometry(self.nodes)):
                self.nodes = draft['nodes']
            if restore_fps:
                self.fps_box.set_val(editor_state.get('fps_text', '' if draft.get('fps') is None else str(draft['fps'])))
        self.history, self.selected, self.dragging, self.approved = [], None, False, None
        self.next_id = max([n['id'] for n in result['candidate_nodes']+self.nodes+self.roster]+[0])+1
        self.next_id = max(self.next_id, next_id or 0, editor_state.get('next_id', 0))
        self.selected = next((i for i, n in enumerate(self.nodes) if n['id'] == editor_state.get('selected_id')), None)
        self.show_labels, self.show_candidates = editor_state.get('show_labels', False), editor_state.get('show_candidates', False)
        self.show_missing, self.show_predictions, self.show_connections = True, True, not review
        self._last_geometry = node_geometry(self.baseline)
        self.modified, self._disposed = False, False
        self.points = ax.scatter([], [], s=40, marker='+', c='white', picker=10)
        self.selected_point = ax.scatter([], [], s=110, facecolors='none', edgecolors='cyan')
        self.candidates = ax.scatter([], [], s=24, facecolors='none', edgecolors='violet')
        self.missing = ax.scatter([], [], s=650, facecolors='none', edgecolors='yellow', linewidths=2.2)
        self.conflicts = ax.scatter([], [], s=180, facecolors='none', edgecolors='red', linewidths=2)
        self.lines, self.labels = [], []
        self.cids = [fig.canvas.mpl_connect(event, callback) for event, callback in
                     (('button_press_event', self.press), ('button_release_event', self.release),
                      ('motion_notify_event', self.move), ('key_press_event', self.key), ('close_event', self.close))]
        self.timer = fig.canvas.new_timer(interval=500)
        if poll is not None:
            self.timer.add_callback(poll)
        self._draft_dirty = False
        self.timer.add_callback(self.flush_autosave)
        self.fps_cid = fps_box.on_text_change(lambda text: self.autosave())
        self.redraw()
        if 'xlim' in editor_state and 'ylim' in editor_state:
            ax.set_xlim(editor_state['xlim'])
            ax.set_ylim(editor_state['ylim'])
        self.autosave()
        self.timer.start()

    def autosave(self):
        from reference_editor import parse_fps, save_reference, save_ui_state
        if self.approved is not None:
            return
        try:
            self.fps = parse_fps(self.fps_box.text)
        except ValueError:
            pass  # Incomplete text is a valid draft; approval still validates it.
        selected = self.nodes[self.selected]['id'] if self.selected is not None else None
        state = dict(next_id=self.next_id, selected_id=selected, show_labels=self.show_labels,
                     show_candidates=self.show_candidates, fps_text=self.fps_box.text,
                     xlim=list(self.ax.get_xlim()), ylim=list(self.ax.get_ylim()), modified=self.modified)
        save_reference(self.draft_path, self.result, self.nodes, approved=False, fps=self.fps, editor_state=state)
        save_ui_state(self.output, self.frame, 'review' if self.review else 'edit')
        self._draft_dirty = False

    def flush_autosave(self):
        if self._draft_dirty:
            self.autosave()
        return True

    def redraw(self):
        from mesh_hints import missing_crossings
        from mesh_hints import identity
        from collections import Counter
        from fit_bezier_curves import node_available
        geometry = node_geometry(self.nodes)
        baseline = node_geometry(self.baseline)
        if self.review:
            originals = {n['id']: n for n in self.baseline}
            for node in self.nodes:
                key = node['id']
                if geometry[key] != self._last_geometry.get(key):
                    if geometry[key] == baseline.get(key):
                        for field in ('measured_xy', 'status', 'measurement_source'):
                            node[field] = originals[key].get(field)
                    else:
                        node.update(measured_xy=[node['x'], node['y']], status='draft', measurement_source='human_draft',
                                    velocity=None, acceleration=None)
        self.modified = geometry != baseline
        self._last_geometry = geometry
        points = np.array([[n['x'], n['y']] for n in self.nodes]).reshape(-1, 2)
        visible = [i for i, n in enumerate(self.nodes) if not self.review or self.show_predictions or node_available(n)]
        self.points.set_offsets(points[visible])
        self.points.set_color([NODE_COLORS[self.nodes[i]['status']] if self.review else 'yellow' if self.nodes[i]['kind'] == 'edge' else 'white' for i in visible])
        markers = [MarkerStyle('o' if self.nodes[i].get('status') == 'held' else
                               'x' if self.review and not node_available(self.nodes[i]) else '+') for i in visible]
        self.points.set_paths([m.get_path().transformed(m.get_transform()) for m in markers])
        self.selected_point.set_offsets(points[[self.selected]] if self.selected is not None else np.empty((0, 2)))
        candidate_points = [[n['x'], n['y']] for n in self.result['candidate_nodes']] if self.show_candidates else []
        self.candidates.set_offsets(np.array(candidate_points).reshape(-1, 2))
        eligible = [n for n in self.nodes if not self.review or node_available(n)]
        self.hints = missing_crossings(eligible, self.result['image_size'], self.result['width']) if self.show_missing else []
        self.missing.set_offsets(np.array([[h['x'], h['y']] for h in self.hints]).reshape(-1, 2))
        counts = Counter(identity(n) for n in self.nodes)
        conflicting = [n for n in self.nodes if counts[identity(n)] > 1]
        self.conflicts.set_offsets(np.array([[n['x'], n['y']] for n in conflicting]).reshape(-1, 2))
        for artist in self.lines+self.labels:
            artist.remove()
        self.lines, self.labels = [], []
        for key, coordinate, color in (('row', 'x', 'cyan'), ('column', 'y', 'magenta')) if self.show_connections else ():
            for identity in {n[key] for n in self.nodes if n[key] is not None}:
                line = sorted([n for n in self.nodes if n[key] == identity], key=lambda n: n[coordinate])
                self.lines += self.ax.plot([n['x'] for n in line], [n['y'] for n in line], color=color, alpha=.35, lw=.7)
        if self.show_labels:
            self.labels = [self.ax.text(n['x']+3, n['y']-3, f"{n['row']},{n['column']}", color='white', fontsize=7) for n in self.nodes]
        detail = '' if self.selected is None else f" | selected ID {self.nodes[self.selected]['id']}: row {self.nodes[self.selected]['row']}, column {self.nodes[self.selected]['column']}"
        self.ax.set_title(f'Frame {self.frame}: {len(self.nodes)} nodes, {len(self.hints)} possible gaps, {len(conflicting)} conflicting nodes{detail}')
        if self.on_redraw is not None:
            self.on_redraw(self)
        self.fig.canvas.draw_idle()

    def nearest(self, event):
        from fit_bezier_curves import node_available
        if not self.nodes:
            return None
        display = self.ax.transData.transform([[n['x'], n['y']] for n in self.nodes])
        distance = np.linalg.norm(display-[event.x, event.y], axis=1)
        if self.review and not self.show_predictions:
            distance[[not node_available(n) for n in self.nodes]] = np.inf
        return int(np.argmin(distance)) if distance.min() <= 14 else None

    def press(self, event):
        if not self.active or event.inaxes != self.ax or getattr(self.fig.canvas.toolbar, 'mode', ''):
            return
        self.selected = self.nearest(event)
        if self.selected is not None:
            self.history.append(deepcopy(self.nodes))
            if event.button == 3:
                self.nodes.pop(self.selected)
                self.selected = None
            else:
                self.dragging = event.button == 1
        self.redraw()
        self.autosave()

    def release(self, event):
        self.dragging = False
        if self.active:
            self.autosave()

    def move(self, event):
        if self.active and self.dragging and self.selected is not None and event.inaxes == self.ax:
            node = self.nodes[self.selected]
            node['x'], node['y'] = [float(np.clip(v, 0, end-1)) for v, end in zip((event.xdata, event.ydata), self.result['image_size'])]
            if node['kind'] == 'edge':
                self.snap_edge(node)
            self._draft_dirty = True
            self.redraw()

    def snap_edge(self, node):
        width, height = self.result['image_size']
        side = node['side']
        node['y' if side in ('top', 'bottom') else 'x'] = 0. if side in ('top', 'left') else float(height-1 if side == 'bottom' else width-1)

    def key(self, event):
        if not self.active or self.fps_box.capturekeystrokes or getattr(event, 'inaxes', None) == self.fps_box.ax:
            return
        if self.review and event.key in ('i', 'h'):
            return  # The review's checkboxes own these shared visibility options.
        edited = True
        if event.key in ('i', 'h'):
            name = 'show_labels' if event.key == 'i' else 'show_candidates'
            setattr(self, name, not getattr(self, name))
        elif event.key == 'u' and self.history:
            self.nodes, self.selected = self.history.pop(), None
        elif event.key in ('delete', 'backspace') and self.selected is not None:
            self.history.append(deepcopy(self.nodes))
            self.nodes.pop(self.selected)
            self.selected = None
        elif event.key in ('[', ']', ',', '.') and self.selected is not None:
            node = self.nodes[self.selected]
            label = 'column' if event.key in ('[', ']') else 'row'
            if node[label] is not None:
                self.history.append(deepcopy(self.nodes))
                node[label] = max(0, node[label]+(-1 if event.key in ('[', ',') else 1))
        elif event.key in ('a', 'e') and event.inaxes == self.ax and not getattr(self.fig.canvas.toolbar, 'mode', ''):
            from mesh_hints import identity, label_from_mesh
            self.history.append(deepcopy(self.nodes))
            node = dict(id=self.next_id, x=float(event.xdata), y=float(event.ydata), kind='interior', scan_axes=['x', 'y'])
            if event.key == 'e':
                width, height = self.result['image_size']
                node.update(kind='edge', side=min(zip(('left', 'right', 'top', 'bottom'), (node['x'], width-1-node['x'], node['y'], height-1-node['y'])), key=lambda v: v[1])[0])
                node['scan_axes'] = ['x'] if node['side'] in ('top', 'bottom') else ['y']
                self.snap_edge(node)
            existing_keys = {identity(n) for n in self.nodes}
            labeling_nodes = self.nodes+[n for n in self.roster if identity(n) not in existing_keys]
            node = label_from_mesh(node, labeling_nodes, self.result)
            known = next((n for n in self.roster if identity(n) == identity(node)), None)
            if known is not None:
                node['id'] = known['id']
                if self.review:
                    from fit_bezier_curves import node_available
                    self.nodes = [n for n in self.nodes if n['id'] != node['id'] or node_available(n)]
            else:
                self.next_id += 1
            self.nodes.append(node)
            self.selected = len(self.nodes)-1
        else:
            edited = False
        self.redraw()
        if edited:
            self.autosave()

    def close(self, event=None):
        self.timer.stop()
        if not self._disposed:
            self.autosave()

    def dispose(self):
        if self._disposed:
            return
        from plot_widgets import release_input
        self.timer.stop()
        self.active, self._disposed = False, True
        for cid in self.cids:
            self.fig.canvas.mpl_disconnect(cid)
        self.fps_box.disconnect(self.fps_cid)
        release_input(self.fps_box)
        for artist in [self.points, self.selected_point, self.candidates, self.missing, self.conflicts]+self.lines+self.labels:
            artist.remove()
