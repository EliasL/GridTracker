"""An optional help box inside the existing application window."""
HELP_TEXT = (
    'EDIT NODES\n'
    'Drag a node to move it. Right-click or Delete removes it.\n'
    'With the pointer over the image: a adds a crossing; e adds an edge node.\n'
    'Cmd-Z / Ctrl-Z, z, or u undoes; [ / ] changes column; , / . changes row.\n'
    'Use the toolbar to zoom or pan; deselect it to move nodes again.\n\n'
    'FRAMES AND SAVING\n'
    'The slider or left/right arrows changes frame. Edits autosave as drafts.\n'
    'Confirm frame fixes the current positions as a human reference.\n'
    'Reanalyze improves automatic frames while preserving confirmed frames.\n'
    'Confirm edits first to use them as references. Next 10 limits changed frames.\n'
    'Unmark frame releases a later reference and reruns tracking.\n'
    'Export CSV includes saved drafts and uses fitted curve intersections.\n\n'
    'COLORS AND CHECKS\n'
    'White: human verified; green: observed; cyan: recovered; blue: edited draft.\n'
    'Gold: held edge; orange: held interior; red x: lost/outside.\n'
    'Yellow rings suggest missing crossings; red rings flag duplicate identities.\n'
    'Violet rings are unclaimed detections, excluded until manually added.\n'
    'The checkboxes show/hide candidates, held/lost nodes, curves, labels, and gaps.\n'
    'Tracking checks image correlation, forward/backward motion agreement,\n'
    'unique matches, and string order. These checks are not probabilities.\n'
    'Yellow gaps assume a regular grid and can be wrong near clipped edges.\n\n'
    'Enter frames/second for timed motion; blank uses frames.\n'
    'Calibrate length sets physical units. Click Help again to close this box.'
)


class HelpPanel:
    def __init__(self, fig):
        self.fig = fig
        self.ax = fig.add_axes([.055, .245, .72, .64], zorder=100, facecolor='#fffde8')
        self.ax.set(xticks=[], yticks=[])
        self.ax.text(.025, .975, HELP_TEXT, va='top', color='#222222', fontsize=9,
                     linespacing=1.25, transform=self.ax.transAxes)
        self.ax.set_visible(False)

    @property
    def visible(self):
        return self.ax.get_visible()

    def toggle(self):
        self.ax.set_visible(not self.visible)
        self.fig.canvas.draw_idle()
