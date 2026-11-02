"""An optional help box inside the existing application window."""
HELP_TEXT = (
    'EDIT NODES\n'
    'Drag a node to move it. Right-click or Delete removes it.\n'
    'With the pointer over the image: a adds a crossing; e adds an edge node.\n'
    'Cmd-Z / Ctrl-Z, z, or u undoes; [ / ] changes column; , / . changes row.\n'
    'Use the toolbar to zoom or pan; deselect it to move nodes again.\n\n'
    'FRAMES AND SAVING\n'
    'Moved/added nodes become human references immediately.\n'
    'Later held/missing nodes update automatically until the next observation.\n'
    'Reanalyze reruns tracking while preserving each human-edited node.\n'
    'Next 10 limits reanalysis to the next frames; Reset edits releases a frame.\n'
    'Select/copy status text below; click an output path to open its folder.\n'
    'Export CSV and Export MP4 use fitted crossings and include saved drafts.\n'
    'MP4 previews play at 30 fps; the measurement sampling rate is unchanged.\n\n'
    'COLORS AND CHECKS\n'
    'In review: green = tracked/confirmed guides; blue = edited guides.\n'
    'Orange x = held/missing guides; white dots = fitted grid crossings.\n'
    'Yellow rings = suggestions; red rings = conflicting identities.\n'
    'Toggle Grid crossings to show fitted intersections, or Guide nodes to edit.\n'
    'Candidates and Missing control the two types of suggestions.\n'
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
