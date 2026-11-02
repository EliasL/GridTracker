"""Bounded, selectable status text with links to output folders in Tk windows."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap


def open_folder(path):
    path = Path(path).resolve()
    if not path.is_dir():
        raise ValueError(f'Folder does not exist: {path}')
    if sys.platform == 'win32':
        os.startfile(str(path))
    else:
        subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', str(path)])


class MessageChannel:
    def __init__(self, panel, name):
        self.panel, self.name = panel, name

    def set_text(self, text):
        self.panel.set(text, channel=self.name)

    def get_text(self):
        return self.panel.values.get(self.name, '')


class MessagePanel:
    def __init__(self, figure, rectangle):
        self.figure, self.rectangle = figure, rectangle
        self.values, self.folder = {}, None
        self.widget, self.artist = None, None
        self.closed = False
        if hasattr(figure.canvas, 'get_tk_widget'):
            import tkinter as tk
            self.widget = tk.Text(figure.canvas.get_tk_widget(), wrap='word', height=3, relief='flat',
                                  background='#f3f3f3', foreground='#222222', font='TkDefaultFont',
                                  state='disabled', cursor='arrow', exportselection=False, takefocus=True)
            self.widget.tag_configure('folder', foreground='#175ca4', underline=True)
            self.widget.tag_bind('folder', '<ButtonRelease-1>', self.follow_link)
            self.widget.tag_bind('folder', '<Enter>', lambda event: self.widget.configure(cursor='hand2'))
            self.widget.tag_bind('folder', '<Leave>', lambda event: self.widget.configure(cursor='arrow'))
            for modifier in {'Control', 'Command' if sys.platform == 'darwin' else 'Control'}:
                self.widget.bind(f'<{modifier}-c>', self.copy)
                self.widget.bind(f'<{modifier}-a>', self.select_all)
        else:
            x, y, width, height = rectangle
            self.artist = figure.text(x, y+height, '', va='top', fontsize=9, color='#222222')
        self.set_visible(True)

    def channel(self, name):
        return MessageChannel(self, name)

    def set(self, text, *, channel='message', path=None):
        self.values[channel] = str(text)
        if channel == 'message':
            self.folder = Path(path) if path is not None else None
        combined = '\n'.join(self.values[key] for key in ('message', 'info') if self.values.get(key))
        if self.folder is not None and str(self.folder) not in combined:
            combined += '\n'+str(self.folder)
        if self.widget is not None:
            self.widget.configure(state='normal')
            self.widget.delete('1.0', 'end')
            self.widget.insert('1.0', combined)
            if self.folder is not None:
                begin = combined.find(str(self.folder))
                self.widget.tag_add('folder', f'1.0+{begin}c', f'1.0+{begin+len(str(self.folder))}c')
            self.widget.configure(state='disabled')
        else:
            width = max(10, int(self.figure.bbox.width*self.rectangle[2]/6.5))
            self.artist.set_text('\n'.join(textwrap.fill(line, width=width) for line in combined.splitlines()))
        self.figure.canvas.draw_idle()

    def select_all(self, event=None):
        self.widget.tag_add('sel', '1.0', 'end-1c')
        return 'break'

    def follow_link(self, event=None):
        if self.folder is not None and not self.widget.tag_ranges('sel'):
            open_folder(self.folder)

    def copy(self, event=None):
        import tkinter as tk
        try:
            selected = self.widget.get('sel.first', 'sel.last')
        except tk.TclError:
            return 'break'
        self.widget.clipboard_clear()
        self.widget.clipboard_append(selected)
        return 'break'

    def set_visible(self, visible):
        if self.widget is not None:
            if visible:
                x, y, width, height = self.rectangle
                self.widget.place(relx=x, rely=1-y-height, relwidth=width, relheight=height)
            else:
                self.widget.place_forget()
        elif self.artist is not None:
            self.artist.set_visible(visible)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.widget is not None:
            self.widget.destroy()
        elif self.artist is not None:
            self.artist.remove()
