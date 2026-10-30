"""Native folder chooser shared by the Windows, Linux, and macOS launchers."""
from pathlib import Path
import json

STATE_PATH = Path(__file__).with_name('.launcher_state.json')


class FolderChooser:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk

        self.root, self.selection = root, None
        saved = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
        self.input = tk.StringVar(root, value=saved.get('input', ''))
        self.output = tk.StringVar(root, value=saved.get('output', ''))
        self.same_folder = tk.BooleanVar(root, value=saved.get('same_folder', True))
        root.title('Open GridTracker')
        root.columnconfigure(1, weight=1)
        root.minsize(600, 200)
        ttk.Label(root, text='Folder of image frames').grid(row=0, column=0, padx=12, pady=12, sticky='w')
        ttk.Entry(root, textvariable=self.input).grid(row=0, column=1, sticky='ew')
        ttk.Button(root, text='Browse…', command=self.choose_input).grid(row=0, column=2, padx=12)
        ttk.Checkbutton(root, text='Save results in a subfolder of the frames folder', variable=self.same_folder,
                        command=self.update_output).grid(row=1, column=1, pady=4, sticky='w')
        ttk.Label(root, text='Results folder').grid(row=2, column=0, padx=12, pady=12, sticky='w')
        self.output_entry = ttk.Entry(root, textvariable=self.output)
        self.output_entry.grid(row=2, column=1, sticky='ew')
        self.output_browse = ttk.Button(root, text='Browse…', command=self.choose_output)
        self.output_browse.grid(row=2, column=2, padx=12)
        ttk.Button(root, text='Open GridTracker', command=self.start).grid(row=3, column=1, pady=12, sticky='e')
        self.input.trace_add('write', lambda *_: self.update_output())
        self.update_output()

    def update_output(self):
        same = self.same_folder.get()
        if same:
            source = self.input.get().strip()
            self.output.set(str(Path(source)/'gridtracker_results') if source else '')
        for widget in (self.output_entry, self.output_browse):
            widget.configure(state='disabled' if same else 'normal')

    def choose_input(self):
        from tkinter.filedialog import askdirectory
        previous = self.input.get().strip()
        path = askdirectory(parent=self.root, title='Choose the folder of image frames',
                            initialdir=str(Path(previous).expanduser().parent if previous else Path.home()), mustexist=True)
        if path:
            self.input.set(path)
            self.save_folders()

    def choose_output(self):
        from tkinter.filedialog import askdirectory
        previous = self.output.get().strip() or self.input.get().strip()
        path = askdirectory(parent=self.root, title='Choose the results folder',
                            initialdir=str(Path(previous).expanduser().parent if previous else Path.home()), mustexist=False)
        if path:
            self.output.set(path)
            self.save_folders()

    def save_folders(self):
        temporary = STATE_PATH.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(input=self.input.get(), output=self.output.get(),
                                             same_folder=self.same_folder.get()), indent=2)+'\n')
        temporary.replace(STATE_PATH)

    def start(self):
        from tkinter import messagebox
        from run_temporal import discover_frames
        try:
            if not self.input.get().strip() or not self.output.get().strip():
                raise ValueError('Choose both an image folder and a results folder.')
            source = Path(self.input.get()).expanduser().resolve()
            destination = Path(self.output.get()).expanduser().resolve()
            discover_frames(source, output=destination)
            if destination.exists() and not destination.is_dir():
                raise ValueError(f'Results destination is not a folder: {destination}')
            self.save_folders()
        except (ValueError, OSError) as error:
            messagebox.showerror('Cannot open GridTracker', str(error), parent=self.root)
            return
        self.selection = source, destination
        self.root.destroy()


def main():
    # Set numerical thread limits before importing any numerical libraries.
    from run_temporal import main as run
    from gui_support import configure_tk
    configure_tk()
    import tkinter as tk
    root = tk.Tk()
    chooser = FolderChooser(root)
    root.mainloop()
    if chooser.selection is not None:
        source, destination = chooser.selection
        run([str(source), '--output', str(destination)])


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
