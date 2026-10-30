"""Shared setup for the native folder chooser and Matplotlib's Tk window."""
import os
from pathlib import Path
import sys


def configure_tk():
    import _tkinter
    # Relocatable uv Python can retain Tcl/Tk paths from its build machine.
    for variable, library in (('TCL_LIBRARY', f'tcl{_tkinter.TCL_VERSION}'),
                              ('TK_LIBRARY', f'tk{_tkinter.TK_VERSION}')):
        path = Path(sys.base_prefix)/'lib'/library
        if path.is_dir():
            os.environ.setdefault(variable, str(path))
