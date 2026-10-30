"""First-click setup using only the Python standard library and a local venv."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ENV = ROOT/'.venv'
PYTHON = ENV/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
REQUIREMENTS = ROOT/'requirements.txt'
LOG = ROOT/'gridtracker_startup.log'
IMPORTS = ('import numpy, cv2; from PIL import Image; '
           'from scipy.optimize import least_squares; '
           'from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg')
TK_CHECK = ('from gui_support import configure_tk; configure_tk(); '
            'import tkinter; root = tkinter.Tk(); root.withdraw(); root.destroy()')


class SetupError(RuntimeError):
    pass


def child_environment():
    environment = dict(os.environ, PYTHONIOENCODING='utf-8',
                       MPLCONFIGDIR=str(ROOT/'.matplotlib-cache'))
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[name] = '1'
    return environment


def probe(code):
    return subprocess.run([str(PYTHON), '-c', code], cwd=ROOT, env=child_environment(),
                          capture_output=True, text=True, encoding='utf-8', errors='replace')


def run_logged(command, log):
    """Keep installation progress visible and retain the complete error output."""
    with subprocess.Popen(command, cwd=ROOT, env=child_environment(), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as process:
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def prepare_environment(log):
    if sys.version_info < (3, 10):
        raise SetupError(f'Python 3.10 or later is required; this launcher found {sys.version.split()[0]}.\n'
                         'Install Python from https://www.python.org/downloads/ and try again.')
    if not PYTHON.is_file():
        if ENV.exists():
            raise SetupError('The .venv folder is incomplete or belongs to another operating system.\n'
                             'Delete only .venv (the installed Python libraries), then start GridTracker again.\n'
                             'Keep your image and gridtracker_results folders.')
        print('First-time setup: creating a local Python environment in .venv ...', flush=True)
        if run_logged([sys.executable, '-m', 'venv', str(ENV)], log):
            raise SetupError('Could not create the Python environment.\n'
                             'Extract the entire ZIP into a writable folder first.\n'
                             'On Linux, install the matching python3-venv system package and try again.')
    try:
        version = probe('import sys; from pathlib import Path; '
                        'assert sys.version_info >= (3, 10), sys.version; '
                        'assert sys.prefix != sys.base_prefix, "Not a virtual environment"; '
                        f'assert Path(sys.prefix).resolve() == Path({str(ENV)!r}).resolve(), "Unexpected environment path"')
    except OSError as error:
        raise SetupError('The local .venv Python cannot run. Remove .venv and start again with Python 3.10+.') from error
    if version.returncode:
        log.write(version.stdout+version.stderr)
        raise SetupError('The local .venv requires a newer or repaired Python installation.\n'
                         'Remove .venv, install Python 3.10 or later, and start again.')
    print('Checking the graphical interface ...', flush=True)
    tk = probe(TK_CHECK)
    if tk.returncode:
        log.write(tk.stdout+tk.stderr)
        if any(message in tk.stderr.lower() for message in ('no display name', "couldn't connect to display")):
            raise SetupError('GridTracker needs a graphical desktop. Run it on your desktop, not a headless terminal.')
        if sys.platform.startswith('linux'):
            remedy = 'Install the matching python3-tk system package using your Linux package manager.'
        else:
            remedy = ('Install/reinstall Python with Tcl/Tk support from https://www.python.org/downloads/.\n'
                      'Then remove .venv and start GridTracker again.')
        raise SetupError('Python is installed, but its Tk graphical interface is unavailable.\n'
                         +remedy+'\nTk is part of Python/system setup; pip cannot install it.')
    fingerprint = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    stamp = ENV/'.gridtracker_requirements.sha256'
    ready = probe(IMPORTS)
    if not stamp.exists() or stamp.read_text().strip() != fingerprint or ready.returncode:
        if probe('import pip').returncode:
            print('Setting up pip in the local environment ...', flush=True)
            if run_logged([str(PYTHON), '-m', 'ensurepip', '--upgrade'], log):
                raise SetupError('Could not set up pip. Reinstall Python with pip/venv support, remove .venv, and retry.')
        print('Installing/checking GridTracker libraries. The first installation needs internet and may take a few minutes.', flush=True)
        command = [str(PYTHON), '-m', 'pip', 'install', '--disable-pip-version-check',
                   '--only-binary=:all:', '--progress-bar', 'off', '-r', str(REQUIREMENTS)]
        if ready.returncode:
            command.append('--force-reinstall')
        if run_logged(command, log):
            raise SetupError('The libraries could not be installed. Check your internet connection and the pip error above.\n'
                             'If pip reports no matching distribution, check that Python is a compatible 64-bit version.\n'
                             'Run the launcher again to retry; your images and saved results are unchanged.')
        ready = probe(IMPORTS)
        if ready.returncode:
            log.write(ready.stdout+ready.stderr)
            raise SetupError('Libraries were installed but could not load. The detailed import error is in the startup log.\n'
                             'Remove .venv and retry with a compatible 64-bit Python installation.')
        stamp.write_text(fingerprint+'\n')
    print('Setup complete. Opening GridTracker ...', flush=True)
    return PYTHON


def main():
    try:
        with LOG.open('w', encoding='utf-8') as log:
            log.write(f'Python: {sys.version}\nInterpreter: {sys.executable}\nProject: {ROOT}\n')
            try:
                python = prepare_environment(log)
                status = run_logged([str(python), str(ROOT/'launch_gridtracker.py')], log)
                if status:
                    raise SetupError('GridTracker stopped with an error. See the output above and the startup log below.')
                return 0
            except SetupError as error:
                message = f'\nGridTracker could not start:\n{error}\n\nStartup log: {LOG}\n'
                print(message, flush=True)
                log.write(message)
                return 1
    except OSError as error:
        print(f'\nGridTracker could not read/write its setup files: {error}\n'
              'Fully extract the ZIP into a writable folder, such as Documents, and try again.', flush=True)
        return 1
    except KeyboardInterrupt:
        print('\nGridTracker startup canceled.', flush=True)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
