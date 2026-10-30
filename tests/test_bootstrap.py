from contextlib import redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import bootstrap_gridtracker as bootstrap


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix='gridtracker setup ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.environment = self.root/'.venv'
        self.python = self.environment/'bin/python'
        self.requirements = self.root/'requirements.txt'
        self.requirements.write_text('numpy>=2\n')
        self.log = io.StringIO()
        constants = patch.multiple(bootstrap, ROOT=self.root, ENV=self.environment,
                                   PYTHON=self.python, REQUIREMENTS=self.requirements,
                                   LOG=self.root/'gridtracker_startup.log')
        constants.start()
        self.addCleanup(constants.stop)
        quiet = redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def existing_environment(self):
        self.python.parent.mkdir(parents=True)
        self.python.touch()

    def result(self, status=0, error=''):
        return subprocess.CompletedProcess([], status, stdout='', stderr=error)

    def test_first_launch_creates_venv_installs_dependencies_and_checks_imports(self):
        calls = []
        import_checks = 0
        def probe(code):
            nonlocal import_checks
            if code == bootstrap.IMPORTS:
                import_checks += 1
                return self.result(1 if import_checks == 1 else 0)
            if code == 'import pip':
                return self.result(1)
            return self.result()
        def run(command, log):
            calls.append(command)
            if 'venv' in command:
                self.existing_environment()
            return 0
        with patch.object(bootstrap, 'probe', side_effect=probe), patch.object(bootstrap, 'run_logged', side_effect=run):
            self.assertEqual(bootstrap.prepare_environment(self.log), self.python)
        self.assertIn('venv', calls[0])
        self.assertIn('ensurepip', calls[1])
        self.assertEqual(calls[2][0], str(self.python))
        self.assertIn('--only-binary=:all:', calls[2])
        self.assertIn(str(self.requirements), calls[2])
        self.assertEqual(import_checks, 2)
        self.assertEqual((self.environment/'.gridtracker_requirements.sha256').read_text().strip(),
                         hashlib.sha256(self.requirements.read_bytes()).hexdigest())

    def test_later_launch_works_offline_and_reinstalls_only_after_requirements_change(self):
        self.existing_environment()
        stamp = self.environment/'.gridtracker_requirements.sha256'
        stamp.write_text(hashlib.sha256(self.requirements.read_bytes()).hexdigest())
        with patch.object(bootstrap, 'probe', return_value=self.result()), patch.object(bootstrap, 'run_logged', return_value=0) as run:
            bootstrap.prepare_environment(self.log)
            run.assert_not_called()
            self.requirements.write_text('numpy>=2\nscipy>=1.14\n')
            bootstrap.prepare_environment(self.log)
            self.assertEqual(run.call_count, 1)
            self.assertIn('pip', run.call_args.args[0])
            self.assertNotIn('--force-reinstall', run.call_args.args[0])

    def test_tk_failure_explains_system_package_before_installing_libraries(self):
        self.existing_environment()
        def probe(code):
            return self.result(1, 'No module named _tkinter') if code == bootstrap.TK_CHECK else self.result()
        with patch.object(bootstrap, 'probe', side_effect=probe), patch.object(bootstrap, 'run_logged') as run, \
             patch.object(bootstrap.sys, 'platform', 'linux'):
            with self.assertRaisesRegex(bootstrap.SetupError, 'python3-tk'):
                bootstrap.prepare_environment(self.log)
            run.assert_not_called()
        self.assertIn('_tkinter', self.log.getvalue())

    def test_headless_desktop_error_is_distinguished_from_missing_tk(self):
        self.existing_environment()
        def probe(code):
            return self.result(1, 'no display name and no $DISPLAY environment variable') if code == bootstrap.TK_CHECK else self.result()
        with patch.object(bootstrap, 'probe', side_effect=probe):
            with self.assertRaisesRegex(bootstrap.SetupError, 'graphical desktop'):
                bootstrap.prepare_environment(self.log)

    def test_failed_install_is_actionable_and_does_not_mark_environment_ready(self):
        self.existing_environment()
        with patch.object(bootstrap, 'probe', return_value=self.result()), \
             patch.object(bootstrap, 'run_logged', return_value=1):
            with self.assertRaisesRegex(bootstrap.SetupError, 'internet connection'):
                bootstrap.prepare_environment(self.log)
        self.assertFalse((self.environment/'.gridtracker_requirements.sha256').exists())

    def test_foreign_environment_is_not_silently_overwritten(self):
        self.environment.mkdir()
        keep = self.environment/'keep'
        keep.write_text('existing environment')
        with self.assertRaisesRegex(bootstrap.SetupError, 'another operating system'):
            bootstrap.prepare_environment(self.log)
        self.assertEqual(keep.read_text(), 'existing environment')

    def test_interpreter_outside_managed_environment_cannot_install_packages(self):
        real_probe = bootstrap.probe
        def probe(code):
            return real_probe(code) if 'sys.version_info' in code else self.result()
        with patch.object(bootstrap, 'PYTHON', Path(sys.executable)), \
             patch.object(bootstrap, 'probe', side_effect=probe), patch.object(bootstrap, 'run_logged') as run:
            with self.assertRaisesRegex(bootstrap.SetupError, 'local .venv'):
                bootstrap.prepare_environment(self.log)
            run.assert_not_called()

    def test_setup_error_stays_in_startup_log_and_application_does_not_launch(self):
        with patch.object(bootstrap, 'prepare_environment', side_effect=bootstrap.SetupError('Install Python Tk')), \
             patch.object(bootstrap, 'run_logged') as run:
            self.assertEqual(bootstrap.main(), 1)
            run.assert_not_called()
        self.assertIn('Install Python Tk', bootstrap.LOG.read_text())

    @unittest.skipIf(os.name == 'nt', 'Unix shell launchers')
    def test_unix_launchers_quote_paths_and_run_the_shared_bootstrap(self):
        self.existing_environment()
        self.python.write_text('#!/bin/sh\nif [ "$1" = "-c" ]; then exit 0; fi\nprintf "%s\\n" "$PWD" "$@"\n')
        self.python.chmod(0o755)
        source = Path(__file__).resolve().parents[1]
        names = ('Start GridTracker (Linux).sh', 'Start GridTracker (Mac).command')
        for name in names:
            shutil.copy2(source/name, self.root/name)
        for name in names:
            with self.subTest(name=name):
                result = subprocess.run([str(self.root/name)], cwd='/tmp', capture_output=True, text=True, check=True)
                lines = result.stdout.splitlines()
                self.assertEqual(Path(lines[0]).resolve(), self.root.resolve())
                self.assertEqual(lines[1:], ['bootstrap_gridtracker.py'])

    @unittest.skipIf(os.name == 'nt', 'Unix shell launchers')
    def test_unix_missing_python_message_remains_visible(self):
        source = Path(__file__).resolve().parents[1]
        launcher = self.root/'Start GridTracker (Linux).sh'
        shutil.copy2(source/launcher.name, launcher)
        commands = self.root/'commands'
        commands.mkdir()
        (commands/'dirname').symlink_to(shutil.which('dirname'))
        result = subprocess.run([str(launcher)], input='\n', capture_output=True, text=True,
                                env=dict(os.environ, PATH=str(commands)))
        self.assertEqual(result.returncode, 1)
        self.assertIn('Python 3.10 or later was not found', result.stdout)
        self.assertIn('python.org/downloads', result.stdout)
        self.assertIn('Press Enter', result.stdout)
