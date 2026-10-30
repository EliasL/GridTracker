#!/bin/sh
cd -- "$(dirname -- "$0")" || exit 1
python=''
for candidate in ./.venv/bin/python python3 python; do
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
        python=$candidate
        break
    fi
done
if [ -n "$python" ]; then
    "$python" bootstrap_gridtracker.py
    status=$?
else
    printf '\nPython 3.10 or later was not found.\nInstall Python with Tk support from https://www.python.org/downloads/\nThen run this launcher again; GridTracker will install its libraries automatically.\n'
    status=1
fi
if [ "$status" -ne 0 ]; then
    printf '\nSetup could not finish, or GridTracker stopped with an error.\nRead the message above; details are in gridtracker_startup.log if setup started.\nPress Enter to close. '
    read -r answer
fi
exit "$status"
