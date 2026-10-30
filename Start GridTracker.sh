#!/bin/sh
cd -- "$(dirname -- "$0")" || exit 1
if [ -x .venv/bin/python ]; then
    .venv/bin/python launch_gridtracker.py
else
    python3 launch_gridtracker.py
fi
status=$?
if [ "$status" -ne 0 ]; then
    printf '\nGridTracker exited with an error (see above). See README.md for setup.\nPress Enter to close. '
    read -r answer
fi
exit "$status"
