@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" launch_gridtracker.py
) else (
    where py >nul 2>nul
    if errorlevel 1 (
        python launch_gridtracker.py
    ) else (
        py -3 launch_gridtracker.py
    )
)
if errorlevel 1 (
    echo.
    echo GridTracker exited with an error ^(see above^). See README.md for setup.
    pause
    exit /b 1
)
