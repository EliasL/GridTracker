@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 (
        ".venv\Scripts\python.exe" bootstrap_gridtracker.py
        goto finished
    )
)
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 (
    py -3 bootstrap_gridtracker.py
    goto finished
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 (
    python bootstrap_gridtracker.py
    goto finished
)
echo.
echo Python 3.10 or later was not found.
echo Install Python with Tk support from https://www.python.org/downloads/
echo Then run this launcher again; GridTracker will install its libraries automatically.
pause
exit /b 1

:finished
if errorlevel 1 (
    echo.
    echo Setup could not finish, or GridTracker stopped with an error.
    echo Read the message above; details are in gridtracker_startup.log if setup started.
    pause
    exit /b 1
)
exit /b 0
