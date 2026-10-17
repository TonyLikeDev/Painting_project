@echo off
rem Double-click to start the neural painter web app; it opens in your browser at http://127.0.0.1:7860.
rem Close this window to stop the app.
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else (
    set "PY=python"
)
echo Starting the neural painter... (the first start takes about half a minute)
"%PY%" -m neural_painter.app.gradio_app
if errorlevel 1 (
    echo.
    echo The app stopped with an error. If it says a module is missing, see README.md, Quick start.
    echo     venv\Scripts\python.exe -m pip install -e .[app]
)
pause
