#!/bin/bash
# Double-click (macOS) to start the neural painter web app; it opens in your browser at http://127.0.0.1:7860.
# The first time, make this file executable once in Terminal:  chmod +x start_app.command
# Close the Terminal window to stop the app.
cd "$(dirname "$0")" || exit 1
if [ -x venv/bin/python ]; then PY=venv/bin/python; else PY=python3; fi
echo "Starting the neural painter... (the first start takes about half a minute)"
"$PY" -m neural_painter.app.gradio_app || {
  echo
  echo "The app stopped with an error. If it says a module is missing, see README.md, Quick start."
}
read -r -p "Press Enter to close."
