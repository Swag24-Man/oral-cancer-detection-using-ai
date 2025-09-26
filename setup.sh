#!/usr/bin/env bash
set -e
PY=python3

if [ ! -f requirements.txt ]; then
  echo "requirements.txt not found in current directory."
  exit 1
fi

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  $PY -m venv venv
fi

# Activate venv and install requirements
echo "Activating virtual environment and installing requirements..."
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. Activate the venv with: source venv/bin/activate"
echo "Then run: python '# app.py' or python app.py"
