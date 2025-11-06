#!/usr/bin/env bash
set -e
PY=python3

if [ ! -f requirements.txt ]; then
  echo "requirements.txt not found in current directory."
  exit 1
fi

if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  $PY -m venv venv
fi

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

# Detect macOS + Apple Silicon to recommend tensorflow-macos + tensorflow-metal
UNAME="$(uname -s)"
ARCH="$(uname -m)"
if [ "$UNAME" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
  echo "Detected macOS on Apple Silicon. Installing tensorflow-macos and tensorflow-metal..."
  pip install tensorflow-macos tensorflow-metal
  echo "Installing other python packages..."
  pip install scikit-learn numpy Pillow
else
  echo "Installing packages from requirements.txt (may install standard tensorflow)..."
  pip install -r requirements.txt
fi

echo "Setup complete. Activate the venv with: source venv/bin/activate"
echo "Then run: python train.py"
