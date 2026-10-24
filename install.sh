#!/bin/bash
# ByteEats Vector Similarity Search Installation Script

echo "ByteEats Vector Similarity Search Installation"
echo "=============================================="
echo

# Check for Python 3.12+ (set PYTHON to use a different interpreter)
PYTHON=${PYTHON:-python3}
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then
    echo "Error: Python 3.12 or higher is required."
    echo "Current version: $("$PYTHON" --version 2>&1)"
    echo "Run with a newer Python, for example: PYTHON=python3.12 ./install.sh"
    exit 1
fi

echo "Creating virtual environment..."
"$PYTHON" -m venv venv

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "Error: Could not find virtual environment activation script."
    exit 1
fi

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setting up directory structure..."
python scripts/setup_directories.py

echo "Installation complete!"
echo
echo "To activate the virtual environment:"
echo "  On Linux/Mac: source venv/bin/activate"
echo "  On Windows: venv\\Scripts\\activate"
echo
echo "To run the API server:"
echo "  ./run_pipeline.sh --start-api"
echo
echo "To run the dashboard:"
echo "  ./run_pipeline.sh --start-dashboard"
echo

# Make the run_pipeline.sh executable
chmod +x run_pipeline.sh 