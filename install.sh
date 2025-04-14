#!/bin/bash
# ByteEats Vector Similarity Search Installation Script

echo "ByteEats Vector Similarity Search Installation"
echo "=============================================="
echo

# Check for Python 3.8+
python_version=$(python3 --version 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\2/')
if [ "$python_version" -lt 38 ]; then
    echo "Error: Python 3.8 or higher is required."
    echo "Current version: $(python3 --version)"
    exit 1
fi

echo "Creating virtual environment..."
python3 -m venv venv

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