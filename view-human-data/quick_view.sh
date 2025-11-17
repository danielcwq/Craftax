#!/bin/bash
# Quick trajectory viewer - runs the Python script with helpful error handling

echo "🔍 Craftax Trajectory Viewer"
echo "=============================="
echo ""

# Check if play_data directory exists
if [ ! -d "../play_data" ]; then
    echo "❌ Error: play_data directory not found"
    echo "   Make sure you've saved at least one trajectory with:"
    echo "   play_craftax --save_trajectories"
    exit 1
fi

# Try to run with python3
python3 view_trajectory.py "$@" 2>&1 | tee /tmp/craftax_viewer_error.log

# Check if it failed due to missing modules
if grep -q "ModuleNotFoundError.*jax\|ModuleNotFoundError.*craftax" /tmp/craftax_viewer_error.log 2>/dev/null; then
    echo ""
    echo "=============================="
    echo "⚠️  Missing Dependencies"
    echo "=============================="
    echo ""
    echo "The script needs to run in the same Python environment where Craftax is installed."
    echo ""
    echo "Try one of these:"
    echo ""
    echo "1. If you're using the system Python with pip:"
    echo "   python3 view_trajectory.py"
    echo ""
    echo "2. If you're using a conda environment:"
    echo "   conda activate <your_craftax_env>"
    echo "   python view_trajectory.py"
    echo ""
    echo "3. If you're using a virtual environment:"
    echo "   source <path_to_venv>/bin/activate"
    echo "   python view_trajectory.py"
    echo ""
    echo "4. Alternative: Use the notebook viewer (doesn't need JAX)"
    echo "   Check: simple_viewer.py"
    echo ""
fi

rm -f /tmp/craftax_viewer_error.log

