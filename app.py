# HuggingFace Spaces entry point for AI Act Navigator
# This redirects to the main Streamlit app in src/ui/app.py

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment for proper imports
os.environ['PYTHONPATH'] = str(project_root)

# Import and run the main app
if __name__ == "__main__":
    from src.ui.app import main
    main()