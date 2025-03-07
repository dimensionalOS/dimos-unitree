"""
Robot Web Interface wrapper for DIMOS.
Provides a clean interface to the dimensional-website FastAPI server.
"""

import os
import sys

# Add the dimensional-website api directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "dimensional-website/api"))

from server import FastAPIServer

class RobotWebInterface(FastAPIServer):
    """Wrapper class for the dimensional-website FastAPI server."""
    
    def __init__(self, port=5555, **streams):
        super().__init__(
            dev_name="Robot Web Interface",
            edge_type="Bidirectional",
            host="0.0.0.0",
            port=port,
            **streams
        ) 