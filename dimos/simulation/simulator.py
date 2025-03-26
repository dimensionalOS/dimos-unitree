# Copyright 2025 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaacsim import SimulationApp
from typing import Optional

class Simulator:
    """Wrapper class for Isaac Sim simulation."""
    
    def __init__(self, headless: bool = True, open_usd: Optional[str] = None):
        """Initialize the Isaac Sim simulation.
        
        Args:
            headless (bool): Whether to run in headless mode. Defaults to True.
            open_usd (Optional[str]): USD file to open on startup. Defaults to None.
        """
        self.app = SimulationApp({
            "headless": headless,
            "open_usd": open_usd
        })
        self.stage = None
        
    def get_stage(self):
        """Get the current USD stage."""
        import omni.usd
        self.stage = omni.usd.get_context().get_stage()
        return self.stage
    
    def close(self):
        """Close the simulation."""
        if self.app:
            self.app.close() 