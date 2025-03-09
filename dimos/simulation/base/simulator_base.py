from typing import Optional
from abc import ABC, abstractmethod

class SimulatorBase(ABC):
    """Base class for simulator implementations."""
    
    @abstractmethod
    def __init__(self, headless: bool = True, open_usd: Optional[str] = None):
        """Initialize the simulator.
        
        Args:
            headless (bool): Whether to run in headless mode
            open_usd (Optional[str]): USD file to open on startup
        """
        self.headless = headless
        self.open_usd = open_usd
        self.stage = None
        
    @abstractmethod
    def get_stage(self):
        """Get the current stage."""
        pass
    
    @abstractmethod
    def close(self):
        """Close the simulation."""
        pass 