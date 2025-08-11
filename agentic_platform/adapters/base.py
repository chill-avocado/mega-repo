"""
Base adapter for integrating external components into the Agentic AI platform.

This module provides a base adapter for integrating external components into the
Agentic AI platform.
"""

import logging
import importlib
import sys
from typing import Any, Dict, List, Optional, Type, Union

from ..interfaces.base import Component


class BaseAdapter:
    """
    Base adapter for integrating external components into the Agentic AI platform.
    
    This class provides a base adapter that can be extended to create adapters for
    specific types of external components.
    """
    
    def __init__(self, component_path: str, component_class: str, config: Dict[str, Any] = None):
        """
        Initialize the base adapter.
        
        Args:
            component_path: Path to the component module.
            component_class: Name of the component class.
            config: Configuration dictionary for the adapter.
        """
        self.logger = logging.getLogger(__name__)
        self.component_path = component_path
        self.component_class = component_class
        self.config = config or {}
        self.component = None
    
    def load_component(self) -> Any:
        """
        Load the external component.
        
        Returns:
            The loaded component, or None if loading failed.
        """
        try:
            # Add the functions directory to the Python path if it's not already there
            if 'functions' not in sys.path:
                sys.path.append('functions')
            
            # Import the component module
            module = importlib.import_module(self.component_path)
            
            # Get the component class
            component_class = getattr(module, self.component_class)
            
            # Create an instance of the component class
            self.component = component_class(**self.config)
            
            return self.component
        except ImportError as e:
            self.logger.error(f"Failed to import component module {self.component_path}: {e}")
            return None
        except AttributeError as e:
            self.logger.error(f"Failed to get component class {self.component_class} from module {self.component_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load component: {e}")
            return None