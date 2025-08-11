"""
Logging utilities for the AGI system.

This module provides logging utilities for the AGI system.
"""

import logging
import sys
from typing import Optional


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """
    Set up logging for the AGI system.
    
    Args:
        level: Logging level.
        log_file: Optional path to a log file.
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


class ComponentLogger:
    """Logger for AGI system components."""
    
    def __init__(self, component_name: str):
        """
        Initialize the component logger.
        
        Args:
            component_name: Name of the component.
        """
        self.logger = logging.getLogger(component_name)
    
    def debug(self, message: str):
        """
        Log a debug message.
        
        Args:
            message: The message to log.
        """
        self.logger.debug(message)
    
    def info(self, message: str):
        """
        Log an info message.
        
        Args:
            message: The message to log.
        """
        self.logger.info(message)
    
    def warning(self, message: str):
        """
        Log a warning message.
        
        Args:
            message: The message to log.
        """
        self.logger.warning(message)
    
    def error(self, message: str):
        """
        Log an error message.
        
        Args:
            message: The message to log.
        """
        self.logger.error(message)
    
    def critical(self, message: str):
        """
        Log a critical message.
        
        Args:
            message: The message to log.
        """
        self.logger.critical(message)