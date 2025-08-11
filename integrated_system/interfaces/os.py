"""
OS interaction interfaces for the integrated system.

This module defines the interfaces for OS interaction components in the integrated system.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple

from .base import Component, Configurable, Executable


class OSInteraction(Component, Configurable, Executable):
    """Interface for OS interaction components in the integrated system."""

    @abstractmethod
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Capture the screen or a region of the screen.

        Args:
            region: Optional tuple (x, y, width, height) specifying the region to capture.

        Returns:
            Dictionary containing the captured screen image and metadata.
        """
        pass

    @abstractmethod
    def control_keyboard(self, action: str, data: Dict[str, Any]) -> bool:
        """
        Control the keyboard.

        Args:
            action: The action to perform (e.g., 'press', 'release', 'type').
            data: Dictionary containing data for the action.

        Returns:
            True if the action was performed successfully, False otherwise.
        """
        pass

    @abstractmethod
    def control_mouse(self, action: str, data: Dict[str, Any]) -> bool:
        """
        Control the mouse.

        Args:
            action: The action to perform (e.g., 'move', 'click', 'drag').
            data: Dictionary containing data for the action.

        Returns:
            True if the action was performed successfully, False otherwise.
        """
        pass

    @abstractmethod
    def execute_command(self, command: str, args: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute a command on the operating system.

        Args:
            command: The command to execute.
            args: Optional list of arguments for the command.

        Returns:
            Dictionary containing the result of the command execution.
        """
        pass

    @abstractmethod
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the operating system.

        Returns:
            Dictionary containing information about the operating system.
        """
        pass

    @abstractmethod
    def get_process_info(self, process_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get information about a process or all processes.

        Args:
            process_id: Optional ID of the process to get information about.

        Returns:
            Dictionary containing information about the process or processes.
        """
        pass

    @abstractmethod
    def get_file_info(self, path: str) -> Dict[str, Any]:
        """
        Get information about a file.

        Args:
            path: Path to the file.

        Returns:
            Dictionary containing information about the file.
        """
        pass

    @abstractmethod
    def read_file(self, path: str) -> Dict[str, Any]:
        """
        Read a file.

        Args:
            path: Path to the file.

        Returns:
            Dictionary containing the file contents and metadata.
        """
        pass

    @abstractmethod
    def write_file(self, path: str, content: Any) -> bool:
        """
        Write to a file.

        Args:
            path: Path to the file.
            content: Content to write to the file.

        Returns:
            True if the file was written successfully, False otherwise.
        """
        pass