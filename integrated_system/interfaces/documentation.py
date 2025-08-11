"""
Documentation interfaces for the integrated system.

This module defines the interfaces for documentation components in the integrated system.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from .base import Component, Configurable


class Documentation(Component, Configurable):
    """Interface for documentation components in the integrated system."""

    @abstractmethod
    def get_documentation(self, component: Optional[Any] = None) -> Dict[str, Any]:
        """
        Get documentation for a component or the entire system.

        Args:
            component: Optional component to get documentation for.

        Returns:
            Dictionary containing the documentation.
        """
        pass

    @abstractmethod
    def search(self, query: str, options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search the documentation.

        Args:
            query: The search query.
            options: Optional dictionary containing search options.

        Returns:
            List of dictionaries containing search results.
        """
        pass

    @abstractmethod
    def generate_documentation(self, component: Any, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate documentation for a component.

        Args:
            component: The component to generate documentation for.
            options: Optional dictionary containing generation options.

        Returns:
            Dictionary containing the generated documentation.
        """
        pass

    @abstractmethod
    def add_documentation(self, component: Any, documentation: Dict[str, Any]) -> bool:
        """
        Add documentation for a component.

        Args:
            component: The component to add documentation for.
            documentation: Dictionary containing the documentation to add.

        Returns:
            True if the documentation was added successfully, False otherwise.
        """
        pass

    @abstractmethod
    def update_documentation(self, component: Any, documentation: Dict[str, Any]) -> bool:
        """
        Update documentation for a component.

        Args:
            component: The component to update documentation for.
            documentation: Dictionary containing the documentation to update.

        Returns:
            True if the documentation was updated successfully, False otherwise.
        """
        pass

    @abstractmethod
    def remove_documentation(self, component: Any) -> bool:
        """
        Remove documentation for a component.

        Args:
            component: The component to remove documentation for.

        Returns:
            True if the documentation was removed successfully, False otherwise.
        """
        pass

    @abstractmethod
    def export_documentation(self, format: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Export documentation in a specific format.

        Args:
            format: The format to export documentation in.
            options: Optional dictionary containing export options.

        Returns:
            Dictionary containing the exported documentation.
        """
        pass