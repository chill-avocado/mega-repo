"""
Cognitive systems interfaces for the integrated system.

This module defines the interfaces for cognitive system components in the integrated system.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from .base import Component, Configurable, Observable


class CognitiveSystem(Component, Configurable, Observable):
    """Interface for cognitive system components in the integrated system."""

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data.

        Args:
            input_data: Dictionary containing the input data to process.

        Returns:
            Dictionary containing the processed data.
        """
        pass

    @abstractmethod
    def learn(self, data: Dict[str, Any]) -> bool:
        """
        Learn from data.

        Args:
            data: Dictionary containing the data to learn from.

        Returns:
            True if learning was successful, False otherwise.
        """
        pass

    @abstractmethod
    def reason(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reason about a query.

        Args:
            query: Dictionary containing the query to reason about.

        Returns:
            Dictionary containing the reasoning result.
        """
        pass

    @abstractmethod
    def add_knowledge(self, knowledge: Dict[str, Any]) -> bool:
        """
        Add knowledge to the cognitive system.

        Args:
            knowledge: Dictionary containing the knowledge to add.

        Returns:
            True if the knowledge was added successfully, False otherwise.
        """
        pass

    @abstractmethod
    def get_knowledge(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get knowledge from the cognitive system that matches a query.

        Args:
            query: Dictionary containing the query to match against knowledge.

        Returns:
            List of dictionaries containing knowledge that matches the query.
        """
        pass

    @abstractmethod
    def update_knowledge(self, knowledge_id: str, knowledge: Dict[str, Any]) -> bool:
        """
        Update knowledge in the cognitive system.

        Args:
            knowledge_id: The ID of the knowledge to update.
            knowledge: Dictionary containing the new knowledge.

        Returns:
            True if the knowledge was updated successfully, False otherwise.
        """
        pass

    @abstractmethod
    def remove_knowledge(self, knowledge_id: str) -> bool:
        """
        Remove knowledge from the cognitive system.

        Args:
            knowledge_id: The ID of the knowledge to remove.

        Returns:
            True if the knowledge was removed successfully, False otherwise.
        """
        pass