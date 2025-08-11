"""
Memory interfaces for the AGI system.

This module defines the memory interfaces for the AGI system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .component import Component


class Memory(Component):
    """Base interface for memory components in the AGI system."""
    
    @abstractmethod
    def store(self, key: str, value: Any) -> bool:
        """
        Store a value in memory.
        
        Args:
            key: Key to store the value under.
            value: Value to store.
        
        Returns:
            True if the value was stored successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def retrieve(self, key: str) -> Any:
        """
        Retrieve a value from memory.
        
        Args:
            key: Key to retrieve the value for.
        
        Returns:
            The retrieved value, or None if the key does not exist.
        """
        pass
    
    @abstractmethod
    def remove(self, key: str) -> bool:
        """
        Remove a value from memory.
        
        Args:
            key: Key to remove the value for.
        
        Returns:
            True if the value was removed successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """
        Clear all values from memory.
        
        Returns:
            True if memory was cleared successfully, False otherwise.
        """
        pass


class WorkingMemory(Memory):
    """Interface for working memory components in the AGI system."""
    
    @abstractmethod
    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update working memory with new data.
        
        Args:
            data: Dictionary containing data to update working memory with.
        
        Returns:
            Dictionary containing the result of the update operation.
        """
        pass
    
    @abstractmethod
    def get_all(self) -> Dict[str, Any]:
        """
        Get all values in working memory.
        
        Returns:
            Dictionary mapping keys to values in working memory.
        """
        pass
    
    @abstractmethod
    def get_capacity(self) -> int:
        """
        Get the capacity of working memory.
        
        Returns:
            The capacity of working memory (maximum number of items).
        """
        pass
    
    @abstractmethod
    def set_capacity(self, capacity: int) -> bool:
        """
        Set the capacity of working memory.
        
        Args:
            capacity: The capacity to set (maximum number of items).
        
        Returns:
            True if the capacity was set successfully, False otherwise.
        """
        pass


class LongTermMemory(Memory):
    """Interface for long-term memory components in the AGI system."""
    
    @abstractmethod
    def store(self, data: Dict[str, Any]) -> bool:
        """
        Store data in long-term memory.
        
        Args:
            data: Dictionary containing data to store in long-term memory.
        
        Returns:
            True if the data was stored successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve data from long-term memory that matches a query.
        
        Args:
            query: Dictionary containing the query to match against memory items.
        
        Returns:
            Dictionary containing the retrieved data.
        """
        pass
    
    @abstractmethod
    def update(self, item_id: str, data: Dict[str, Any]) -> bool:
        """
        Update an item in long-term memory.
        
        Args:
            item_id: ID of the item to update.
            data: Dictionary containing the new data for the item.
        
        Returns:
            True if the item was updated successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def search(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search for items in long-term memory that match a query.
        
        Args:
            query: Dictionary containing the query to match against memory items.
        
        Returns:
            List of items that match the query.
        """
        pass
    
    @abstractmethod
    def get_semantic_network(self) -> Dict[str, Any]:
        """
        Get the semantic network of concepts in long-term memory.
        
        Returns:
            Dictionary representing the semantic network.
        """
        pass
    
    @abstractmethod
    def get_episodic_memory(self) -> List[Dict[str, Any]]:
        """
        Get episodic memories from long-term memory.
        
        Returns:
            List of episodic memories.
        """
        pass
    
    @abstractmethod
    def get_procedural_memory(self) -> List[Dict[str, Any]]:
        """
        Get procedural memories from long-term memory.
        
        Returns:
            List of procedural memories.
        """
        pass