"""
Simple memory implementation for the Agentic AI platform.

This module provides a simple implementation of a memory component for the Agentic AI platform.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from ..interfaces.agent import Memory
from ..interfaces.base import Observer


class SimpleMemory(Memory):
    """
    Simple memory implementation for the Agentic AI platform.
    
    This class provides a simple in-memory implementation of a memory component.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the simple memory.
        
        Args:
            config: Configuration dictionary for the memory.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.items = {}
        self.observers = {}
        self.initialized = False
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the memory with the given configuration.
        
        Args:
            config: Configuration dictionary for the memory.
        
        Returns:
            True if initialization was successful, False otherwise.
        """
        try:
            self.config.update(config)
            self.initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize memory: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the memory.
        
        Returns:
            Dictionary containing information about the memory.
        """
        return {
            'type': self.__class__.__name__,
            'item_count': len(self.items),
            'initialized': self.initialized
        }
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the memory.
        
        Returns:
            Dictionary containing the current configuration.
        """
        return self.config
    
    def set_config(self, config: Dict[str, Any]) -> bool:
        """
        Set the configuration of the memory.
        
        Args:
            config: Dictionary containing the configuration to set.
        
        Returns:
            True if the configuration was set successfully, False otherwise.
        """
        try:
            self.config.update(config)
            return True
        except Exception as e:
            self.logger.error(f"Failed to set memory configuration: {e}")
            return False
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration of the memory.
        
        Returns:
            Dictionary containing the default configuration.
        """
        return {
            'max_items': 1000
        }
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the memory.
        
        Returns:
            Dictionary containing the current state of the memory.
        """
        return {
            'config': self.config,
            'items': self.items
        }
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        """
        Set the state of the memory.
        
        Args:
            state: Dictionary containing the state to set.
        
        Returns:
            True if the state was set successfully, False otherwise.
        """
        try:
            if 'config' in state:
                self.config.update(state['config'])
            
            if 'items' in state:
                self.items = state['items']
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to set memory state: {e}")
            return False
    
    def add_observer(self, observer: Observer, event_type: Optional[str] = None) -> bool:
        """
        Add an observer to this memory.
        
        Args:
            observer: The observer to add.
            event_type: The type of event to observe, or None to observe all events.
        
        Returns:
            True if the observer was added successfully, False otherwise.
        """
        try:
            if event_type not in self.observers:
                self.observers[event_type] = []
            
            if observer not in self.observers[event_type]:
                self.observers[event_type].append(observer)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to add observer to memory: {e}")
            return False
    
    def remove_observer(self, observer: Observer, event_type: Optional[str] = None) -> bool:
        """
        Remove an observer from this memory.
        
        Args:
            observer: The observer to remove.
            event_type: The type of event to stop observing, or None to stop observing all events.
        
        Returns:
            True if the observer was removed successfully, False otherwise.
        """
        try:
            if event_type is None:
                # Remove the observer from all event types
                for event_observers in self.observers.values():
                    if observer in event_observers:
                        event_observers.remove(observer)
            elif event_type in self.observers:
                # Remove the observer from the specified event type
                if observer in self.observers[event_type]:
                    self.observers[event_type].remove(observer)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to remove observer from memory: {e}")
            return False
    
    def notify_observers(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Notify all observers of an event.
        
        Args:
            event_type: The type of event that occurred.
            event_data: Dictionary containing data about the event.
        """
        try:
            # Notify observers of the specific event type
            if event_type in self.observers:
                for observer in self.observers[event_type]:
                    observer.update(self, event_type, event_data)
            
            # Notify observers of all events
            if None in self.observers:
                for observer in self.observers[None]:
                    observer.update(self, event_type, event_data)
        except Exception as e:
            self.logger.error(f"Failed to notify observers: {e}")
    
    def add(self, item: Dict[str, Any]) -> bool:
        """
        Add an item to memory.
        
        Args:
            item: The item to add to memory.
        
        Returns:
            True if the item was added successfully, False otherwise.
        """
        try:
            # Check if we've reached the maximum number of items
            max_items = self.config.get('max_items', 1000)
            if len(self.items) >= max_items:
                # Remove the oldest item
                oldest_id = None
                oldest_time = float('inf')
                
                for item_id, item_data in self.items.items():
                    if 'timestamp' in item_data and item_data['timestamp'] < oldest_time:
                        oldest_id = item_id
                        oldest_time = item_data['timestamp']
                
                if oldest_id:
                    del self.items[oldest_id]
            
            # Generate an ID for the item if it doesn't have one
            if 'id' not in item:
                item_id = str(uuid.uuid4())
                item['id'] = item_id
            else:
                item_id = item['id']
            
            # Add a timestamp to the item if it doesn't have one
            if 'timestamp' not in item:
                item['timestamp'] = time.time()
            
            # Add the item to memory
            self.items[item_id] = item
            
            # Notify observers
            self.notify_observers('add', {'item': item})
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to add item to memory: {e}")
            return False
    
    def get(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get items from memory that match a query.
        
        Args:
            query: The query to match against memory items.
        
        Returns:
            List of items that match the query.
        """
        try:
            # If the query is empty, return all items
            if not query:
                return list(self.items.values())
            
            # If the query has an ID, return the item with that ID
            if 'id' in query:
                item_id = query['id']
                if item_id in self.items:
                    return [self.items[item_id]]
                else:
                    return []
            
            # Otherwise, filter items based on the query
            results = []
            
            for item in self.items.values():
                match = True
                
                for key, value in query.items():
                    if key not in item or item[key] != value:
                        match = False
                        break
                
                if match:
                    results.append(item)
            
            return results
        except Exception as e:
            self.logger.error(f"Failed to get items from memory: {e}")
            return []
    
    def update(self, item_id: str, item: Dict[str, Any]) -> bool:
        """
        Update an item in memory.
        
        Args:
            item_id: The ID of the item to update.
            item: The new item data.
        
        Returns:
            True if the item was updated successfully, False otherwise.
        """
        try:
            # Check if the item exists
            if item_id not in self.items:
                return False
            
            # Update the item
            self.items[item_id].update(item)
            
            # Make sure the ID is preserved
            self.items[item_id]['id'] = item_id
            
            # Notify observers
            self.notify_observers('update', {'item_id': item_id, 'item': self.items[item_id]})
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to update item in memory: {e}")
            return False
    
    def remove(self, item_id: str) -> bool:
        """
        Remove an item from memory.
        
        Args:
            item_id: The ID of the item to remove.
        
        Returns:
            True if the item was removed successfully, False otherwise.
        """
        try:
            # Check if the item exists
            if item_id not in self.items:
                return False
            
            # Remove the item
            del self.items[item_id]
            
            # Notify observers
            self.notify_observers('remove', {'item_id': item_id})
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to remove item from memory: {e}")
            return False
    
    def clear(self) -> bool:
        """
        Clear all items from memory.
        
        Returns:
            True if memory was cleared successfully, False otherwise.
        """
        try:
            # Clear all items
            self.items = {}
            
            # Notify observers
            self.notify_observers('clear', {})
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear memory: {e}")
            return False