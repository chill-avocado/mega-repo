"""
Memory adapter for integrating external memory components into the Agentic AI platform.

This module provides an adapter for integrating external memory components into the
Agentic AI platform.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from ...interfaces.agent import Memory
from ...interfaces.base import Observable, Observer
from ..base import BaseAdapter


class MemoryAdapter(Memory):
    """
    Memory adapter for integrating external memory components into the Agentic AI platform.
    
    This class provides an adapter that wraps an external memory component and exposes
    it through the Memory interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the memory adapter.
        
        Args:
            config: Configuration dictionary for the adapter.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.observers = {}
        self.initialized = False
        
        # Get the component path and class from the config
        self.component_path = self.config.get('component_path')
        self.component_class = self.config.get('component_class')
        
        # Create the adapter
        self.adapter = BaseAdapter(self.component_path, self.component_class, self.config.get('component_config'))
        
        # Load the component
        self.component = self.adapter.load_component()
    
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
            
            # Initialize the component if it has an initialize method
            if hasattr(self.component, 'initialize') and callable(self.component.initialize):
                self.component.initialize(config)
            
            self.initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize memory adapter: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the memory.
        
        Returns:
            Dictionary containing information about the memory.
        """
        return {
            'type': self.__class__.__name__,
            'component_path': self.component_path,
            'component_class': self.component_class,
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
            
            # Set the component config if it has a set_config method
            if hasattr(self.component, 'set_config') and callable(self.component.set_config):
                self.component.set_config(config)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to set memory adapter configuration: {e}")
            return False
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration of the memory.
        
        Returns:
            Dictionary containing the default configuration.
        """
        # Get the component default config if it has a get_default_config method
        if hasattr(self.component, 'get_default_config') and callable(self.component.get_default_config):
            return self.component.get_default_config()
        
        return {}
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the memory.
        
        Returns:
            Dictionary containing the current state of the memory.
        """
        # Get the component state if it has a get_state method
        if hasattr(self.component, 'get_state') and callable(self.component.get_state):
            return self.component.get_state()
        
        return {
            'config': self.config
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
            # Set the component state if it has a set_state method
            if hasattr(self.component, 'set_state') and callable(self.component.set_state):
                return self.component.set_state(state)
            
            if 'config' in state:
                self.config.update(state['config'])
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to set memory adapter state: {e}")
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
            # Add the observer to the component if it has an add_observer method
            if hasattr(self.component, 'add_observer') and callable(self.component.add_observer):
                return self.component.add_observer(observer, event_type)
            
            if event_type not in self.observers:
                self.observers[event_type] = []
            
            if observer not in self.observers[event_type]:
                self.observers[event_type].append(observer)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to add observer to memory adapter: {e}")
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
            # Remove the observer from the component if it has a remove_observer method
            if hasattr(self.component, 'remove_observer') and callable(self.component.remove_observer):
                return self.component.remove_observer(observer, event_type)
            
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
            self.logger.error(f"Failed to remove observer from memory adapter: {e}")
            return False
    
    def notify_observers(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Notify all observers of an event.
        
        Args:
            event_type: The type of event that occurred.
            event_data: Dictionary containing data about the event.
        """
        try:
            # Notify observers using the component if it has a notify_observers method
            if hasattr(self.component, 'notify_observers') and callable(self.component.notify_observers):
                self.component.notify_observers(event_type, event_data)
                return
            
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
            # Add the item to the component if it has an add method
            if hasattr(self.component, 'add') and callable(self.component.add):
                result = self.component.add(item)
                
                # Notify observers
                self.notify_observers('add', {'item': item})
                
                return result
            
            # Try to add the item to the component's items attribute
            if hasattr(self.component, 'items'):
                self.component.items.append(item)
                
                # Notify observers
                self.notify_observers('add', {'item': item})
                
                return True
            
            return False
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
            # Get items from the component if it has a get method
            if hasattr(self.component, 'get') and callable(self.component.get):
                return self.component.get(query)
            
            # Try to search the component's items attribute
            if hasattr(self.component, 'search') and callable(self.component.search):
                return self.component.search(query)
            
            # Try to get all items from the component
            if hasattr(self.component, 'get_all') and callable(self.component.get_all):
                items = self.component.get_all()
                
                # Filter items based on the query
                if not query:
                    return items
                
                results = []
                
                for item in items:
                    match = True
                    
                    for key, value in query.items():
                        if key not in item or item[key] != value:
                            match = False
                            break
                    
                    if match:
                        results.append(item)
                
                return results
            
            # Try to access the component's items attribute directly
            if hasattr(self.component, 'items'):
                items = self.component.items
                
                # Filter items based on the query
                if not query:
                    return items
                
                results = []
                
                for item in items:
                    match = True
                    
                    for key, value in query.items():
                        if key not in item or item[key] != value:
                            match = False
                            break
                    
                    if match:
                        results.append(item)
                
                return results
            
            return []
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
            # Update the item in the component if it has an update method
            if hasattr(self.component, 'update') and callable(self.component.update):
                result = self.component.update(item_id, item)
                
                # Notify observers
                self.notify_observers('update', {'item_id': item_id, 'item': item})
                
                return result
            
            # Try to update the item in the component's items attribute
            if hasattr(self.component, 'items'):
                for i, existing_item in enumerate(self.component.items):
                    if 'id' in existing_item and existing_item['id'] == item_id:
                        self.component.items[i].update(item)
                        
                        # Make sure the ID is preserved
                        self.component.items[i]['id'] = item_id
                        
                        # Notify observers
                        self.notify_observers('update', {'item_id': item_id, 'item': self.component.items[i]})
                        
                        return True
            
            return False
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
            # Remove the item from the component if it has a remove method
            if hasattr(self.component, 'remove') and callable(self.component.remove):
                result = self.component.remove(item_id)
                
                # Notify observers
                self.notify_observers('remove', {'item_id': item_id})
                
                return result
            
            # Try to remove the item from the component's items attribute
            if hasattr(self.component, 'items'):
                for i, existing_item in enumerate(self.component.items):
                    if 'id' in existing_item and existing_item['id'] == item_id:
                        del self.component.items[i]
                        
                        # Notify observers
                        self.notify_observers('remove', {'item_id': item_id})
                        
                        return True
            
            return False
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
            # Clear the component if it has a clear method
            if hasattr(self.component, 'clear') and callable(self.component.clear):
                result = self.component.clear()
                
                # Notify observers
                self.notify_observers('clear', {})
                
                return result
            
            # Try to clear the component's items attribute
            if hasattr(self.component, 'items'):
                self.component.items = []
                
                # Notify observers
                self.notify_observers('clear', {})
                
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"Failed to clear memory: {e}")
            return False