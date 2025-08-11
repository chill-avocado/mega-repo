"""
Component interfaces for the AGI system.

This module defines the base component interfaces for the AGI system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union


class Component(ABC):
    """Base interface for all components in the AGI system."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the component with the given configuration.
        
        Args:
            config: Configuration dictionary for the component.
        
        Returns:
            True if initialization was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the component.
        
        Returns:
            Dictionary containing the current state of the component.
        """
        pass
    
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> bool:
        """
        Set the state of the component.
        
        Args:
            state: Dictionary containing the state to set.
        
        Returns:
            True if the state was set successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def adjust(self, adjustment: Dict[str, Any]) -> bool:
        """
        Adjust the component based on meta-cognitive feedback.
        
        Args:
            adjustment: Dictionary containing the adjustment to apply.
        
        Returns:
            True if the adjustment was applied successfully, False otherwise.
        """
        pass


class ComponentRegistry:
    """Registry for components in the AGI system."""
    
    def __init__(self):
        """Initialize the component registry."""
        self.components = {}
        self.component_types = {}
    
    def register(self, name: str, component: Component, component_type: Optional[str] = None) -> bool:
        """
        Register a component with the registry.
        
        Args:
            name: Name of the component.
            component: The component to register.
            component_type: Optional type of the component.
        
        Returns:
            True if the component was registered successfully, False otherwise.
        """
        self.components[name] = component
        
        if component_type:
            if component_type not in self.component_types:
                self.component_types[component_type] = {}
            self.component_types[component_type][name] = component
        
        return True
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a component from the registry.
        
        Args:
            name: Name of the component to unregister.
        
        Returns:
            True if the component was unregistered successfully, False otherwise.
        """
        if name in self.components:
            component = self.components[name]
            del self.components[name]
            
            # Remove from component types
            for component_type, components in self.component_types.items():
                if name in components:
                    del components[name]
            
            return True
        
        return False
    
    def get(self, name: str) -> Optional[Component]:
        """
        Get a component by name.
        
        Args:
            name: Name of the component to get.
        
        Returns:
            The component with the given name, or None if no such component exists.
        """
        return self.components.get(name)
    
    def get_by_type(self, component_type: str) -> Dict[str, Component]:
        """
        Get all components of a specific type.
        
        Args:
            component_type: Type of components to get.
        
        Returns:
            Dictionary mapping component names to components of the specified type.
        """
        return self.component_types.get(component_type, {})
    
    def get_all(self) -> Dict[str, Component]:
        """
        Get all registered components.
        
        Returns:
            Dictionary mapping component names to components.
        """
        return self.components