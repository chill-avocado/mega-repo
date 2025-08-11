"""
User interface interfaces for the integrated system.

This module defines the interfaces for user interface components in the integrated system.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable

from .base import Component, Connectable, Configurable, Observable, Observer


class UserInterface(Component, Connectable, Configurable, Observable, Observer):
    """Interface for user interface components in the integrated system."""

    @abstractmethod
    def render(self) -> Any:
        """
        Render the user interface.

        Returns:
            The rendered user interface.
        """
        pass

    @abstractmethod
    def handle_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle user input.

        Args:
            input_data: Dictionary containing the user input.

        Returns:
            Dictionary containing the result of handling the input.
        """
        pass

    @abstractmethod
    def update(self, observable: Observable, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Update the user interface in response to an event.

        Args:
            observable: The component that generated the event.
            event_type: The type of event that occurred.
            event_data: Dictionary containing data about the event.
        """
        pass

    @abstractmethod
    def add_component(self, component: 'UIComponent') -> bool:
        """
        Add a component to this user interface.

        Args:
            component: The component to add.

        Returns:
            True if the component was added successfully, False otherwise.
        """
        pass

    @abstractmethod
    def remove_component(self, component: 'UIComponent') -> bool:
        """
        Remove a component from this user interface.

        Args:
            component: The component to remove.

        Returns:
            True if the component was removed successfully, False otherwise.
        """
        pass

    @abstractmethod
    def get_components(self) -> List['UIComponent']:
        """
        Get all components in this user interface.

        Returns:
            List of components in this user interface.
        """
        pass


class UIComponent(Component, Configurable, Observable):
    """Interface for user interface components in the integrated system."""

    @abstractmethod
    def render(self) -> Any:
        """
        Render the component.

        Returns:
            The rendered component.
        """
        pass

    @abstractmethod
    def handle_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an event.

        Args:
            event_type: The type of event to handle.
            event_data: Dictionary containing data about the event.

        Returns:
            Dictionary containing the result of handling the event.
        """
        pass

    @abstractmethod
    def get_parent(self) -> Optional['UIComponent']:
        """
        Get the parent component of this component.

        Returns:
            The parent component, or None if this component has no parent.
        """
        pass

    @abstractmethod
    def set_parent(self, parent: Optional['UIComponent']) -> bool:
        """
        Set the parent component of this component.

        Args:
            parent: The parent component, or None to remove the parent.

        Returns:
            True if the parent was set successfully, False otherwise.
        """
        pass


class Layout(UIComponent):
    """Interface for layout components in the integrated system."""

    @abstractmethod
    def add_child(self, child: UIComponent, position: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a child component to this layout.

        Args:
            child: The child component to add.
            position: Optional dictionary specifying the position of the child.

        Returns:
            True if the child was added successfully, False otherwise.
        """
        pass

    @abstractmethod
    def remove_child(self, child: UIComponent) -> bool:
        """
        Remove a child component from this layout.

        Args:
            child: The child component to remove.

        Returns:
            True if the child was removed successfully, False otherwise.
        """
        pass

    @abstractmethod
    def get_children(self) -> List[UIComponent]:
        """
        Get all child components of this layout.

        Returns:
            List of child components.
        """
        pass