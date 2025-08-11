"""
Integration interfaces for the integrated system.

This module defines the interfaces for integration components in the integrated system.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union, Type

from .base import Component, Connectable, Configurable, Observable, Observer


class Integration(Component, Connectable, Configurable, Observable, Observer):
    """Interface for integration components in the integrated system."""

    @abstractmethod
    def connect_components(self, component1: Component, component2: Component, connection_type: str) -> bool:
        """
        Connect two components.

        Args:
            component1: The first component to connect.
            component2: The second component to connect.
            connection_type: The type of connection to establish.

        Returns:
            True if the connection was established successfully, False otherwise.
        """
        pass

    @abstractmethod
    def disconnect_components(self, component1: Component, component2: Component) -> bool:
        """
        Disconnect two components.

        Args:
            component1: The first component to disconnect.
            component2: The second component to disconnect.

        Returns:
            True if the disconnection was successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_connected_components(self, component: Component) -> List[Component]:
        """
        Get all components connected to a component.

        Args:
            component: The component to get connections for.

        Returns:
            List of connected components.
        """
        pass

    @abstractmethod
    def register_component(self, component: Component) -> bool:
        """
        Register a component with this integration component.

        Args:
            component: The component to register.

        Returns:
            True if the component was registered successfully, False otherwise.
        """
        pass

    @abstractmethod
    def unregister_component(self, component: Component) -> bool:
        """
        Unregister a component from this integration component.

        Args:
            component: The component to unregister.

        Returns:
            True if the component was unregistered successfully, False otherwise.
        """
        pass

    @abstractmethod
    def get_registered_components(self) -> List[Component]:
        """
        Get all components registered with this integration component.

        Returns:
            List of registered components.
        """
        pass

    @abstractmethod
    def get_component_by_id(self, component_id: str) -> Optional[Component]:
        """
        Get a component by its ID.

        Args:
            component_id: The ID of the component to get.

        Returns:
            The component with the given ID, or None if no such component exists.
        """
        pass

    @abstractmethod
    def get_components_by_type(self, component_type: Type[Component]) -> List[Component]:
        """
        Get all components of a specific type.

        Args:
            component_type: The type of components to get.

        Returns:
            List of components of the specified type.
        """
        pass

    @abstractmethod
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a message.

        Args:
            message: Dictionary containing the message to process.

        Returns:
            Dictionary containing the processed message.
        """
        pass

    @abstractmethod
    def send_message(self, source: Component, target: Component, message: Dict[str, Any]) -> bool:
        """
        Send a message from one component to another.

        Args:
            source: The component sending the message.
            target: The component receiving the message.
            message: Dictionary containing the message to send.

        Returns:
            True if the message was sent successfully, False otherwise.
        """
        pass

    @abstractmethod
    def broadcast_message(self, source: Component, message: Dict[str, Any]) -> bool:
        """
        Broadcast a message to all registered components.

        Args:
            source: The component broadcasting the message.
            message: Dictionary containing the message to broadcast.

        Returns:
            True if the message was broadcast successfully, False otherwise.
        """
        pass