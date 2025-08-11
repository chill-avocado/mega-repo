"""
Base interfaces for the Agentic AI platform.

This module defines the base interfaces for components in the Agentic AI platform.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class Component(ABC):
    """Base interface for all components in the Agentic AI platform."""

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
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the component.

        Returns:
            Dictionary containing information about the component.
        """
        pass


class Configurable(ABC):
    """Interface for components that can be configured."""

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the component.

        Returns:
            Dictionary containing the current configuration.
        """
        pass

    @abstractmethod
    def set_config(self, config: Dict[str, Any]) -> bool:
        """
        Set the configuration of the component.

        Args:
            config: Dictionary containing the configuration to set.

        Returns:
            True if the configuration was set successfully, False otherwise.
        """
        pass

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration of the component.

        Returns:
            Dictionary containing the default configuration.
        """
        pass


class Stateful(ABC):
    """Interface for components that maintain state."""

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


class Executable(ABC):
    """Interface for components that can execute tasks."""

    @abstractmethod
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task.

        Args:
            task: Dictionary containing the task to execute.

        Returns:
            Dictionary containing the result of the task execution.
        """
        pass

    @abstractmethod
    def can_execute(self, task: Dict[str, Any]) -> bool:
        """
        Check if this component can execute the given task.

        Args:
            task: Dictionary containing the task to check.

        Returns:
            True if this component can execute the task, False otherwise.
        """
        pass


class Observable(ABC):
    """Interface for components that can be observed."""

    @abstractmethod
    def add_observer(self, observer: 'Observer', event_type: Optional[str] = None) -> bool:
        """
        Add an observer to this component.

        Args:
            observer: The observer to add.
            event_type: The type of event to observe, or None to observe all events.

        Returns:
            True if the observer was added successfully, False otherwise.
        """
        pass

    @abstractmethod
    def remove_observer(self, observer: 'Observer', event_type: Optional[str] = None) -> bool:
        """
        Remove an observer from this component.

        Args:
            observer: The observer to remove.
            event_type: The type of event to stop observing, or None to stop observing all events.

        Returns:
            True if the observer was removed successfully, False otherwise.
        """
        pass

    @abstractmethod
    def notify_observers(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Notify all observers of an event.

        Args:
            event_type: The type of event that occurred.
            event_data: Dictionary containing data about the event.
        """
        pass


class Observer(ABC):
    """Interface for components that can observe other components."""

    @abstractmethod
    def update(self, observable: Observable, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Update this observer with an event from an observable component.

        Args:
            observable: The component that generated the event.
            event_type: The type of event that occurred.
            event_data: Dictionary containing data about the event.
        """
        pass