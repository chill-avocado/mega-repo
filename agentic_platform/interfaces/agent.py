"""
Agent interfaces for the Agentic AI platform.

This module defines the interfaces for agent components in the Agentic AI platform.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from .base import Component, Configurable, Executable, Observable, Stateful


class Agent(Component, Configurable, Executable, Observable, Stateful):
    """Interface for agent components in the Agentic AI platform."""

    @abstractmethod
    def plan(self, goal: str) -> List[Dict[str, Any]]:
        """
        Create a plan to achieve a goal.

        Args:
            goal: The goal to achieve.

        Returns:
            List of steps to achieve the goal.
        """
        pass

    @abstractmethod
    def execute_plan(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a plan.

        Args:
            plan: The plan to execute.

        Returns:
            Dictionary containing the result of the plan execution.
        """
        pass

    @abstractmethod
    def add_tool(self, tool: 'Tool') -> bool:
        """
        Add a tool to this agent.

        Args:
            tool: The tool to add.

        Returns:
            True if the tool was added successfully, False otherwise.
        """
        pass

    @abstractmethod
    def remove_tool(self, tool: 'Tool') -> bool:
        """
        Remove a tool from this agent.

        Args:
            tool: The tool to remove.

        Returns:
            True if the tool was removed successfully, False otherwise.
        """
        pass

    @abstractmethod
    def get_tools(self) -> List['Tool']:
        """
        Get all tools available to this agent.

        Returns:
            List of tools available to this agent.
        """
        pass

    @abstractmethod
    def add_memory(self, memory: 'Memory') -> bool:
        """
        Add a memory to this agent.

        Args:
            memory: The memory to add.

        Returns:
            True if the memory was added successfully, False otherwise.
        """
        pass

    @abstractmethod
    def remove_memory(self, memory: 'Memory') -> bool:
        """
        Remove a memory from this agent.

        Args:
            memory: The memory to remove.

        Returns:
            True if the memory was removed successfully, False otherwise.
        """
        pass

    @abstractmethod
    def get_memories(self) -> List['Memory']:
        """
        Get all memories available to this agent.

        Returns:
            List of memories available to this agent.
        """
        pass

    @abstractmethod
    def run(self, goal: str) -> Dict[str, Any]:
        """
        Run the agent to achieve a goal.

        This method combines planning and execution.

        Args:
            goal: The goal to achieve.

        Returns:
            Dictionary containing the result of achieving the goal.
        """
        pass


class Tool(Component, Configurable, Executable, Stateful):
    """Interface for tool components in the Agentic AI platform."""

    @abstractmethod
    def get_description(self) -> str:
        """
        Get a description of this tool.

        Returns:
            Description of this tool.
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the parameters required by this tool.

        Returns:
            Dictionary mapping parameter names to parameter specifications.
        """
        pass


class Memory(Component, Configurable, Observable, Stateful):
    """Interface for memory components in the Agentic AI platform."""

    @abstractmethod
    def add(self, item: Dict[str, Any]) -> bool:
        """
        Add an item to memory.

        Args:
            item: The item to add to memory.

        Returns:
            True if the item was added successfully, False otherwise.
        """
        pass

    @abstractmethod
    def get(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get items from memory that match a query.

        Args:
            query: The query to match against memory items.

        Returns:
            List of items that match the query.
        """
        pass

    @abstractmethod
    def update(self, item_id: str, item: Dict[str, Any]) -> bool:
        """
        Update an item in memory.

        Args:
            item_id: The ID of the item to update.
            item: The new item data.

        Returns:
            True if the item was updated successfully, False otherwise.
        """
        pass

    @abstractmethod
    def remove(self, item_id: str) -> bool:
        """
        Remove an item from memory.

        Args:
            item_id: The ID of the item to remove.

        Returns:
            True if the item was removed successfully, False otherwise.
        """
        pass

    @abstractmethod
    def clear(self) -> bool:
        """
        Clear all items from memory.

        Returns:
            True if memory was cleared successfully, False otherwise.
        """
        pass