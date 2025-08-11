"""
Core system for the Agentic AI platform.

This module defines the core system that integrates all components.
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from ..interfaces.base import Component, Observer
from ..interfaces.agent import Agent, Tool, Memory


class AgenticSystem:
    """
    Core system for the Agentic AI platform.

    This class provides a unified interface for accessing and using all components
    in the Agentic AI platform.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agentic system.

        Args:
            config: Optional configuration dictionary for the system.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.components = {}
        self.agents = {}
        self.tools = {}
        self.memories = {}

    def initialize(self) -> bool:
        """
        Initialize the agentic system.

        Returns:
            True if initialization was successful, False otherwise.
        """
        self.logger.info("Initializing agentic system")
        
        # Initialize all components
        for component_id, component in self.components.items():
            if not component.initialize(self.config.get(component_id, {})):
                self.logger.error(f"Failed to initialize component: {component_id}")
                return False
        
        self.logger.info("Agentic system initialized successfully")
        return True

    def add_component(self, component_id: str, component: Component) -> bool:
        """
        Add a component to the agentic system.

        Args:
            component_id: ID for the component.
            component: The component to add.

        Returns:
            True if the component was added successfully, False otherwise.
        """
        if component_id in self.components:
            self.logger.warning(f"Component with ID {component_id} already exists")
            return False
        
        self.components[component_id] = component
        
        # Add to specific collections based on type
        if isinstance(component, Agent):
            self.agents[component_id] = component
        elif isinstance(component, Tool):
            self.tools[component_id] = component
        elif isinstance(component, Memory):
            self.memories[component_id] = component
        
        self.logger.info(f"Added component: {component_id}")
        return True

    def remove_component(self, component_id: str) -> bool:
        """
        Remove a component from the agentic system.

        Args:
            component_id: ID of the component to remove.

        Returns:
            True if the component was removed successfully, False otherwise.
        """
        if component_id not in self.components:
            self.logger.warning(f"Component with ID {component_id} does not exist")
            return False
        
        component = self.components[component_id]
        
        # Remove from specific collections based on type
        if isinstance(component, Agent):
            if component_id in self.agents:
                del self.agents[component_id]
        elif isinstance(component, Tool):
            if component_id in self.tools:
                del self.tools[component_id]
        elif isinstance(component, Memory):
            if component_id in self.memories:
                del self.memories[component_id]
        
        del self.components[component_id]
        self.logger.info(f"Removed component: {component_id}")
        
        return True

    def get_component(self, component_id: str) -> Optional[Component]:
        """
        Get a component by its ID.

        Args:
            component_id: ID of the component to get.

        Returns:
            The component with the given ID, or None if no such component exists.
        """
        return self.components.get(component_id)

    def get_components_by_type(self, component_type: Type[Component]) -> List[Component]:
        """
        Get all components of a specific type.

        Args:
            component_type: The type of components to get.

        Returns:
            List of components of the specified type.
        """
        return [component for component in self.components.values() if isinstance(component, component_type)]

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """
        Get an agent by its ID.

        Args:
            agent_id: ID of the agent to get.

        Returns:
            The agent with the given ID, or None if no such agent exists.
        """
        return self.agents.get(agent_id)

    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """
        Get a tool by its ID.

        Args:
            tool_id: ID of the tool to get.

        Returns:
            The tool with the given ID, or None if no such tool exists.
        """
        return self.tools.get(tool_id)

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        Get a memory by its ID.

        Args:
            memory_id: ID of the memory to get.

        Returns:
            The memory with the given ID, or None if no such memory exists.
        """
        return self.memories.get(memory_id)

    def connect_agent_to_tool(self, agent_id: str, tool_id: str) -> bool:
        """
        Connect an agent to a tool.

        Args:
            agent_id: ID of the agent to connect.
            tool_id: ID of the tool to connect to.

        Returns:
            True if the connection was established successfully, False otherwise.
        """
        agent = self.get_agent(agent_id)
        tool = self.get_tool(tool_id)
        
        if not agent:
            self.logger.error(f"Agent with ID {agent_id} does not exist")
            return False
        
        if not tool:
            self.logger.error(f"Tool with ID {tool_id} does not exist")
            return False
        
        return agent.add_tool(tool)

    def connect_agent_to_memory(self, agent_id: str, memory_id: str) -> bool:
        """
        Connect an agent to a memory.

        Args:
            agent_id: ID of the agent to connect.
            memory_id: ID of the memory to connect to.

        Returns:
            True if the connection was established successfully, False otherwise.
        """
        agent = self.get_agent(agent_id)
        memory = self.get_memory(memory_id)
        
        if not agent:
            self.logger.error(f"Agent with ID {agent_id} does not exist")
            return False
        
        if not memory:
            self.logger.error(f"Memory with ID {memory_id} does not exist")
            return False
        
        return agent.add_memory(memory)

    def run_agent(self, agent_id: str, goal: str) -> Dict[str, Any]:
        """
        Run an agent to achieve a goal.

        Args:
            agent_id: ID of the agent to run.
            goal: The goal to achieve.

        Returns:
            Dictionary containing the result of achieving the goal.
        """
        agent = self.get_agent(agent_id)
        
        if not agent:
            self.logger.error(f"Agent with ID {agent_id} does not exist")
            return {'success': False, 'error': f"Agent with ID {agent_id} does not exist"}
        
        try:
            result = agent.run(goal)
            return result
        except Exception as e:
            self.logger.error(f"Error running agent: {e}")
            return {'success': False, 'error': str(e)}

    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task using the appropriate component.

        Args:
            task: Dictionary containing the task to execute.

        Returns:
            Dictionary containing the result of the task execution.
        """
        task_type = task.get('type')
        
        if not task_type:
            self.logger.error("Task has no type")
            return {'success': False, 'error': 'Task has no type'}
        
        # Find components that can execute this task
        components = []
        
        for component in self.components.values():
            if hasattr(component, 'can_execute') and callable(component.can_execute):
                if component.can_execute(task):
                    components.append(component)
        
        if not components:
            self.logger.error(f"No component can execute task of type {task_type}")
            return {'success': False, 'error': f"No component can execute task of type {task_type}"}
        
        # Use the first component that can execute the task
        component = components[0]
        
        try:
            result = component.execute(task)
            return result
        except Exception as e:
            self.logger.error(f"Error executing task: {e}")
            return {'success': False, 'error': str(e)}

    def shutdown(self) -> bool:
        """
        Shut down the agentic system.

        Returns:
            True if shutdown was successful, False otherwise.
        """
        self.logger.info("Shutting down agentic system")
        
        # Clear components
        self.components = {}
        self.agents = {}
        self.tools = {}
        self.memories = {}
        
        self.logger.info("Agentic system shut down successfully")
        return True