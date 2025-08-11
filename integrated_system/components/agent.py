"""
Agent component adapters for the integrated system.

This module provides adapters for agent components in the integrated system.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from ..interfaces import Agent, Tool, Memory, Component


class AgentAdapter(Agent):
    """
    Adapter for agent components.

    This class adapts agent components from merged functions to the Agent interface.
    """

    def __init__(self, agent_instance: Any, config: Dict[str, Any]):
        """
        Initialize the agent adapter.

        Args:
            agent_instance: The agent instance to adapt.
            config: Configuration dictionary for the agent.
        """
        self.agent = agent_instance
        self.config = config
        self.tools = []
        self.memories = []
        self.connections = []
        self.observers = {}
        self.logger = logging.getLogger(__name__)

    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the agent with the given configuration.

        Args:
            config: Configuration dictionary for the agent.

        Returns:
            True if initialization was successful, False otherwise.
        """
        try:
            self.config.update(config)
            
            # Initialize the agent if it has an initialize method
            if hasattr(self.agent, 'initialize') and callable(self.agent.initialize):
                return self.agent.initialize(config)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize agent: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the agent.

        Returns:
            Dictionary containing information about the agent.
        """
        info = {
            'type': type(self.agent).__name__,
            'tools': len(self.tools),
            'memories': len(self.memories),
            'connections': len(self.connections),
        }
        
        # Add additional info if the agent has a get_info method
        if hasattr(self.agent, 'get_info') and callable(self.agent.get_info):
            info.update(self.agent.get_info())
        
        return info

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the agent.

        Returns:
            Dictionary containing the current state of the agent.
        """
        state = {
            'config': self.config,
            'tools': [tool.get_info() for tool in self.tools],
            'memories': [memory.get_info() for memory in self.memories],
            'connections': [{'id': id(connection), 'type': type(connection).__name__} for connection in self.connections],
        }
        
        # Add additional state if the agent has a get_state method
        if hasattr(self.agent, 'get_state') and callable(self.agent.get_state):
            state.update(self.agent.get_state())
        
        return state

    def set_state(self, state: Dict[str, Any]) -> bool:
        """
        Set the state of the agent.

        Args:
            state: Dictionary containing the state to set.

        Returns:
            True if the state was set successfully, False otherwise.
        """
        try:
            if 'config' in state:
                self.config.update(state['config'])
            
            # Set additional state if the agent has a set_state method
            if hasattr(self.agent, 'set_state') and callable(self.agent.set_state):
                return self.agent.set_state(state)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to set agent state: {e}")
            return False

    def connect(self, component: Component, connection_type: str) -> bool:
        """
        Connect this agent to another component.

        Args:
            component: The component to connect to.
            connection_type: The type of connection to establish.

        Returns:
            True if the connection was established successfully, False otherwise.
        """
        try:
            # Connect the agent if it has a connect method
            if hasattr(self.agent, 'connect') and callable(self.agent.connect):
                result = self.agent.connect(component, connection_type)
                if result:
                    self.connections.append(component)
                return result
            
            # Otherwise, just add the component to the connections list
            self.connections.append(component)
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect agent to component: {e}")
            return False

    def disconnect(self, component: Component) -> bool:
        """
        Disconnect this agent from another component.

        Args:
            component: The component to disconnect from.

        Returns:
            True if the disconnection was successful, False otherwise.
        """
        try:
            # Disconnect the agent if it has a disconnect method
            if hasattr(self.agent, 'disconnect') and callable(self.agent.disconnect):
                result = self.agent.disconnect(component)
                if result and component in self.connections:
                    self.connections.remove(component)
                return result
            
            # Otherwise, just remove the component from the connections list
            if component in self.connections:
                self.connections.remove(component)
            return True
        except Exception as e:
            self.logger.error(f"Failed to disconnect agent from component: {e}")
            return False

    def get_connections(self) -> List[Component]:
        """
        Get all components connected to this agent.

        Returns:
            List of connected components.
        """
        return self.connections

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the agent.

        Returns:
            Dictionary containing the current configuration.
        """
        return self.config

    def set_config(self, config: Dict[str, Any]) -> bool:
        """
        Set the configuration of the agent.

        Args:
            config: Dictionary containing the configuration to set.

        Returns:
            True if the configuration was set successfully, False otherwise.
        """
        try:
            self.config.update(config)
            
            # Set the configuration if the agent has a set_config method
            if hasattr(self.agent, 'set_config') and callable(self.agent.set_config):
                return self.agent.set_config(config)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to set agent configuration: {e}")
            return False

    def get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration of the agent.

        Returns:
            Dictionary containing the default configuration.
        """
        # Get the default configuration if the agent has a get_default_config method
        if hasattr(self.agent, 'get_default_config') and callable(self.agent.get_default_config):
            return self.agent.get_default_config()
        
        return {}

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task.

        Args:
            task: Dictionary containing the task to execute.

        Returns:
            Dictionary containing the result of the task execution.
        """
        try:
            # Execute the task if the agent has an execute method
            if hasattr(self.agent, 'execute') and callable(self.agent.execute):
                return self.agent.execute(task)
            
            # Otherwise, try to run the task as a goal
            if 'goal' in task:
                return self.run(task['goal'])
            
            return {'success': False, 'error': 'Agent does not support task execution'}
        except Exception as e:
            self.logger.error(f"Failed to execute task: {e}")
            return {'success': False, 'error': str(e)}

    def can_execute(self, task: Dict[str, Any]) -> bool:
        """
        Check if this agent can execute the given task.

        Args:
            task: Dictionary containing the task to check.

        Returns:
            True if this agent can execute the task, False otherwise.
        """
        # Check if the agent has a can_execute method
        if hasattr(self.agent, 'can_execute') and callable(self.agent.can_execute):
            return self.agent.can_execute(task)
        
        # Otherwise, check if the task has a goal
        return 'goal' in task

    def add_observer(self, observer: 'Observer', event_type: Optional[str] = None) -> bool:
        """
        Add an observer to this agent.

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
            self.logger.error(f"Failed to add observer to agent: {e}")
            return False

    def remove_observer(self, observer: 'Observer', event_type: Optional[str] = None) -> bool:
        """
        Remove an observer from this agent.

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
            self.logger.error(f"Failed to remove observer from agent: {e}")
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

    def plan(self, goal: str) -> List[Dict[str, Any]]:
        """
        Create a plan to achieve a goal.

        Args:
            goal: The goal to achieve.

        Returns:
            List of steps to achieve the goal.
        """
        try:
            # Plan if the agent has a plan method
            if hasattr(self.agent, 'plan') and callable(self.agent.plan):
                return self.agent.plan(goal)
            
            # Otherwise, return a simple plan with just the goal
            return [{'type': 'goal', 'goal': goal}]
        except Exception as e:
            self.logger.error(f"Failed to create plan: {e}")
            return []

    def execute_plan(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a plan.

        Args:
            plan: The plan to execute.

        Returns:
            Dictionary containing the result of the plan execution.
        """
        try:
            # Execute the plan if the agent has an execute_plan method
            if hasattr(self.agent, 'execute_plan') and callable(self.agent.execute_plan):
                return self.agent.execute_plan(plan)
            
            # Otherwise, execute each step in the plan
            results = []
            for step in plan:
                if 'type' in step and step['type'] == 'goal':
                    # Execute the goal if the agent has a run method
                    if hasattr(self.agent, 'run') and callable(self.agent.run):
                        result = self.agent.run(step['goal'])
                        results.append(result)
                else:
                    # Execute the step as a task
                    result = self.execute(step)
                    results.append(result)
            
            return {'success': True, 'results': results}
        except Exception as e:
            self.logger.error(f"Failed to execute plan: {e}")
            return {'success': False, 'error': str(e)}

    def add_tool(self, tool: 'Tool') -> bool:
        """
        Add a tool to this agent.

        Args:
            tool: The tool to add.

        Returns:
            True if the tool was added successfully, False otherwise.
        """
        try:
            # Add the tool if the agent has an add_tool method
            if hasattr(self.agent, 'add_tool') and callable(self.agent.add_tool):
                result = self.agent.add_tool(tool)
                if result:
                    self.tools.append(tool)
                return result
            
            # Otherwise, just add the tool to the tools list
            self.tools.append(tool)
            return True
        except Exception as e:
            self.logger.error(f"Failed to add tool to agent: {e}")
            return False

    def remove_tool(self, tool: 'Tool') -> bool:
        """
        Remove a tool from this agent.

        Args:
            tool: The tool to remove.

        Returns:
            True if the tool was removed successfully, False otherwise.
        """
        try:
            # Remove the tool if the agent has a remove_tool method
            if hasattr(self.agent, 'remove_tool') and callable(self.agent.remove_tool):
                result = self.agent.remove_tool(tool)
                if result and tool in self.tools:
                    self.tools.remove(tool)
                return result
            
            # Otherwise, just remove the tool from the tools list
            if tool in self.tools:
                self.tools.remove(tool)
            return True
        except Exception as e:
            self.logger.error(f"Failed to remove tool from agent: {e}")
            return False

    def get_tools(self) -> List['Tool']:
        """
        Get all tools available to this agent.

        Returns:
            List of tools available to this agent.
        """
        return self.tools

    def add_memory(self, memory: 'Memory') -> bool:
        """
        Add a memory to this agent.

        Args:
            memory: The memory to add.

        Returns:
            True if the memory was added successfully, False otherwise.
        """
        try:
            # Add the memory if the agent has an add_memory method
            if hasattr(self.agent, 'add_memory') and callable(self.agent.add_memory):
                result = self.agent.add_memory(memory)
                if result:
                    self.memories.append(memory)
                return result
            
            # Otherwise, just add the memory to the memories list
            self.memories.append(memory)
            return True
        except Exception as e:
            self.logger.error(f"Failed to add memory to agent: {e}")
            return False

    def remove_memory(self, memory: 'Memory') -> bool:
        """
        Remove a memory from this agent.

        Args:
            memory: The memory to remove.

        Returns:
            True if the memory was removed successfully, False otherwise.
        """
        try:
            # Remove the memory if the agent has a remove_memory method
            if hasattr(self.agent, 'remove_memory') and callable(self.agent.remove_memory):
                result = self.agent.remove_memory(memory)
                if result and memory in self.memories:
                    self.memories.remove(memory)
                return result
            
            # Otherwise, just remove the memory from the memories list
            if memory in self.memories:
                self.memories.remove(memory)
            return True
        except Exception as e:
            self.logger.error(f"Failed to remove memory from agent: {e}")
            return False

    def get_memories(self) -> List['Memory']:
        """
        Get all memories available to this agent.

        Returns:
            List of memories available to this agent.
        """
        return self.memories

    def run(self, goal: str) -> Dict[str, Any]:
        """
        Run the agent to achieve a goal.

        This method combines planning and execution.

        Args:
            goal: The goal to achieve.

        Returns:
            Dictionary containing the result of achieving the goal.
        """
        try:
            # Run if the agent has a run method
            if hasattr(self.agent, 'run') and callable(self.agent.run):
                return self.agent.run(goal)
            
            # Otherwise, create a plan and execute it
            plan = self.plan(goal)
            return self.execute_plan(plan)
        except Exception as e:
            self.logger.error(f"Failed to run agent: {e}")
            return {'success': False, 'error': str(e)}


class ToolAdapter(Tool):
    """
    Adapter for tool components.

    This class adapts tool components from merged functions to the Tool interface.
    """

    def __init__(self, tool_instance: Any, config: Dict[str, Any]):
        """
        Initialize the tool adapter.

        Args:
            tool_instance: The tool instance to adapt.
            config: Configuration dictionary for the tool.
        """
        self.tool = tool_instance
        self.config = config
        self.logger = logging.getLogger(__name__)

    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the tool with the given configuration.

        Args:
            config: Configuration dictionary for the tool.

        Returns:
            True if initialization was successful, False otherwise.
        """
        try:
            self.config.update(config)
            
            # Initialize the tool if it has an initialize method
            if hasattr(self.tool, 'initialize') and callable(self.tool.initialize):
                return self.tool.initialize(config)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize tool: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the tool.

        Returns:
            Dictionary containing information about the tool.
        """
        info = {
            'type': type(self.tool).__name__,
            'description': self.get_description(),
        }
        
        # Add additional info if the tool has a get_info method
        if hasattr(self.tool, 'get_info') and callable(self.tool.get_info):
            info.update(self.tool.get_info())
        
        return info

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the tool.

        Returns:
            Dictionary containing the current state of the tool.
        """
        state = {
            'config': self.config,
        }
        
        # Add additional state if the tool has a get_state method
        if hasattr(self.tool, 'get_state') and callable(self.tool.get_state):
            state.update(self.tool.get_state())
        
        return state

    def set_state(self, state: Dict[str, Any]) -> bool:
        """
        Set the state of the tool.

        Args:
            state: Dictionary containing the state to set.

        Returns:
            True if the state was set successfully, False otherwise.
        """
        try:
            if 'config' in state:
                self.config.update(state['config'])
            
            # Set additional state if the tool has a set_state method
            if hasattr(self.tool, 'set_state') and callable(self.tool.set_state):
                return self.tool.set_state(state)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to set tool state: {e}")
            return False

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the tool.

        Returns:
            Dictionary containing the current configuration.
        """
        return self.config

    def set_config(self, config: Dict[str, Any]) -> bool:
        """
        Set the configuration of the tool.

        Args:
            config: Dictionary containing the configuration to set.

        Returns:
            True if the configuration was set successfully, False otherwise.
        """
        try:
            self.config.update(config)
            
            # Set the configuration if the tool has a set_config method
            if hasattr(self.tool, 'set_config') and callable(self.tool.set_config):
                return self.tool.set_config(config)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to set tool configuration: {e}")
            return False

    def get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration of the tool.

        Returns:
            Dictionary containing the default configuration.
        """
        # Get the default configuration if the tool has a get_default_config method
        if hasattr(self.tool, 'get_default_config') and callable(self.tool.get_default_config):
            return self.tool.get_default_config()
        
        return {}

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task.

        Args:
            task: Dictionary containing the task to execute.

        Returns:
            Dictionary containing the result of the task execution.
        """
        try:
            # Execute the task if the tool has an execute method
            if hasattr(self.tool, 'execute') and callable(self.tool.execute):
                return self.tool.execute(task)
            
            # Otherwise, try to run the task as a function call
            if hasattr(self.tool, '__call__') and callable(self.tool.__call__):
                result = self.tool(**task)
                return {'success': True, 'result': result}
            
            return {'success': False, 'error': 'Tool does not support task execution'}
        except Exception as e:
            self.logger.error(f"Failed to execute task: {e}")
            return {'success': False, 'error': str(e)}

    def can_execute(self, task: Dict[str, Any]) -> bool:
        """
        Check if this tool can execute the given task.

        Args:
            task: Dictionary containing the task to check.

        Returns:
            True if this tool can execute the task, False otherwise.
        """
        # Check if the tool has a can_execute method
        if hasattr(self.tool, 'can_execute') and callable(self.tool.can_execute):
            return self.tool.can_execute(task)
        
        # Otherwise, check if the tool has an execute method or is callable
        return (hasattr(self.tool, 'execute') and callable(self.tool.execute)) or (hasattr(self.tool, '__call__') and callable(self.tool.__call__))

    def get_description(self) -> str:
        """
        Get a description of this tool.

        Returns:
            Description of this tool.
        """
        # Get the description if the tool has a get_description method
        if hasattr(self.tool, 'get_description') and callable(self.tool.get_description):
            return self.tool.get_description()
        
        # Otherwise, use the tool's docstring or class name
        return getattr(self.tool, '__doc__', '') or type(self.tool).__name__

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the parameters required by this tool.

        Returns:
            Dictionary mapping parameter names to parameter specifications.
        """
        # Get the parameters if the tool has a get_parameters method
        if hasattr(self.tool, 'get_parameters') and callable(self.tool.get_parameters):
            return self.tool.get_parameters()
        
        # Otherwise, return an empty dictionary
        return {}


class MemoryAdapter(Memory):
    """
    Adapter for memory components.

    This class adapts memory components from merged functions to the Memory interface.
    """

    def __init__(self, memory_instance: Any, config: Dict[str, Any]):
        """
        Initialize the memory adapter.

        Args:
            memory_instance: The memory instance to adapt.
            config: Configuration dictionary for the memory.
        """
        self.memory = memory_instance
        self.config = config
        self.observers = {}
        self.logger = logging.getLogger(__name__)

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
            
            # Initialize the memory if it has an initialize method
            if hasattr(self.memory, 'initialize') and callable(self.memory.initialize):
                return self.memory.initialize(config)
            
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
        info = {
            'type': type(self.memory).__name__,
        }
        
        # Add additional info if the memory has a get_info method
        if hasattr(self.memory, 'get_info') and callable(self.memory.get_info):
            info.update(self.memory.get_info())
        
        return info

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the memory.

        Returns:
            Dictionary containing the current state of the memory.
        """
        state = {
            'config': self.config,
        }
        
        # Add additional state if the memory has a get_state method
        if hasattr(self.memory, 'get_state') and callable(self.memory.get_state):
            state.update(self.memory.get_state())
        
        return state

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
            
            # Set additional state if the memory has a set_state method
            if hasattr(self.memory, 'set_state') and callable(self.memory.set_state):
                return self.memory.set_state(state)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to set memory state: {e}")
            return False

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
            
            # Set the configuration if the memory has a set_config method
            if hasattr(self.memory, 'set_config') and callable(self.memory.set_config):
                return self.memory.set_config(config)
            
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
        # Get the default configuration if the memory has a get_default_config method
        if hasattr(self.memory, 'get_default_config') and callable(self.memory.get_default_config):
            return self.memory.get_default_config()
        
        return {}

    def add_observer(self, observer: 'Observer', event_type: Optional[str] = None) -> bool:
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

    def remove_observer(self, observer: 'Observer', event_type: Optional[str] = None) -> bool:
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
            # Add the item if the memory has an add method
            if hasattr(self.memory, 'add') and callable(self.memory.add):
                result = self.memory.add(item)
                
                # Notify observers
                self.notify_observers('add', {'item': item})
                
                return result
            
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
            # Get items if the memory has a get method
            if hasattr(self.memory, 'get') and callable(self.memory.get):
                return self.memory.get(query)
            
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
            # Update the item if the memory has an update method
            if hasattr(self.memory, 'update') and callable(self.memory.update):
                result = self.memory.update(item_id, item)
                
                # Notify observers
                self.notify_observers('update', {'item_id': item_id, 'item': item})
                
                return result
            
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
            # Remove the item if the memory has a remove method
            if hasattr(self.memory, 'remove') and callable(self.memory.remove):
                result = self.memory.remove(item_id)
                
                # Notify observers
                self.notify_observers('remove', {'item_id': item_id})
                
                return result
            
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
            # Clear memory if the memory has a clear method
            if hasattr(self.memory, 'clear') and callable(self.memory.clear):
                result = self.memory.clear()
                
                # Notify observers
                self.notify_observers('clear', {})
                
                return result
            
            return False
        except Exception as e:
            self.logger.error(f"Failed to clear memory: {e}")
            return False