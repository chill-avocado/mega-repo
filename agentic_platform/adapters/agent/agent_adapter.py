"""
Agent adapter for integrating external agent components into the Agentic AI platform.

This module provides an adapter for integrating external agent components into the
Agentic AI platform.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from ...interfaces.agent import Agent, Tool, Memory
from ...interfaces.base import Observable, Observer
from ..base import BaseAdapter


class AgentAdapter(Agent):
    """
    Agent adapter for integrating external agent components into the Agentic AI platform.
    
    This class provides an adapter that wraps an external agent component and exposes
    it through the Agent interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the agent adapter.
        
        Args:
            config: Configuration dictionary for the adapter.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.tools = []
        self.memories = []
        self.observers = {}
        self.initialized = False
        self.state = {}
        
        # Get the component path and class from the config
        self.component_path = self.config.get('component_path')
        self.component_class = self.config.get('component_class')
        
        # Create the adapter
        self.adapter = BaseAdapter(self.component_path, self.component_class, self.config.get('component_config'))
        
        # Load the component
        self.component = self.adapter.load_component()
    
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
            
            # Initialize the component if it has an initialize method
            if hasattr(self.component, 'initialize') and callable(self.component.initialize):
                self.component.initialize(config)
            
            self.initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize agent adapter: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the agent.
        
        Returns:
            Dictionary containing information about the agent.
        """
        return {
            'type': self.__class__.__name__,
            'component_path': self.component_path,
            'component_class': self.component_class,
            'tools': len(self.tools),
            'memories': len(self.memories),
            'initialized': self.initialized
        }
    
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
            
            # Set the component config if it has a set_config method
            if hasattr(self.component, 'set_config') and callable(self.component.set_config):
                self.component.set_config(config)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to set agent adapter configuration: {e}")
            return False
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration of the agent.
        
        Returns:
            Dictionary containing the default configuration.
        """
        # Get the component default config if it has a get_default_config method
        if hasattr(self.component, 'get_default_config') and callable(self.component.get_default_config):
            return self.component.get_default_config()
        
        return {}
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the agent.
        
        Returns:
            Dictionary containing the current state of the agent.
        """
        # Get the component state if it has a get_state method
        if hasattr(self.component, 'get_state') and callable(self.component.get_state):
            return self.component.get_state()
        
        return {
            'config': self.config,
            'tools': [tool.get_info() for tool in self.tools],
            'memories': [memory.get_info() for memory in self.memories],
            'state': self.state
        }
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        """
        Set the state of the agent.
        
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
            
            if 'state' in state:
                self.state.update(state['state'])
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to set agent adapter state: {e}")
            return False
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task.
        
        Args:
            task: Dictionary containing the task to execute.
        
        Returns:
            Dictionary containing the result of the task execution.
        """
        try:
            # Execute the task using the component if it has an execute method
            if hasattr(self.component, 'execute') and callable(self.component.execute):
                return self.component.execute(task)
            
            # Check if the task has a goal
            if 'goal' in task:
                return self.run(task['goal'])
            
            # Otherwise, try to execute the task directly
            return {'success': False, 'error': 'Task type not supported'}
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
        # Check if the component can execute the task if it has a can_execute method
        if hasattr(self.component, 'can_execute') and callable(self.component.can_execute):
            return self.component.can_execute(task)
        
        # Check if the task has a goal
        return 'goal' in task
    
    def add_observer(self, observer: Observer, event_type: Optional[str] = None) -> bool:
        """
        Add an observer to this agent.
        
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
            self.logger.error(f"Failed to add observer to agent adapter: {e}")
            return False
    
    def remove_observer(self, observer: Observer, event_type: Optional[str] = None) -> bool:
        """
        Remove an observer from this agent.
        
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
            self.logger.error(f"Failed to remove observer from agent adapter: {e}")
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
    
    def plan(self, goal: str) -> List[Dict[str, Any]]:
        """
        Create a plan to achieve a goal.
        
        Args:
            goal: The goal to achieve.
        
        Returns:
            List of steps to achieve the goal.
        """
        try:
            # Create a plan using the component if it has a plan method
            if hasattr(self.component, 'plan') and callable(self.component.plan):
                return self.component.plan(goal)
            
            # Base implementation just returns a simple plan with the goal
            return [{'type': 'goal', 'goal': goal}]
        except Exception as e:
            self.logger.error(f"Failed to create plan: {e}")
            return [{'type': 'goal', 'goal': goal}]
    
    def execute_plan(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a plan.
        
        Args:
            plan: The plan to execute.
        
        Returns:
            Dictionary containing the result of the plan execution.
        """
        try:
            # Execute the plan using the component if it has an execute_plan method
            if hasattr(self.component, 'execute_plan') and callable(self.component.execute_plan):
                return self.component.execute_plan(plan)
            
            # Execute each step in the plan
            results = []
            
            for step in plan:
                # Notify observers of step execution
                self.notify_observers('execute_step', {'step': step})
                
                if 'type' in step and step['type'] == 'tool':
                    # Execute a tool
                    tool_name = step.get('tool')
                    tool_args = step.get('args', {})
                    
                    # Find the tool
                    tool = None
                    for t in self.tools:
                        if t.get_info().get('type') == tool_name:
                            tool = t
                            break
                    
                    if tool:
                        # Execute the tool
                        result = tool.execute(tool_args)
                        results.append(result)
                    else:
                        # Tool not found
                        result = {'success': False, 'error': f"Tool not found: {tool_name}"}
                        results.append(result)
                else:
                    # Unknown step type, just add it to the results
                    results.append(step)
            
            # Notify observers of plan completion
            self.notify_observers('execute_plan', {'plan': plan, 'results': results})
            
            return {'success': True, 'results': results}
        except Exception as e:
            self.logger.error(f"Failed to execute plan: {e}")
            return {'success': False, 'error': str(e)}
    
    def add_tool(self, tool: Tool) -> bool:
        """
        Add a tool to this agent.
        
        Args:
            tool: The tool to add.
        
        Returns:
            True if the tool was added successfully, False otherwise.
        """
        try:
            # Add the tool to the component if it has an add_tool method
            if hasattr(self.component, 'add_tool') and callable(self.component.add_tool):
                return self.component.add_tool(tool)
            
            self.tools.append(tool)
            return True
        except Exception as e:
            self.logger.error(f"Failed to add tool to agent adapter: {e}")
            return False
    
    def remove_tool(self, tool: Tool) -> bool:
        """
        Remove a tool from this agent.
        
        Args:
            tool: The tool to remove.
        
        Returns:
            True if the tool was removed successfully, False otherwise.
        """
        try:
            # Remove the tool from the component if it has a remove_tool method
            if hasattr(self.component, 'remove_tool') and callable(self.component.remove_tool):
                return self.component.remove_tool(tool)
            
            if tool in self.tools:
                self.tools.remove(tool)
            return True
        except Exception as e:
            self.logger.error(f"Failed to remove tool from agent adapter: {e}")
            return False
    
    def get_tools(self) -> List[Tool]:
        """
        Get all tools available to this agent.
        
        Returns:
            List of tools available to this agent.
        """
        # Get the tools from the component if it has a get_tools method
        if hasattr(self.component, 'get_tools') and callable(self.component.get_tools):
            return self.component.get_tools()
        
        return self.tools
    
    def add_memory(self, memory: Memory) -> bool:
        """
        Add a memory to this agent.
        
        Args:
            memory: The memory to add.
        
        Returns:
            True if the memory was added successfully, False otherwise.
        """
        try:
            # Add the memory to the component if it has an add_memory method
            if hasattr(self.component, 'add_memory') and callable(self.component.add_memory):
                return self.component.add_memory(memory)
            
            self.memories.append(memory)
            return True
        except Exception as e:
            self.logger.error(f"Failed to add memory to agent adapter: {e}")
            return False
    
    def remove_memory(self, memory: Memory) -> bool:
        """
        Remove a memory from this agent.
        
        Args:
            memory: The memory to remove.
        
        Returns:
            True if the memory was removed successfully, False otherwise.
        """
        try:
            # Remove the memory from the component if it has a remove_memory method
            if hasattr(self.component, 'remove_memory') and callable(self.component.remove_memory):
                return self.component.remove_memory(memory)
            
            if memory in self.memories:
                self.memories.remove(memory)
            return True
        except Exception as e:
            self.logger.error(f"Failed to remove memory from agent adapter: {e}")
            return False
    
    def get_memories(self) -> List[Memory]:
        """
        Get all memories available to this agent.
        
        Returns:
            List of memories available to this agent.
        """
        # Get the memories from the component if it has a get_memories method
        if hasattr(self.component, 'get_memories') and callable(self.component.get_memories):
            return self.component.get_memories()
        
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
            # Run the agent using the component if it has a run method
            if hasattr(self.component, 'run') and callable(self.component.run):
                return self.component.run(goal)
            
            # Notify observers of goal
            self.notify_observers('run', {'goal': goal})
            
            # Create a plan
            plan = self.plan(goal)
            
            # Notify observers of plan
            self.notify_observers('plan', {'goal': goal, 'plan': plan})
            
            # Execute the plan
            result = self.execute_plan(plan)
            
            # Notify observers of result
            self.notify_observers('result', {'goal': goal, 'plan': plan, 'result': result})
            
            return result
        except Exception as e:
            self.logger.error(f"Failed to run agent: {e}")
            return {'success': False, 'error': str(e)}