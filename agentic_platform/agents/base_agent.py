"""
Base agent implementation for the Agentic AI platform.

This module provides a base implementation of an agent for the Agentic AI platform.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from ..interfaces.agent import Agent, Tool, Memory
from ..interfaces.base import Observable, Observer


class BaseAgent(Agent):
    """
    Base implementation of an agent for the Agentic AI platform.
    
    This class provides a base implementation of an agent that can be extended
    to create more specialized agents.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base agent.
        
        Args:
            config: Configuration dictionary for the agent.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.tools = []
        self.memories = []
        self.observers = {}
        self.initialized = False
        self.state = {}
    
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
            self.initialized = True
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
        return {
            'type': self.__class__.__name__,
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
        return {}
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the agent.
        
        Returns:
            Dictionary containing the current state of the agent.
        """
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
            if 'config' in state:
                self.config.update(state['config'])
            
            if 'state' in state:
                self.state.update(state['state'])
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to set agent state: {e}")
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
            if event_type not in self.observers:
                self.observers[event_type] = []
            
            if observer not in self.observers[event_type]:
                self.observers[event_type].append(observer)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to add observer to agent: {e}")
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
        # Base implementation just returns a simple plan with the goal
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
            self.tools.append(tool)
            return True
        except Exception as e:
            self.logger.error(f"Failed to add tool to agent: {e}")
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
            if tool in self.tools:
                self.tools.remove(tool)
            return True
        except Exception as e:
            self.logger.error(f"Failed to remove tool from agent: {e}")
            return False
    
    def get_tools(self) -> List[Tool]:
        """
        Get all tools available to this agent.
        
        Returns:
            List of tools available to this agent.
        """
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
            self.memories.append(memory)
            return True
        except Exception as e:
            self.logger.error(f"Failed to add memory to agent: {e}")
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
            if memory in self.memories:
                self.memories.remove(memory)
            return True
        except Exception as e:
            self.logger.error(f"Failed to remove memory from agent: {e}")
            return False
    
    def get_memories(self) -> List[Memory]:
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