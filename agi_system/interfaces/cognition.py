"""
Cognition interfaces for the AGI system.

This module defines the cognition interfaces for the AGI system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .component import Component


class Attention(Component):
    """Interface for attention components in the AGI system."""
    
    @abstractmethod
    def focus(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Focus attention on relevant information.
        
        Args:
            data: Dictionary containing data to focus attention on.
        
        Returns:
            Dictionary containing the result of the focus operation.
        """
        pass
    
    @abstractmethod
    def get_focus(self) -> Dict[str, Any]:
        """
        Get the current focus of attention.
        
        Returns:
            Dictionary containing the current focus of attention.
        """
        pass
    
    @abstractmethod
    def set_focus(self, focus: Dict[str, Any]) -> bool:
        """
        Set the focus of attention.
        
        Args:
            focus: Dictionary containing the focus to set.
        
        Returns:
            True if the focus was set successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_salience_map(self) -> Dict[str, float]:
        """
        Get the salience map of attention.
        
        Returns:
            Dictionary mapping items to their salience values.
        """
        pass
    
    @abstractmethod
    def set_salience(self, item: str, salience: float) -> bool:
        """
        Set the salience of an item.
        
        Args:
            item: The item to set salience for.
            salience: The salience value to set.
        
        Returns:
            True if the salience was set successfully, False otherwise.
        """
        pass


class MetaCognition(Component):
    """Interface for meta-cognition components in the AGI system."""
    
    @abstractmethod
    def monitor(self, context: Dict[str, Any], learning_results: Dict[str, Any], cognitive_cycle_count: int) -> Dict[str, Any]:
        """
        Monitor and regulate cognitive processes.
        
        Args:
            context: Dictionary containing the current context.
            learning_results: Dictionary containing the results of the learning phase.
            cognitive_cycle_count: The current cognitive cycle count.
        
        Returns:
            Dictionary containing the result of the monitoring operation.
        """
        pass
    
    @abstractmethod
    def evaluate(self, component: str, performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the performance of a component.
        
        Args:
            component: The component to evaluate.
            performance: Dictionary containing performance metrics for the component.
        
        Returns:
            Dictionary containing the evaluation result.
        """
        pass
    
    @abstractmethod
    def regulate(self, component: str, regulation: Dict[str, Any]) -> bool:
        """
        Regulate a component.
        
        Args:
            component: The component to regulate.
            regulation: Dictionary containing regulation parameters.
        
        Returns:
            True if the regulation was applied successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_meta_knowledge(self) -> Dict[str, Any]:
        """
        Get meta-knowledge about the system's cognitive processes.
        
        Returns:
            Dictionary containing meta-knowledge.
        """
        pass
    
    @abstractmethod
    def update_meta_knowledge(self, meta_knowledge: Dict[str, Any]) -> bool:
        """
        Update meta-knowledge about the system's cognitive processes.
        
        Args:
            meta_knowledge: Dictionary containing meta-knowledge to update.
        
        Returns:
            True if meta-knowledge was updated successfully, False otherwise.
        """
        pass


class ExecutiveFunction(Component):
    """Interface for executive function components in the AGI system."""
    
    @abstractmethod
    def set_goal(self, goal: Any) -> bool:
        """
        Set a goal for the executive function.
        
        Args:
            goal: The goal to set.
        
        Returns:
            True if the goal was set successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_goal(self) -> Any:
        """
        Get the current goal of the executive function.
        
        Returns:
            The current goal.
        """
        pass
    
    @abstractmethod
    def select_actions(self, context: Dict[str, Any], reasoning_results: Dict[str, Any], goal: Any) -> Dict[str, Any]:
        """
        Select actions based on reasoning results and current goal.
        
        Args:
            context: Dictionary containing the current context.
            reasoning_results: Dictionary containing the results of the reasoning phase.
            goal: The current goal.
        
        Returns:
            Dictionary containing the selected actions.
        """
        pass
    
    @abstractmethod
    def check_goal_achievement(self, goal: Any) -> bool:
        """
        Check if a goal has been achieved.
        
        Args:
            goal: The goal to check.
        
        Returns:
            True if the goal has been achieved, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_plan(self) -> List[Dict[str, Any]]:
        """
        Get the current plan of the executive function.
        
        Returns:
            List of actions in the current plan.
        """
        pass
    
    @abstractmethod
    def set_plan(self, plan: List[Dict[str, Any]]) -> bool:
        """
        Set the plan of the executive function.
        
        Args:
            plan: List of actions to set as the plan.
        
        Returns:
            True if the plan was set successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def update_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the current plan based on the current context.
        
        Args:
            context: Dictionary containing the current context.
        
        Returns:
            Dictionary containing the updated plan.
        """
        pass