"""
Core cognitive architecture for the AGI system.

This module defines the core cognitive architecture that forms the foundation of the AGI system.
It implements a comprehensive cognitive architecture inspired by human cognition, with
components for executive function, working memory, long-term memory, attention, and
meta-cognition.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

from ..interfaces.component import Component, ComponentRegistry
from ..interfaces.memory import Memory, WorkingMemory, LongTermMemory
from ..interfaces.cognition import Attention, MetaCognition, ExecutiveFunction


class CognitiveArchitecture:
    """
    Core cognitive architecture for the AGI system.
    
    This class implements a comprehensive cognitive architecture inspired by human cognition,
    with components for executive function, working memory, long-term memory, attention, and
    meta-cognition.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cognitive architecture.
        
        Args:
            config: Optional configuration dictionary for the cognitive architecture.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.registry = ComponentRegistry()
        
        # Core cognitive components
        self.executive_function = None
        self.working_memory = None
        self.long_term_memory = None
        self.attention = None
        self.meta_cognition = None
        
        # System state
        self.initialized = False
        self.active = False
        self.cognitive_cycle_count = 0
        self.start_time = None
        self.current_goal = None
        self.current_context = {}
        
        self.logger.info("Cognitive architecture created")
    
    def initialize(self, components: Optional[Dict[str, Component]] = None) -> bool:
        """
        Initialize the cognitive architecture with components.
        
        Args:
            components: Optional dictionary mapping component names to component instances.
        
        Returns:
            True if initialization was successful, False otherwise.
        """
        self.logger.info("Initializing cognitive architecture")
        
        try:
            # Register components if provided
            if components:
                for name, component in components.items():
                    self.registry.register(name, component)
            
            # Initialize core cognitive components
            self._initialize_core_components()
            
            # Initialize all registered components
            for name, component in self.registry.get_all().items():
                if not component.initialize(self.config.get(name, {})):
                    self.logger.error(f"Failed to initialize component: {name}")
                    return False
            
            self.initialized = True
            self.logger.info("Cognitive architecture initialized successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to initialize cognitive architecture: {e}")
            return False
    
    def _initialize_core_components(self) -> None:
        """Initialize the core cognitive components."""
        # Get core components from registry or create defaults
        self.executive_function = self.registry.get('executive_function')
        if not self.executive_function:
            from ..modules.executive import DefaultExecutiveFunction
            self.executive_function = DefaultExecutiveFunction()
            self.registry.register('executive_function', self.executive_function)
        
        self.working_memory = self.registry.get('working_memory')
        if not self.working_memory:
            from ..modules.memory import DefaultWorkingMemory
            self.working_memory = DefaultWorkingMemory()
            self.registry.register('working_memory', self.working_memory)
        
        self.long_term_memory = self.registry.get('long_term_memory')
        if not self.long_term_memory:
            from ..modules.memory import DefaultLongTermMemory
            self.long_term_memory = DefaultLongTermMemory()
            self.registry.register('long_term_memory', self.long_term_memory)
        
        self.attention = self.registry.get('attention')
        if not self.attention:
            from ..modules.attention import DefaultAttention
            self.attention = DefaultAttention()
            self.registry.register('attention', self.attention)
        
        self.meta_cognition = self.registry.get('meta_cognition')
        if not self.meta_cognition:
            from ..modules.metacognition import DefaultMetaCognition
            self.meta_cognition = DefaultMetaCognition()
            self.registry.register('meta_cognition', self.meta_cognition)
    
    def set_goal(self, goal: Any) -> bool:
        """
        Set a goal for the cognitive architecture.
        
        Args:
            goal: The goal to set.
        
        Returns:
            True if the goal was set successfully, False otherwise.
        """
        if not self.initialized:
            self.logger.error("Cannot set goal: Cognitive architecture not initialized")
            return False
        
        try:
            self.current_goal = goal
            
            # Store the goal in working memory
            self.working_memory.store('current_goal', goal)
            
            # Inform the executive function about the new goal
            self.executive_function.set_goal(goal)
            
            self.logger.info(f"Goal set: {goal}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to set goal: {e}")
            return False
    
    def start(self) -> bool:
        """
        Start the cognitive architecture.
        
        Returns:
            True if the cognitive architecture was started successfully, False otherwise.
        """
        if not self.initialized:
            self.logger.error("Cannot start: Cognitive architecture not initialized")
            return False
        
        if self.active:
            self.logger.warning("Cognitive architecture already active")
            return True
        
        try:
            self.active = True
            self.start_time = time.time()
            self.cognitive_cycle_count = 0
            
            self.logger.info("Cognitive architecture started")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to start cognitive architecture: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the cognitive architecture.
        
        Returns:
            True if the cognitive architecture was stopped successfully, False otherwise.
        """
        if not self.active:
            self.logger.warning("Cognitive architecture not active")
            return True
        
        try:
            self.active = False
            duration = time.time() - self.start_time if self.start_time else 0
            
            self.logger.info(f"Cognitive architecture stopped after {duration:.2f} seconds and {self.cognitive_cycle_count} cognitive cycles")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to stop cognitive architecture: {e}")
            return False
    
    def run_cognitive_cycle(self) -> Dict[str, Any]:
        """
        Run a single cognitive cycle.
        
        A cognitive cycle consists of the following phases:
        1. Perception: Process sensory input
        2. Attention: Focus on relevant information
        3. Working Memory: Update working memory
        4. Reasoning: Perform reasoning on the current context
        5. Action Selection: Select actions to perform
        6. Action Execution: Execute selected actions
        7. Learning: Update knowledge based on experience
        8. Meta-Cognition: Monitor and regulate cognitive processes
        
        Returns:
            Dictionary containing the results of the cognitive cycle.
        """
        if not self.active:
            self.logger.error("Cannot run cognitive cycle: Cognitive architecture not active")
            return {'success': False, 'error': 'Cognitive architecture not active'}
        
        try:
            self.cognitive_cycle_count += 1
            cycle_start_time = time.time()
            
            # Phase 1: Perception
            perception_results = self._run_perception_phase()
            
            # Phase 2: Attention
            attention_results = self._run_attention_phase(perception_results)
            
            # Phase 3: Working Memory
            memory_results = self._run_memory_phase(attention_results)
            
            # Phase 4: Reasoning
            reasoning_results = self._run_reasoning_phase(memory_results)
            
            # Phase 5: Action Selection
            action_selection_results = self._run_action_selection_phase(reasoning_results)
            
            # Phase 6: Action Execution
            action_execution_results = self._run_action_execution_phase(action_selection_results)
            
            # Phase 7: Learning
            learning_results = self._run_learning_phase(action_execution_results)
            
            # Phase 8: Meta-Cognition
            meta_cognition_results = self._run_meta_cognition_phase(learning_results)
            
            cycle_duration = time.time() - cycle_start_time
            
            results = {
                'success': True,
                'cycle_id': self.cognitive_cycle_count,
                'duration': cycle_duration,
                'perception': perception_results,
                'attention': attention_results,
                'memory': memory_results,
                'reasoning': reasoning_results,
                'action_selection': action_selection_results,
                'action_execution': action_execution_results,
                'learning': learning_results,
                'meta_cognition': meta_cognition_results
            }
            
            self.logger.debug(f"Cognitive cycle {self.cognitive_cycle_count} completed in {cycle_duration:.4f} seconds")
            return results
        
        except Exception as e:
            self.logger.error(f"Failed to run cognitive cycle: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_perception_phase(self) -> Dict[str, Any]:
        """Run the perception phase of the cognitive cycle."""
        self.logger.debug("Running perception phase")
        
        # Get all perception components
        perception_components = self.registry.get_by_type('perception')
        
        results = {}
        for name, component in perception_components.items():
            try:
                # Process sensory input
                perception_result = component.process()
                results[name] = perception_result
            except Exception as e:
                self.logger.error(f"Error in perception component {name}: {e}")
                results[name] = {'success': False, 'error': str(e)}
        
        return results
    
    def _run_attention_phase(self, perception_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run the attention phase of the cognitive cycle."""
        self.logger.debug("Running attention phase")
        
        try:
            # Focus attention on relevant information
            attention_result = self.attention.focus(perception_results)
            
            # Update current context with attended information
            self.current_context.update(attention_result.get('attended_info', {}))
            
            return attention_result
        except Exception as e:
            self.logger.error(f"Error in attention phase: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_memory_phase(self, attention_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run the memory phase of the cognitive cycle."""
        self.logger.debug("Running memory phase")
        
        results = {}
        
        try:
            # Update working memory with attended information
            working_memory_result = self.working_memory.update(attention_results)
            results['working_memory'] = working_memory_result
            
            # Retrieve relevant information from long-term memory
            retrieval_cues = working_memory_result.get('retrieval_cues', {})
            long_term_memory_result = self.long_term_memory.retrieve(retrieval_cues)
            results['long_term_memory'] = long_term_memory_result
            
            # Update working memory with retrieved information
            self.working_memory.store('retrieved_info', long_term_memory_result.get('retrieved_info', {}))
            
            return results
        except Exception as e:
            self.logger.error(f"Error in memory phase: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_reasoning_phase(self, memory_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run the reasoning phase of the cognitive cycle."""
        self.logger.debug("Running reasoning phase")
        
        # Get all reasoning components
        reasoning_components = self.registry.get_by_type('reasoning')
        
        results = {}
        for name, component in reasoning_components.items():
            try:
                # Get current context from working memory
                context = self.working_memory.get_all()
                
                # Perform reasoning
                reasoning_result = component.reason(context)
                results[name] = reasoning_result
                
                # Store reasoning results in working memory
                self.working_memory.store(f'reasoning_{name}', reasoning_result)
            except Exception as e:
                self.logger.error(f"Error in reasoning component {name}: {e}")
                results[name] = {'success': False, 'error': str(e)}
        
        return results
    
    def _run_action_selection_phase(self, reasoning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run the action selection phase of the cognitive cycle."""
        self.logger.debug("Running action selection phase")
        
        try:
            # Get current context from working memory
            context = self.working_memory.get_all()
            
            # Select actions based on reasoning results and current goal
            action_selection_result = self.executive_function.select_actions(
                context=context,
                reasoning_results=reasoning_results,
                goal=self.current_goal
            )
            
            # Store selected actions in working memory
            self.working_memory.store('selected_actions', action_selection_result.get('selected_actions', []))
            
            return action_selection_result
        except Exception as e:
            self.logger.error(f"Error in action selection phase: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_action_execution_phase(self, action_selection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run the action execution phase of the cognitive cycle."""
        self.logger.debug("Running action execution phase")
        
        results = {}
        
        try:
            # Get selected actions
            selected_actions = action_selection_results.get('selected_actions', [])
            
            # Execute each selected action
            for i, action in enumerate(selected_actions):
                action_type = action.get('type')
                action_params = action.get('params', {})
                
                # Get the appropriate component for this action type
                component = self.registry.get(action_type)
                
                if component:
                    # Execute the action
                    action_result = component.execute(action_params)
                    results[f'action_{i}'] = {
                        'type': action_type,
                        'params': action_params,
                        'result': action_result
                    }
                else:
                    self.logger.error(f"No component found for action type: {action_type}")
                    results[f'action_{i}'] = {
                        'type': action_type,
                        'params': action_params,
                        'result': {'success': False, 'error': f"No component found for action type: {action_type}"}
                    }
            
            return {'success': True, 'action_results': results}
        except Exception as e:
            self.logger.error(f"Error in action execution phase: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_learning_phase(self, action_execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run the learning phase of the cognitive cycle."""
        self.logger.debug("Running learning phase")
        
        # Get all learning components
        learning_components = self.registry.get_by_type('learning')
        
        results = {}
        for name, component in learning_components.items():
            try:
                # Get current context from working memory
                context = self.working_memory.get_all()
                
                # Learn from experience
                learning_result = component.learn(
                    context=context,
                    action_results=action_execution_results
                )
                results[name] = learning_result
                
                # Store learning results in working memory
                self.working_memory.store(f'learning_{name}', learning_result)
                
                # Update long-term memory with learned information
                if 'learned_info' in learning_result:
                    self.long_term_memory.store(learning_result['learned_info'])
            except Exception as e:
                self.logger.error(f"Error in learning component {name}: {e}")
                results[name] = {'success': False, 'error': str(e)}
        
        return results
    
    def _run_meta_cognition_phase(self, learning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run the meta-cognition phase of the cognitive cycle."""
        self.logger.debug("Running meta-cognition phase")
        
        try:
            # Get current context from working memory
            context = self.working_memory.get_all()
            
            # Monitor and regulate cognitive processes
            meta_cognition_result = self.meta_cognition.monitor(
                context=context,
                learning_results=learning_results,
                cognitive_cycle_count=self.cognitive_cycle_count
            )
            
            # Apply meta-cognitive adjustments
            if 'adjustments' in meta_cognition_result:
                self._apply_meta_cognitive_adjustments(meta_cognition_result['adjustments'])
            
            return meta_cognition_result
        except Exception as e:
            self.logger.error(f"Error in meta-cognition phase: {e}")
            return {'success': False, 'error': str(e)}
    
    def _apply_meta_cognitive_adjustments(self, adjustments: Dict[str, Any]) -> None:
        """Apply meta-cognitive adjustments to the system."""
        for component_name, adjustment in adjustments.items():
            component = self.registry.get(component_name)
            if component:
                try:
                    component.adjust(adjustment)
                except Exception as e:
                    self.logger.error(f"Error applying adjustment to {component_name}: {e}")
    
    def run(self, max_cycles: Optional[int] = None, goal_check_interval: int = 10) -> Dict[str, Any]:
        """
        Run the cognitive architecture until the goal is achieved or max_cycles is reached.
        
        Args:
            max_cycles: Maximum number of cognitive cycles to run, or None for unlimited.
            goal_check_interval: Number of cycles between goal achievement checks.
        
        Returns:
            Dictionary containing the results of the run.
        """
        if not self.initialized:
            self.logger.error("Cannot run: Cognitive architecture not initialized")
            return {'success': False, 'error': 'Cognitive architecture not initialized'}
        
        if not self.current_goal:
            self.logger.error("Cannot run: No goal set")
            return {'success': False, 'error': 'No goal set'}
        
        try:
            # Start the cognitive architecture
            if not self.active:
                self.start()
            
            cycle_results = []
            goal_achieved = False
            
            # Run cognitive cycles until goal is achieved or max_cycles is reached
            while (max_cycles is None or self.cognitive_cycle_count < max_cycles) and not goal_achieved:
                # Run a cognitive cycle
                result = self.run_cognitive_cycle()
                cycle_results.append(result)
                
                # Check if goal is achieved every goal_check_interval cycles
                if self.cognitive_cycle_count % goal_check_interval == 0:
                    goal_achieved = self._check_goal_achievement()
                    if goal_achieved:
                        self.logger.info(f"Goal achieved after {self.cognitive_cycle_count} cognitive cycles")
            
            # Stop the cognitive architecture
            self.stop()
            
            return {
                'success': True,
                'goal_achieved': goal_achieved,
                'cycles_run': self.cognitive_cycle_count,
                'cycle_results': cycle_results
            }
        
        except Exception as e:
            self.logger.error(f"Failed to run cognitive architecture: {e}")
            self.stop()
            return {'success': False, 'error': str(e)}
    
    def _check_goal_achievement(self) -> bool:
        """Check if the current goal has been achieved."""
        try:
            # Get goal achievement status from executive function
            return self.executive_function.check_goal_achievement(self.current_goal)
        except Exception as e:
            self.logger.error(f"Error checking goal achievement: {e}")
            return False
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the cognitive architecture.
        
        Returns:
            Dictionary containing the current state.
        """
        state = {
            'initialized': self.initialized,
            'active': self.active,
            'cognitive_cycle_count': self.cognitive_cycle_count,
            'current_goal': self.current_goal,
            'current_context': self.current_context,
            'components': {}
        }
        
        # Get state of all registered components
        for name, component in self.registry.get_all().items():
            try:
                component_state = component.get_state()
                state['components'][name] = component_state
            except Exception as e:
                self.logger.error(f"Error getting state of component {name}: {e}")
                state['components'][name] = {'error': str(e)}
        
        return state
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        """
        Set the state of the cognitive architecture.
        
        Args:
            state: Dictionary containing the state to set.
        
        Returns:
            True if the state was set successfully, False otherwise.
        """
        try:
            # Set state of all registered components
            if 'components' in state:
                for name, component_state in state['components'].items():
                    component = self.registry.get(name)
                    if component:
                        component.set_state(component_state)
            
            # Set cognitive architecture state
            if 'current_goal' in state:
                self.current_goal = state['current_goal']
            
            if 'current_context' in state:
                self.current_context = state['current_context']
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to set state: {e}")
            return False