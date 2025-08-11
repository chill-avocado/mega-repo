"""
Core system for the integrated system.

This module defines the core system that integrates all components.
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from ..interfaces import (
    Component, Integration, Agent, UserInterface, OSInteraction,
    BrowserAutomation, CodeExecution, CognitiveSystem, EvolutionOptimizer,
    NLPProcessor, Documentation
)


class IntegratedSystem:
    """
    Core system that integrates all components.

    This class provides a unified interface for accessing and using all components
    in the integrated system.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the integrated system.

        Args:
            config: Optional configuration dictionary for the system.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.components = {}
        self.integration = None

    def initialize(self) -> bool:
        """
        Initialize the integrated system.

        Returns:
            True if initialization was successful, False otherwise.
        """
        self.logger.info("Initializing integrated system")
        
        # Initialize integration component first
        if 'integration' in self.components:
            self.integration = self.components['integration']
            if not self.integration.initialize(self.config.get('integration', {})):
                self.logger.error("Failed to initialize integration component")
                return False
        
        # Initialize all other components
        for component_id, component in self.components.items():
            if component_id != 'integration':
                if not component.initialize(self.config.get(component_id, {})):
                    self.logger.error(f"Failed to initialize component: {component_id}")
                    return False
                
                # Register component with integration component
                if self.integration:
                    if not self.integration.register_component(component):
                        self.logger.error(f"Failed to register component with integration: {component_id}")
                        return False
        
        self.logger.info("Integrated system initialized successfully")
        return True

    def add_component(self, component_id: str, component: Component) -> bool:
        """
        Add a component to the integrated system.

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
        self.logger.info(f"Added component: {component_id}")
        
        # If this is an integration component, set it as the system's integration component
        if isinstance(component, Integration) and not self.integration:
            self.integration = component
            self.logger.info(f"Set {component_id} as integration component")
        
        return True

    def remove_component(self, component_id: str) -> bool:
        """
        Remove a component from the integrated system.

        Args:
            component_id: ID of the component to remove.

        Returns:
            True if the component was removed successfully, False otherwise.
        """
        if component_id not in self.components:
            self.logger.warning(f"Component with ID {component_id} does not exist")
            return False
        
        component = self.components[component_id]
        
        # Unregister component from integration component
        if self.integration and component_id != 'integration':
            if not self.integration.unregister_component(component):
                self.logger.error(f"Failed to unregister component from integration: {component_id}")
                return False
        
        # If removing the integration component, clear the reference
        if component_id == 'integration' and self.integration == component:
            self.integration = None
        
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
        Get an agent component by its ID.

        Args:
            agent_id: ID of the agent to get.

        Returns:
            The agent with the given ID, or None if no such agent exists.
        """
        component = self.get_component(agent_id)
        return component if isinstance(component, Agent) else None

    def get_ui(self, ui_id: str) -> Optional[UserInterface]:
        """
        Get a user interface component by its ID.

        Args:
            ui_id: ID of the user interface to get.

        Returns:
            The user interface with the given ID, or None if no such user interface exists.
        """
        component = self.get_component(ui_id)
        return component if isinstance(component, UserInterface) else None

    def get_os(self, os_id: str) -> Optional[OSInteraction]:
        """
        Get an OS interaction component by its ID.

        Args:
            os_id: ID of the OS interaction component to get.

        Returns:
            The OS interaction component with the given ID, or None if no such component exists.
        """
        component = self.get_component(os_id)
        return component if isinstance(component, OSInteraction) else None

    def get_browser(self, browser_id: str) -> Optional[BrowserAutomation]:
        """
        Get a browser automation component by its ID.

        Args:
            browser_id: ID of the browser automation component to get.

        Returns:
            The browser automation component with the given ID, or None if no such component exists.
        """
        component = self.get_component(browser_id)
        return component if isinstance(component, BrowserAutomation) else None

    def get_code_executor(self, executor_id: str) -> Optional[CodeExecution]:
        """
        Get a code execution component by its ID.

        Args:
            executor_id: ID of the code execution component to get.

        Returns:
            The code execution component with the given ID, or None if no such component exists.
        """
        component = self.get_component(executor_id)
        return component if isinstance(component, CodeExecution) else None

    def get_cognitive_system(self, cognitive_id: str) -> Optional[CognitiveSystem]:
        """
        Get a cognitive system component by its ID.

        Args:
            cognitive_id: ID of the cognitive system component to get.

        Returns:
            The cognitive system component with the given ID, or None if no such component exists.
        """
        component = self.get_component(cognitive_id)
        return component if isinstance(component, CognitiveSystem) else None

    def get_evolution_optimizer(self, optimizer_id: str) -> Optional[EvolutionOptimizer]:
        """
        Get an evolution optimizer component by its ID.

        Args:
            optimizer_id: ID of the evolution optimizer component to get.

        Returns:
            The evolution optimizer component with the given ID, or None if no such component exists.
        """
        component = self.get_component(optimizer_id)
        return component if isinstance(component, EvolutionOptimizer) else None

    def get_nlp_processor(self, processor_id: str) -> Optional[NLPProcessor]:
        """
        Get an NLP processor component by its ID.

        Args:
            processor_id: ID of the NLP processor component to get.

        Returns:
            The NLP processor component with the given ID, or None if no such component exists.
        """
        component = self.get_component(processor_id)
        return component if isinstance(component, NLPProcessor) else None

    def get_documentation(self, documentation_id: str) -> Optional[Documentation]:
        """
        Get a documentation component by its ID.

        Args:
            documentation_id: ID of the documentation component to get.

        Returns:
            The documentation component with the given ID, or None if no such component exists.
        """
        component = self.get_component(documentation_id)
        return component if isinstance(component, Documentation) else None

    def connect_components(self, component1_id: str, component2_id: str, connection_type: str) -> bool:
        """
        Connect two components.

        Args:
            component1_id: ID of the first component to connect.
            component2_id: ID of the second component to connect.
            connection_type: The type of connection to establish.

        Returns:
            True if the connection was established successfully, False otherwise.
        """
        if not self.integration:
            self.logger.error("No integration component available")
            return False
        
        component1 = self.get_component(component1_id)
        component2 = self.get_component(component2_id)
        
        if not component1:
            self.logger.error(f"Component with ID {component1_id} does not exist")
            return False
        
        if not component2:
            self.logger.error(f"Component with ID {component2_id} does not exist")
            return False
        
        return self.integration.connect_components(component1, component2, connection_type)

    def disconnect_components(self, component1_id: str, component2_id: str) -> bool:
        """
        Disconnect two components.

        Args:
            component1_id: ID of the first component to disconnect.
            component2_id: ID of the second component to disconnect.

        Returns:
            True if the disconnection was successful, False otherwise.
        """
        if not self.integration:
            self.logger.error("No integration component available")
            return False
        
        component1 = self.get_component(component1_id)
        component2 = self.get_component(component2_id)
        
        if not component1:
            self.logger.error(f"Component with ID {component1_id} does not exist")
            return False
        
        if not component2:
            self.logger.error(f"Component with ID {component2_id} does not exist")
            return False
        
        return self.integration.disconnect_components(component1, component2)

    def send_message(self, source_id: str, target_id: str, message: Dict[str, Any]) -> bool:
        """
        Send a message from one component to another.

        Args:
            source_id: ID of the component sending the message.
            target_id: ID of the component receiving the message.
            message: Dictionary containing the message to send.

        Returns:
            True if the message was sent successfully, False otherwise.
        """
        if not self.integration:
            self.logger.error("No integration component available")
            return False
        
        source = self.get_component(source_id)
        target = self.get_component(target_id)
        
        if not source:
            self.logger.error(f"Component with ID {source_id} does not exist")
            return False
        
        if not target:
            self.logger.error(f"Component with ID {target_id} does not exist")
            return False
        
        return self.integration.send_message(source, target, message)

    def broadcast_message(self, source_id: str, message: Dict[str, Any]) -> bool:
        """
        Broadcast a message to all components.

        Args:
            source_id: ID of the component broadcasting the message.
            message: Dictionary containing the message to broadcast.

        Returns:
            True if the message was broadcast successfully, False otherwise.
        """
        if not self.integration:
            self.logger.error("No integration component available")
            return False
        
        source = self.get_component(source_id)
        
        if not source:
            self.logger.error(f"Component with ID {source_id} does not exist")
            return False
        
        return self.integration.broadcast_message(source, message)

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
        Shut down the integrated system.

        Returns:
            True if shutdown was successful, False otherwise.
        """
        self.logger.info("Shutting down integrated system")
        
        # Disconnect all components
        if self.integration:
            for component_id, component in self.components.items():
                if component_id != 'integration':
                    if not self.integration.unregister_component(component):
                        self.logger.error(f"Failed to unregister component from integration: {component_id}")
        
        # Clear components
        self.components = {}
        self.integration = None
        
        self.logger.info("Integrated system shut down successfully")
        return True