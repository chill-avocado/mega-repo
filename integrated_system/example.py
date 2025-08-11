#!/usr/bin/env python3
"""
Example usage of the integrated system.

This script demonstrates how to use the integrated system to create a simple
agent that can interact with the operating system and browser.
"""

import logging
import sys
from typing import Dict, Any

from integrated_system.core import IntegratedSystem
from integrated_system.components.factory import ComponentFactory


def setup_logging():
    """Set up logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def create_system() -> IntegratedSystem:
    """Create and initialize the integrated system."""
    # Create the integrated system
    system = IntegratedSystem()
    
    try:
        # Create and add an integration component
        integration = ComponentFactory.create_integration('DefaultIntegration')
        system.add_component('integration', integration)
        
        # Create and add an agent component
        agent = ComponentFactory.create_agent('Agent')
        system.add_component('agent', agent)
        
        # Create and add an OS interaction component
        os_interaction = ComponentFactory.create_os_interaction('OperatingSystem')
        system.add_component('os', os_interaction)
        
        # Create and add a browser automation component
        browser = ComponentFactory.create_browser_automation('Browser')
        system.add_component('browser', browser)
        
        # Create and add an NLP processor component
        nlp = ComponentFactory.create_nlp_processor('NLP')
        system.add_component('nlp', nlp)
        
        # Initialize the system
        system.initialize()
        
        # Connect components
        system.connect_components('agent', 'os', 'uses')
        system.connect_components('agent', 'browser', 'uses')
        system.connect_components('agent', 'nlp', 'uses')
        
        return system
    except Exception as e:
        logging.error(f"Failed to create system: {e}")
        # If any component fails to be created, create a minimal system
        return create_minimal_system()


def create_minimal_system() -> IntegratedSystem:
    """Create a minimal integrated system with default components."""
    # Create the integrated system
    system = IntegratedSystem()
    
    # Create default components
    from integrated_system.components.integration import DefaultIntegration
    from integrated_system.components.agent import DefaultAgent
    
    # Add components
    system.add_component('integration', DefaultIntegration({}))
    system.add_component('agent', DefaultAgent({}))
    
    # Initialize the system
    system.initialize()
    
    return system


def run_example(system: IntegratedSystem):
    """Run an example task using the integrated system."""
    # Get the agent
    agent = system.get_agent('agent')
    
    if not agent:
        logging.error("Agent not found")
        return
    
    # Run the agent with a goal
    result = agent.run("Search for information about artificial intelligence and summarize the results")
    
    # Print the result
    logging.info(f"Agent result: {result}")


def main():
    """Main function."""
    # Set up logging
    setup_logging()
    
    # Create the integrated system
    system = create_system()
    
    # Run the example
    run_example(system)
    
    # Shut down the system
    system.shutdown()


if __name__ == "__main__":
    main()