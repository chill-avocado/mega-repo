#!/usr/bin/env python3
"""
Example of using adapters to integrate external components into the Agentic AI platform.

This script demonstrates how to use adapters to integrate external components from
the functions codebase into the Agentic AI platform.
"""

import logging
import sys
import os
from typing import Dict, Any

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agentic_platform.core.system import AgenticSystem
from agentic_platform.adapters.agent.agent_adapter import AgentAdapter
from agentic_platform.adapters.memory.memory_adapter import MemoryAdapter
from agentic_platform.adapters.tool.tool_adapter import ToolAdapter


def setup_logging():
    """Set up logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def create_system() -> AgenticSystem:
    """Create and initialize the agentic system with adapters."""
    # Create the agentic system
    system = AgenticSystem()
    
    # Create and add an agent adapter
    agent_config = {
        'component_path': 'agent_frameworks.common.agent_base',
        'component_class': 'Agent',
        'component_config': {
            'name': 'ExternalAgent'
        }
    }
    agent = AgentAdapter(agent_config)
    system.add_component('agent', agent)
    
    # Create and add a memory adapter
    memory_config = {
        'component_path': 'agent_frameworks.common.memory_system',
        'component_class': 'Memory',
        'component_config': {
            'capacity': 100
        }
    }
    memory = MemoryAdapter(memory_config)
    system.add_component('memory', memory)
    
    # Create and add a tool adapter
    tool_config = {
        'component_path': 'cognitive_systems.openagi.src.openagi.actions.tools.ddg_search',
        'component_class': 'DuckDuckGoSearch',
        'component_config': {
            'name': 'DuckDuckGoSearch',
            'description': 'Search the web using DuckDuckGo',
            'max_results': 5
        }
    }
    tool = ToolAdapter(tool_config)
    system.add_component('search', tool)
    
    # Connect the agent to the memory and tool
    system.connect_agent_to_memory('agent', 'memory')
    system.connect_agent_to_tool('agent', 'search')
    
    # Initialize the system
    system.initialize()
    
    return system


def run_example(system: AgenticSystem):
    """Run an example task using the agentic system with adapters."""
    # Get the agent
    agent = system.get_agent('agent')
    
    if not agent:
        logging.error("Agent not found")
        return
    
    # Get the memory
    memory = system.get_memory('memory')
    
    if not memory:
        logging.error("Memory not found")
        return
    
    # Get the search tool
    search = system.get_tool('search')
    
    if not search:
        logging.error("Search tool not found")
        return
    
    # Add some items to memory
    memory.add({'id': '1', 'content': 'This is a test item'})
    memory.add({'id': '2', 'content': 'This is another test item'})
    
    # Get items from memory
    items = memory.get({})
    logging.info(f"Memory items: {items}")
    
    # Execute a search task
    task = {
        'query': 'Agentic AI platform',
        'max_results': 3
    }
    
    try:
        result = search.execute(task)
        logging.info(f"Search result: {result}")
    except Exception as e:
        logging.error(f"Failed to execute search: {e}")
    
    # Run the agent with a goal
    try:
        result = agent.run("Search for information about artificial intelligence")
        logging.info(f"Agent result: {result}")
    except Exception as e:
        logging.error(f"Failed to run agent: {e}")


def main():
    """Main function."""
    # Set up logging
    setup_logging()
    
    # Create the agentic system
    system = create_system()
    
    # Run the example
    run_example(system)
    
    # Shut down the system
    system.shutdown()


if __name__ == "__main__":
    main()