#!/usr/bin/env python3
"""
Simple example of using the Agentic AI platform.

This script demonstrates how to create and use a simple agent with the Agentic AI platform.
"""

import logging
import sys
from typing import Dict, Any

from agentic_platform.core.system import AgenticSystem
from agentic_platform.agents.base_agent import BaseAgent
from agentic_platform.memory.simple_memory import SimpleMemory
from agentic_platform.tools.calculator_tool import CalculatorTool


def setup_logging():
    """Set up logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def create_system() -> AgenticSystem:
    """Create and initialize the agentic system."""
    # Create the agentic system
    system = AgenticSystem()
    
    # Create and add an agent
    agent = BaseAgent({})
    system.add_component('agent', agent)
    
    # Create and add a memory
    memory = SimpleMemory({})
    system.add_component('memory', memory)
    
    # Create and add a calculator tool
    calculator = CalculatorTool({})
    system.add_component('calculator', calculator)
    
    # Connect the agent to the memory and tool
    system.connect_agent_to_memory('agent', 'memory')
    system.connect_agent_to_tool('agent', 'calculator')
    
    # Initialize the system
    system.initialize()
    
    return system


def run_example(system: AgenticSystem):
    """Run an example task using the agentic system."""
    # Get the agent
    agent = system.get_agent('agent')
    
    if not agent:
        logging.error("Agent not found")
        return
    
    # Get the calculator tool
    calculator = system.get_tool('calculator')
    
    if not calculator:
        logging.error("Calculator tool not found")
        return
    
    # Execute a calculator task directly
    task = {
        'operation': 'add',
        'a': 2,
        'b': 3
    }
    
    result = calculator.execute(task)
    logging.info(f"Calculator result: {result}")
    
    # Run the agent with a goal
    result = agent.run("Calculate the sum of 5 and 7")
    logging.info(f"Agent result: {result}")


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