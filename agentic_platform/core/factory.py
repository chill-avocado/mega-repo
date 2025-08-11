"""
Factory for creating components for the Agentic AI platform.

This module provides a factory for creating components for the Agentic AI platform.
"""

import importlib
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Type, Union

from ..interfaces.agent import Agent, Tool, Memory
from ..interfaces.llm import LLMProvider, ChatLLMProvider


class ComponentFactory:
    """
    Factory for creating components for the Agentic AI platform.

    This class provides methods for creating components for the Agentic AI platform.
    """

    @staticmethod
    def create_agent(agent_type: str, config: Optional[Dict[str, Any]] = None) -> Agent:
        """
        Create an agent component.

        Args:
            agent_type: Type of agent to create.
            config: Optional configuration dictionary for the agent.

        Returns:
            The created agent component.

        Raises:
            ImportError: If the agent type cannot be imported.
            ValueError: If the agent type is not a valid agent.
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Import the agent class
            module = importlib.import_module(f"..agents.{agent_type.lower()}", package="agentic_platform.core")
            
            # Find the agent class
            for name in dir(module):
                if name.lower() == agent_type.lower() or name.lower() == f"{agent_type.lower()}agent":
                    agent_class = getattr(module, name)
                    if isinstance(agent_class, type) and issubclass(agent_class, Agent):
                        # Create an instance of the agent class
                        return agent_class(config or {})
            
            raise ValueError(f"Agent type not found: {agent_type}")
            
        except ImportError as e:
            logger.error(f"Failed to import agent type {agent_type}: {e}")
            raise ImportError(f"Failed to import agent type {agent_type}: {e}")
        except Exception as e:
            logger.error(f"Failed to create agent of type {agent_type}: {e}")
            raise ValueError(f"Failed to create agent of type {agent_type}: {e}")

    @staticmethod
    def create_tool(tool_type: str, config: Optional[Dict[str, Any]] = None) -> Tool:
        """
        Create a tool component.

        Args:
            tool_type: Type of tool to create.
            config: Optional configuration dictionary for the tool.

        Returns:
            The created tool component.

        Raises:
            ImportError: If the tool type cannot be imported.
            ValueError: If the tool type is not a valid tool.
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Import the tool class
            module = importlib.import_module(f"..tools.{tool_type.lower()}", package="agentic_platform.core")
            
            # Find the tool class
            for name in dir(module):
                if name.lower() == tool_type.lower() or name.lower() == f"{tool_type.lower()}tool":
                    tool_class = getattr(module, name)
                    if isinstance(tool_class, type) and issubclass(tool_class, Tool):
                        # Create an instance of the tool class
                        return tool_class(config or {})
            
            raise ValueError(f"Tool type not found: {tool_type}")
            
        except ImportError as e:
            logger.error(f"Failed to import tool type {tool_type}: {e}")
            raise ImportError(f"Failed to import tool type {tool_type}: {e}")
        except Exception as e:
            logger.error(f"Failed to create tool of type {tool_type}: {e}")
            raise ValueError(f"Failed to create tool of type {tool_type}: {e}")

    @staticmethod
    def create_memory(memory_type: str, config: Optional[Dict[str, Any]] = None) -> Memory:
        """
        Create a memory component.

        Args:
            memory_type: Type of memory to create.
            config: Optional configuration dictionary for the memory.

        Returns:
            The created memory component.

        Raises:
            ImportError: If the memory type cannot be imported.
            ValueError: If the memory type is not a valid memory.
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Import the memory class
            module = importlib.import_module(f"..memory.{memory_type.lower()}", package="agentic_platform.core")
            
            # Find the memory class
            for name in dir(module):
                if name.lower() == memory_type.lower() or name.lower() == f"{memory_type.lower()}memory":
                    memory_class = getattr(module, name)
                    if isinstance(memory_class, type) and issubclass(memory_class, Memory):
                        # Create an instance of the memory class
                        return memory_class(config or {})
            
            raise ValueError(f"Memory type not found: {memory_type}")
            
        except ImportError as e:
            logger.error(f"Failed to import memory type {memory_type}: {e}")
            raise ImportError(f"Failed to import memory type {memory_type}: {e}")
        except Exception as e:
            logger.error(f"Failed to create memory of type {memory_type}: {e}")
            raise ValueError(f"Failed to create memory of type {memory_type}: {e}")

    @staticmethod
    def create_llm_provider(provider_type: str, config: Optional[Dict[str, Any]] = None) -> LLMProvider:
        """
        Create a language model provider component.

        Args:
            provider_type: Type of language model provider to create.
            config: Optional configuration dictionary for the provider.

        Returns:
            The created language model provider component.

        Raises:
            ImportError: If the provider type cannot be imported.
            ValueError: If the provider type is not a valid language model provider.
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Import the provider class
            module = importlib.import_module(f"..llm.{provider_type.lower()}", package="agentic_platform.core")
            
            # Find the provider class
            for name in dir(module):
                if name.lower() == provider_type.lower() or name.lower() == f"{provider_type.lower()}provider":
                    provider_class = getattr(module, name)
                    if isinstance(provider_class, type) and issubclass(provider_class, LLMProvider):
                        # Create an instance of the provider class
                        return provider_class(config or {})
            
            raise ValueError(f"LLM provider type not found: {provider_type}")
            
        except ImportError as e:
            logger.error(f"Failed to import LLM provider type {provider_type}: {e}")
            raise ImportError(f"Failed to import LLM provider type {provider_type}: {e}")
        except Exception as e:
            logger.error(f"Failed to create LLM provider of type {provider_type}: {e}")
            raise ValueError(f"Failed to create LLM provider of type {provider_type}: {e}")

    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create components from a configuration dictionary.

        Args:
            config: Configuration dictionary containing component specifications.

        Returns:
            Dictionary mapping component IDs to created components.

        Raises:
            ValueError: If the configuration is invalid.
        """
        logger = logging.getLogger(__name__)
        
        components = {}
        
        # Create agents
        if 'agents' in config:
            for agent_id, agent_config in config['agents'].items():
                try:
                    agent_type = agent_config.get('type')
                    if not agent_type:
                        logger.warning(f"Agent {agent_id} has no type, skipping")
                        continue
                    
                    agent = ComponentFactory.create_agent(agent_type, agent_config.get('config'))
                    components[agent_id] = agent
                except Exception as e:
                    logger.error(f"Failed to create agent {agent_id}: {e}")
        
        # Create tools
        if 'tools' in config:
            for tool_id, tool_config in config['tools'].items():
                try:
                    tool_type = tool_config.get('type')
                    if not tool_type:
                        logger.warning(f"Tool {tool_id} has no type, skipping")
                        continue
                    
                    tool = ComponentFactory.create_tool(tool_type, tool_config.get('config'))
                    components[tool_id] = tool
                except Exception as e:
                    logger.error(f"Failed to create tool {tool_id}: {e}")
        
        # Create memories
        if 'memories' in config:
            for memory_id, memory_config in config['memories'].items():
                try:
                    memory_type = memory_config.get('type')
                    if not memory_type:
                        logger.warning(f"Memory {memory_id} has no type, skipping")
                        continue
                    
                    memory = ComponentFactory.create_memory(memory_type, memory_config.get('config'))
                    components[memory_id] = memory
                except Exception as e:
                    logger.error(f"Failed to create memory {memory_id}: {e}")
        
        # Create LLM providers
        if 'llm_providers' in config:
            for provider_id, provider_config in config['llm_providers'].items():
                try:
                    provider_type = provider_config.get('type')
                    if not provider_type:
                        logger.warning(f"LLM provider {provider_id} has no type, skipping")
                        continue
                    
                    provider = ComponentFactory.create_llm_provider(provider_type, provider_config.get('config'))
                    components[provider_id] = provider
                except Exception as e:
                    logger.error(f"Failed to create LLM provider {provider_id}: {e}")
        
        return components