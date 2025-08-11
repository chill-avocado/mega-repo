"""
LLM-powered agent implementation for the Agentic AI platform.

This module provides an implementation of an agent that uses a language model
to plan and execute tasks.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from .base_agent import BaseAgent
from ..interfaces.agent import Tool, Memory
from ..interfaces.llm import LLMProvider, ChatLLMProvider, ChatMessage


class LLMAgent(BaseAgent):
    """
    LLM-powered agent implementation for the Agentic AI platform.
    
    This class provides an implementation of an agent that uses a language model
    to plan and execute tasks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM agent.
        
        Args:
            config: Configuration dictionary for the agent.
        """
        super().__init__(config)
        self.llm_provider = None
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the agent with the given configuration.
        
        Args:
            config: Configuration dictionary for the agent.
        
        Returns:
            True if initialization was successful, False otherwise.
        """
        if not super().initialize(config):
            return False
        
        try:
            # Get the LLM provider
            if 'llm_provider' in config:
                self.llm_provider = config['llm_provider']
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM agent: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the agent.
        
        Returns:
            Dictionary containing information about the agent.
        """
        info = super().get_info()
        info['has_llm_provider'] = self.llm_provider is not None
        return info
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration of the agent.
        
        Returns:
            Dictionary containing the default configuration.
        """
        return {
            'system_prompt': "You are an AI assistant that helps users accomplish tasks. You can use tools to interact with the world.",
            'planning_prompt': "Create a step-by-step plan to accomplish the following goal: {goal}. Use the available tools: {tools}.",
            'execution_prompt': "Execute the following plan to accomplish the goal: {goal}. Plan: {plan}. Use the available tools: {tools}."
        }
    
    def set_llm_provider(self, llm_provider: LLMProvider) -> bool:
        """
        Set the language model provider for this agent.
        
        Args:
            llm_provider: The language model provider to use.
        
        Returns:
            True if the provider was set successfully, False otherwise.
        """
        try:
            self.llm_provider = llm_provider
            return True
        except Exception as e:
            self.logger.error(f"Failed to set LLM provider: {e}")
            return False
    
    def plan(self, goal: str) -> List[Dict[str, Any]]:
        """
        Create a plan to achieve a goal using the language model.
        
        Args:
            goal: The goal to achieve.
        
        Returns:
            List of steps to achieve the goal.
        """
        if not self.llm_provider:
            self.logger.error("No LLM provider available")
            return super().plan(goal)
        
        try:
            # Get the planning prompt
            planning_prompt = self.config.get('planning_prompt', "Create a step-by-step plan to accomplish the following goal: {goal}. Use the available tools: {tools}.")
            
            # Get the available tools
            tools_info = []
            for tool in self.tools:
                tools_info.append({
                    'name': tool.get_info().get('type'),
                    'description': tool.get_description(),
                    'parameters': tool.get_parameters()
                })
            
            # Format the prompt
            prompt = planning_prompt.format(
                goal=goal,
                tools=json.dumps(tools_info, indent=2)
            )
            
            # Generate a plan using the language model
            if isinstance(self.llm_provider, ChatLLMProvider):
                # Use chat interface
                system_prompt = self.config.get('system_prompt', "You are an AI assistant that helps users accomplish tasks. You can use tools to interact with the world.")
                
                messages = [
                    ChatMessage("system", system_prompt),
                    ChatMessage("user", prompt)
                ]
                
                response = self.llm_provider.chat(messages)
            else:
                # Use completion interface
                response = self.llm_provider.generate(prompt)
            
            # Parse the response
            plan_text = response.get('text', '')
            
            # Try to extract a JSON plan
            try:
                # Look for JSON in the response
                start_idx = plan_text.find('[')
                end_idx = plan_text.rfind(']')
                
                if start_idx >= 0 and end_idx > start_idx:
                    plan_json = plan_text[start_idx:end_idx+1]
                    plan = json.loads(plan_json)
                    return plan
            except:
                pass
            
            # If JSON extraction failed, create a simple plan
            return [{'type': 'text', 'content': plan_text}]
        
        except Exception as e:
            self.logger.error(f"Failed to create plan: {e}")
            return super().plan(goal)
    
    def execute_plan(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a plan using the language model.
        
        Args:
            plan: The plan to execute.
        
        Returns:
            Dictionary containing the result of the plan execution.
        """
        if not self.llm_provider:
            self.logger.error("No LLM provider available")
            return super().execute_plan(plan)
        
        try:
            # Check if the plan has tool steps
            has_tool_steps = False
            for step in plan:
                if step.get('type') == 'tool':
                    has_tool_steps = True
                    break
            
            if has_tool_steps:
                # Execute the plan using the base implementation
                return super().execute_plan(plan)
            else:
                # The plan doesn't have tool steps, so we'll use the LLM to execute it
                
                # Get the execution prompt
                execution_prompt = self.config.get('execution_prompt', "Execute the following plan to accomplish the goal: {goal}. Plan: {plan}. Use the available tools: {tools}.")
                
                # Get the available tools
                tools_info = []
                for tool in self.tools:
                    tools_info.append({
                        'name': tool.get_info().get('type'),
                        'description': tool.get_description(),
                        'parameters': tool.get_parameters()
                    })
                
                # Format the prompt
                prompt = execution_prompt.format(
                    goal="<goal>",  # We don't have the goal here
                    plan=json.dumps(plan, indent=2),
                    tools=json.dumps(tools_info, indent=2)
                )
                
                # Generate a response using the language model
                if isinstance(self.llm_provider, ChatLLMProvider):
                    # Use chat interface
                    system_prompt = self.config.get('system_prompt', "You are an AI assistant that helps users accomplish tasks. You can use tools to interact with the world.")
                    
                    messages = [
                        ChatMessage("system", system_prompt),
                        ChatMessage("user", prompt)
                    ]
                    
                    response = self.llm_provider.chat_with_tools(messages, tools_info)
                else:
                    # Use completion interface
                    response = self.llm_provider.generate_with_tools(prompt, tools_info)
                
                # Check if the response has tool calls
                tool_calls = response.get('tool_calls', [])
                
                if tool_calls:
                    # Execute the tool calls
                    results = []
                    
                    for tool_call in tool_calls:
                        tool_name = tool_call.get('name')
                        tool_args = tool_call.get('arguments', {})
                        
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
                    
                    return {'success': True, 'results': results}
                else:
                    # No tool calls, just return the response
                    return {'success': True, 'text': response.get('text', '')}
        
        except Exception as e:
            self.logger.error(f"Failed to execute plan: {e}")
            return super().execute_plan(plan)