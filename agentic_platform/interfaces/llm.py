"""
LLM interfaces for the Agentic AI platform.

This module defines the interfaces for language model components in the Agentic AI platform.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from .base import Component, Configurable, Stateful


class LLMProvider(Component, Configurable, Stateful):
    """Interface for language model providers in the Agentic AI platform."""

    @abstractmethod
    def generate(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate text using the language model.

        Args:
            prompt: The prompt to generate text from.
            options: Optional dictionary containing generation options.

        Returns:
            Dictionary containing the generated text and metadata.
        """
        pass

    @abstractmethod
    def generate_with_tools(self, prompt: str, tools: List[Dict[str, Any]], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate text with tool-calling capabilities.

        Args:
            prompt: The prompt to generate text from.
            tools: List of tools available to the language model.
            options: Optional dictionary containing generation options.

        Returns:
            Dictionary containing the generated text, tool calls, and metadata.
        """
        pass

    @abstractmethod
    def embed(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate embeddings for text.

        Args:
            text: The text to generate embeddings for.
            options: Optional dictionary containing embedding options.

        Returns:
            Dictionary containing the embeddings and metadata.
        """
        pass


class ChatMessage:
    """Class representing a chat message."""

    def __init__(self, role: str, content: str, name: Optional[str] = None):
        """
        Initialize a chat message.

        Args:
            role: The role of the message sender (e.g., "user", "assistant", "system").
            content: The content of the message.
            name: Optional name of the message sender.
        """
        self.role = role
        self.content = content
        self.name = name

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the message to a dictionary.

        Returns:
            Dictionary representation of the message.
        """
        message = {
            "role": self.role,
            "content": self.content
        }
        if self.name:
            message["name"] = self.name
        return message


class ChatLLMProvider(LLMProvider):
    """Interface for chat-based language model providers in the Agentic AI platform."""

    @abstractmethod
    def chat(self, messages: List[ChatMessage], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a chat response using the language model.

        Args:
            messages: List of chat messages.
            options: Optional dictionary containing generation options.

        Returns:
            Dictionary containing the generated response and metadata.
        """
        pass

    @abstractmethod
    def chat_with_tools(self, messages: List[ChatMessage], tools: List[Dict[str, Any]], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a chat response with tool-calling capabilities.

        Args:
            messages: List of chat messages.
            tools: List of tools available to the language model.
            options: Optional dictionary containing generation options.

        Returns:
            Dictionary containing the generated response, tool calls, and metadata.
        """
        pass