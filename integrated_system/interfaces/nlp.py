"""
NLP interfaces for the integrated system.

This module defines the interfaces for NLP components in the integrated system.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from .base import Component, Configurable, Executable


class NLPProcessor(Component, Configurable, Executable):
    """Interface for NLP components in the integrated system."""

    @abstractmethod
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text.

        Args:
            text: The text to process.
            options: Optional dictionary containing processing options.

        Returns:
            Dictionary containing the processed text and metadata.
        """
        pass

    @abstractmethod
    def generate_text(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate text based on a prompt.

        Args:
            prompt: The prompt to generate text from.
            options: Optional dictionary containing generation options.

        Returns:
            Dictionary containing the generated text and metadata.
        """
        pass

    @abstractmethod
    def analyze_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze text.

        Args:
            text: The text to analyze.
            options: Optional dictionary containing analysis options.

        Returns:
            Dictionary containing the analysis results.
        """
        pass

    @abstractmethod
    def tokenize(self, text: str, options: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Tokenize text.

        Args:
            text: The text to tokenize.
            options: Optional dictionary containing tokenization options.

        Returns:
            List of tokens.
        """
        pass

    @abstractmethod
    def parse(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse text.

        Args:
            text: The text to parse.
            options: Optional dictionary containing parsing options.

        Returns:
            Dictionary containing the parsing results.
        """
        pass

    @abstractmethod
    def extract_entities(self, text: str, options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Extract entities from text.

        Args:
            text: The text to extract entities from.
            options: Optional dictionary containing extraction options.

        Returns:
            List of dictionaries containing extracted entities.
        """
        pass

    @abstractmethod
    def extract_keywords(self, text: str, options: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Extract keywords from text.

        Args:
            text: The text to extract keywords from.
            options: Optional dictionary containing extraction options.

        Returns:
            List of extracted keywords.
        """
        pass

    @abstractmethod
    def extract_sentiment(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract sentiment from text.

        Args:
            text: The text to extract sentiment from.
            options: Optional dictionary containing extraction options.

        Returns:
            Dictionary containing sentiment information.
        """
        pass