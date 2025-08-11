"""
Code execution interfaces for the integrated system.

This module defines the interfaces for code execution components in the integrated system.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from .base import Component, Configurable, Executable


class CodeExecution(Component, Configurable, Executable):
    """Interface for code execution components in the integrated system."""

    @abstractmethod
    def execute_code(self, code: str, language: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute code in a specific language.

        Args:
            code: The code to execute.
            language: The programming language of the code.
            options: Optional dictionary containing execution options.

        Returns:
            Dictionary containing the result of the code execution.
        """
        pass

    @abstractmethod
    def analyze_code(self, code: str, language: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze code in a specific language.

        Args:
            code: The code to analyze.
            language: The programming language of the code.
            options: Optional dictionary containing analysis options.

        Returns:
            Dictionary containing the result of the code analysis.
        """
        pass

    @abstractmethod
    def generate_code(self, spec: Dict[str, Any], language: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate code based on a specification.

        Args:
            spec: Dictionary containing the code specification.
            language: The programming language to generate code in.
            options: Optional dictionary containing generation options.

        Returns:
            Dictionary containing the generated code and metadata.
        """
        pass

    @abstractmethod
    def test_code(self, code: str, tests: List[Dict[str, Any]], language: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Test code against a set of tests.

        Args:
            code: The code to test.
            tests: List of dictionaries containing test specifications.
            language: The programming language of the code.
            options: Optional dictionary containing test options.

        Returns:
            Dictionary containing the test results.
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Get the list of programming languages supported by this component.

        Returns:
            List of supported programming languages.
        """
        pass

    @abstractmethod
    def get_language_info(self, language: str) -> Dict[str, Any]:
        """
        Get information about a programming language.

        Args:
            language: The programming language to get information about.

        Returns:
            Dictionary containing information about the programming language.
        """
        pass