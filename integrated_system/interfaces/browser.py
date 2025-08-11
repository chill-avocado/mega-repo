"""
Browser automation interfaces for the integrated system.

This module defines the interfaces for browser automation components in the integrated system.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from .base import Component, Configurable, Executable


class BrowserAutomation(Component, Configurable, Executable):
    """Interface for browser automation components in the integrated system."""

    @abstractmethod
    def navigate(self, url: str) -> bool:
        """
        Navigate to a URL.

        Args:
            url: The URL to navigate to.

        Returns:
            True if navigation was successful, False otherwise.
        """
        pass

    @abstractmethod
    def click(self, selector: str, options: Optional[Dict[str, Any]] = None) -> bool:
        """
        Click an element on the page.

        Args:
            selector: CSS selector for the element to click.
            options: Optional dictionary containing click options.

        Returns:
            True if the click was successful, False otherwise.
        """
        pass

    @abstractmethod
    def type(self, selector: str, text: str, options: Optional[Dict[str, Any]] = None) -> bool:
        """
        Type text into an element on the page.

        Args:
            selector: CSS selector for the element to type into.
            text: The text to type.
            options: Optional dictionary containing typing options.

        Returns:
            True if typing was successful, False otherwise.
        """
        pass

    @abstractmethod
    def extract_content(self, selector: str) -> Dict[str, Any]:
        """
        Extract content from an element on the page.

        Args:
            selector: CSS selector for the element to extract content from.

        Returns:
            Dictionary containing the extracted content.
        """
        pass

    @abstractmethod
    def wait_for(self, condition: str, options: Optional[Dict[str, Any]] = None) -> bool:
        """
        Wait for a condition to be met.

        Args:
            condition: The condition to wait for.
            options: Optional dictionary containing wait options.

        Returns:
            True if the condition was met, False otherwise.
        """
        pass

    @abstractmethod
    def execute_script(self, script: str, args: Optional[List[Any]] = None) -> Any:
        """
        Execute JavaScript on the page.

        Args:
            script: The JavaScript to execute.
            args: Optional list of arguments to pass to the script.

        Returns:
            The result of the script execution.
        """
        pass

    @abstractmethod
    def take_screenshot(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Take a screenshot of the page.

        Args:
            options: Optional dictionary containing screenshot options.

        Returns:
            Dictionary containing the screenshot and metadata.
        """
        pass

    @abstractmethod
    def get_cookies(self) -> List[Dict[str, Any]]:
        """
        Get all cookies for the current page.

        Returns:
            List of dictionaries containing cookie information.
        """
        pass

    @abstractmethod
    def set_cookie(self, cookie: Dict[str, Any]) -> bool:
        """
        Set a cookie for the current page.

        Args:
            cookie: Dictionary containing cookie information.

        Returns:
            True if the cookie was set successfully, False otherwise.
        """
        pass