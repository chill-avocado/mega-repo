# Merged file for browser_automation/utils
# This file contains code merged from multiple repositories

import asyncio
import logging
import os
import platform
import signal
import time
from collections.abc import Callable
from collections.abc import Coroutine
from fnmatch import fnmatch
from functools import cache
from functools import wraps
from pathlib import Path
from sys import stderr
from typing import Any
from typing import ParamSpec
from typing import TypeVar
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import BadRequestError
from groq import BadRequestError
from importlib.metadata import version
import subprocess
import re

# From browser_use/utils.py
class SignalHandler:
	"""
	A modular and reusable signal handling system for managing SIGINT (Ctrl+C), SIGTERM,
	and other signals in asyncio applications.

	This class provides:
	- Configurable signal handling for SIGINT and SIGTERM
	- Support for custom pause/resume callbacks
	- Management of event loop state across signals
	- Standardized handling of first and second Ctrl+C presses
	- Cross-platform compatibility (with simplified behavior on Windows)
	"""

	def __init__(
		self,
		loop: asyncio.AbstractEventLoop | None = None,
		pause_callback: Callable[[], None] | None = None,
		resume_callback: Callable[[], None] | None = None,
		custom_exit_callback: Callable[[], None] | None = None,
		exit_on_second_int: bool = True,
		interruptible_task_patterns: list[str] | None = None,
	):
		"""
		Initialize the signal handler.

		Args:
			loop: The asyncio event loop to use. Defaults to current event loop.
			pause_callback: Function to call when system is paused (first Ctrl+C)
			resume_callback: Function to call when system is resumed
			custom_exit_callback: Function to call on exit (second Ctrl+C or SIGTERM)
			exit_on_second_int: Whether to exit on second SIGINT (Ctrl+C)
			interruptible_task_patterns: List of patterns to match task names that should be
										 canceled on first Ctrl+C (default: ['step', 'multi_act', 'get_next_action'])
		"""
		self.loop = loop or asyncio.get_event_loop()
		self.pause_callback = pause_callback
		self.resume_callback = resume_callback
		self.custom_exit_callback = custom_exit_callback
		self.exit_on_second_int = exit_on_second_int
		self.interruptible_task_patterns = interruptible_task_patterns or ['step', 'multi_act', 'get_next_action']
		self.is_windows = platform.system() == 'Windows'

		# Initialize loop state attributes
		self._initialize_loop_state()

		# Store original signal handlers to restore them later if needed
		self.original_sigint_handler = None
		self.original_sigterm_handler = None

	def _initialize_loop_state(self) -> None:
		"""Initialize loop state attributes used for signal handling."""
		setattr(self.loop, 'ctrl_c_pressed', False)
		setattr(self.loop, 'waiting_for_input', False)

	def register(self) -> None:
		"""Register signal handlers for SIGINT and SIGTERM."""
		try:
			if self.is_windows:
				# On Windows, use simple signal handling with immediate exit on Ctrl+C
				def windows_handler(sig, frame):
					print('\n\n🛑 Got Ctrl+C. Exiting immediately on Windows...\n', file=stderr)
					# Run the custom exit callback if provided
					if self.custom_exit_callback:
						self.custom_exit_callback()
					os._exit(0)

				self.original_sigint_handler = signal.signal(signal.SIGINT, windows_handler)
			else:
				# On Unix-like systems, use asyncio's signal handling for smoother experience
				self.original_sigint_handler = self.loop.add_signal_handler(signal.SIGINT, lambda: self.sigint_handler())
				self.original_sigterm_handler = self.loop.add_signal_handler(signal.SIGTERM, lambda: self.sigterm_handler())

		except Exception:
			# there are situations where signal handlers are not supported, e.g.
			# - when running in a thread other than the main thread
			# - some operating systems
			# - inside jupyter notebooks
			pass

	def unregister(self) -> None:
		"""Unregister signal handlers and restore original handlers if possible."""
		try:
			if self.is_windows:
				# On Windows, just restore the original SIGINT handler
				if self.original_sigint_handler:
					signal.signal(signal.SIGINT, self.original_sigint_handler)
			else:
				# On Unix-like systems, use asyncio's signal handler removal
				self.loop.remove_signal_handler(signal.SIGINT)
				self.loop.remove_signal_handler(signal.SIGTERM)

				# Restore original handlers if available
				if self.original_sigint_handler:
					signal.signal(signal.SIGINT, self.original_sigint_handler)
				if self.original_sigterm_handler:
					signal.signal(signal.SIGTERM, self.original_sigterm_handler)
		except Exception as e:
			logger.warning(f'Error while unregistering signal handlers: {e}')

	def _handle_second_ctrl_c(self) -> None:
		"""
		Handle a second Ctrl+C press by performing cleanup and exiting.
		This is shared logic used by both sigint_handler and wait_for_resume.
		"""
		global _exiting

		if not _exiting:
			_exiting = True

			# Call custom exit callback if provided
			if self.custom_exit_callback:
				try:
					self.custom_exit_callback()
				except Exception as e:
					logger.error(f'Error in exit callback: {e}')

		# Force immediate exit - more reliable than sys.exit()
		print('\n\n🛑  Got second Ctrl+C. Exiting immediately...\n', file=stderr)

		# Reset terminal to a clean state by sending multiple escape sequences
		# Order matters for terminal resets - we try different approaches

		# Reset terminal modes for both stdout and stderr
		print('\033[?25h', end='', flush=True, file=stderr)  # Show cursor
		print('\033[?25h', end='', flush=True)  # Show cursor

		# Reset text attributes and terminal modes
		print('\033[0m', end='', flush=True, file=stderr)  # Reset text attributes
		print('\033[0m', end='', flush=True)  # Reset text attributes

		# Disable special input modes that may cause arrow keys to output control chars
		print('\033[?1l', end='', flush=True, file=stderr)  # Reset cursor keys to normal mode
		print('\033[?1l', end='', flush=True)  # Reset cursor keys to normal mode

		# Disable bracketed paste mode
		print('\033[?2004l', end='', flush=True, file=stderr)
		print('\033[?2004l', end='', flush=True)

		# Carriage return helps ensure a clean line
		print('\r', end='', flush=True, file=stderr)
		print('\r', end='', flush=True)

		# these ^^ attempts dont work as far as we can tell
		# we still dont know what causes the broken input, if you know how to fix it, please let us know
		print('(tip: press [Enter] once to fix escape codes appearing after chrome exit)', file=stderr)

		os._exit(0)

	def sigint_handler(self) -> None:
		"""
		SIGINT (Ctrl+C) handler.

		First Ctrl+C: Cancel current step and pause.
		Second Ctrl+C: Exit immediately if exit_on_second_int is True.
		"""
		global _exiting

		if _exiting:
			# Already exiting, force exit immediately
			os._exit(0)

		if getattr(self.loop, 'ctrl_c_pressed', False):
			# If we're in the waiting for input state, let the pause method handle it
			if getattr(self.loop, 'waiting_for_input', False):
				return

			# Second Ctrl+C - exit immediately if configured to do so
			if self.exit_on_second_int:
				self._handle_second_ctrl_c()

		# Mark that Ctrl+C was pressed
		setattr(self.loop, 'ctrl_c_pressed', True)

		# Cancel current tasks that should be interruptible - this is crucial for immediate pausing
		self._cancel_interruptible_tasks()

		# Call pause callback if provided - this sets the paused flag
		if self.pause_callback:
			try:
				self.pause_callback()
			except Exception as e:
				logger.error(f'Error in pause callback: {e}')

		# Log pause message after pause_callback is called (not before)
		print('----------------------------------------------------------------------', file=stderr)

	def sigterm_handler(self) -> None:
		"""
		SIGTERM handler.

		Always exits the program completely.
		"""
		global _exiting
		if not _exiting:
			_exiting = True
			print('\n\n🛑 SIGTERM received. Exiting immediately...\n\n', file=stderr)

			# Call custom exit callback if provided
			if self.custom_exit_callback:
				self.custom_exit_callback()

		os._exit(0)

	def _cancel_interruptible_tasks(self) -> None:
		"""Cancel current tasks that should be interruptible."""
		current_task = asyncio.current_task(self.loop)
		for task in asyncio.all_tasks(self.loop):
			if task != current_task and not task.done():
				task_name = task.get_name() if hasattr(task, 'get_name') else str(task)
				# Cancel tasks that match certain patterns
				if any(pattern in task_name for pattern in self.interruptible_task_patterns):
					logger.debug(f'Cancelling task: {task_name}')
					task.cancel()
					# Add exception handler to silence "Task exception was never retrieved" warnings
					task.add_done_callback(lambda t: t.exception() if t.cancelled() else None)

		# Also cancel the current task if it's interruptible
		if current_task and not current_task.done():
			task_name = current_task.get_name() if hasattr(current_task, 'get_name') else str(current_task)
			if any(pattern in task_name for pattern in self.interruptible_task_patterns):
				logger.debug(f'Cancelling current task: {task_name}')
				current_task.cancel()

	def wait_for_resume(self) -> None:
		"""
		Wait for user input to resume or exit.

		This method should be called after handling the first Ctrl+C.
		It temporarily restores default signal handling to allow catching
		a second Ctrl+C directly.
		"""
		# Set flag to indicate we're waiting for input
		setattr(self.loop, 'waiting_for_input', True)

		# Temporarily restore default signal handling for SIGINT
		# This ensures KeyboardInterrupt will be raised during input()
		original_handler = signal.getsignal(signal.SIGINT)
		try:
			signal.signal(signal.SIGINT, signal.default_int_handler)
		except ValueError:
			# we are running in a thread other than the main thread
			# or signal handlers are not supported for some other reason
			pass

		green = '\x1b[32;1m'
		red = '\x1b[31m'
		blink = '\033[33;5m'
		unblink = '\033[0m'
		reset = '\x1b[0m'

		try:  # escape code is to blink the ...
			print(
				f'➡️  Press {green}[Enter]{reset} to resume or {red}[Ctrl+C]{reset} again to exit{blink}...{unblink} ',
				end='',
				flush=True,
				file=stderr,
			)
			input()  # This will raise KeyboardInterrupt on Ctrl+C

			# Call resume callback if provided
			if self.resume_callback:
				self.resume_callback()
		except KeyboardInterrupt:
			# Use the shared method to handle second Ctrl+C
			self._handle_second_ctrl_c()
		finally:
			try:
				# Restore our signal handler
				signal.signal(signal.SIGINT, original_handler)
				setattr(self.loop, 'waiting_for_input', False)
			except Exception:
				pass

	def reset(self) -> None:
		"""Reset state after resuming."""
		# Clear the flags
		if hasattr(self.loop, 'ctrl_c_pressed'):
			setattr(self.loop, 'ctrl_c_pressed', False)
		if hasattr(self.loop, 'waiting_for_input'):
			setattr(self.loop, 'waiting_for_input', False)

# From browser_use/utils.py
def time_execution_sync(additional_text: str = '') -> Callable[[Callable[P, R]], Callable[P, R]]:
	def decorator(func: Callable[P, R]) -> Callable[P, R]:
		@wraps(func)
		def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
			start_time = time.time()
			result = func(*args, **kwargs)
			execution_time = time.time() - start_time
			# Only log if execution takes more than 0.25 seconds
			if execution_time > 0.25:
				self_has_logger = args and getattr(args[0], 'logger', None)
				if self_has_logger:
					logger = getattr(args[0], 'logger')
				elif 'agent' in kwargs:
					logger = getattr(kwargs['agent'], 'logger')
				elif 'browser_session' in kwargs:
					logger = getattr(kwargs['browser_session'], 'logger')
				else:
					logger = logging.getLogger(__name__)
				logger.debug(f'⏳ {additional_text.strip("-")}() took {execution_time:.2f}s')
			return result

		return wrapper

	return decorator

# From browser_use/utils.py
def time_execution_async(
	additional_text: str = '',
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
	def decorator(func: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, Coroutine[Any, Any, R]]:
		@wraps(func)
		async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
			start_time = time.time()
			result = await func(*args, **kwargs)
			execution_time = time.time() - start_time
			# Only log if execution takes more than 0.25 seconds to avoid spamming the logs
			# you can lower this threshold locally when you're doing dev work to performance optimize stuff
			if execution_time > 0.25:
				self_has_logger = args and getattr(args[0], 'logger', None)
				if self_has_logger:
					logger = getattr(args[0], 'logger')
				elif 'agent' in kwargs:
					logger = getattr(kwargs['agent'], 'logger')
				elif 'browser_session' in kwargs:
					logger = getattr(kwargs['browser_session'], 'logger')
				else:
					logger = logging.getLogger(__name__)
				logger.debug(f'⏳ {additional_text.strip("-")}() took {execution_time:.2f}s')
			return result

		return wrapper

	return decorator

# From browser_use/utils.py
def singleton(cls):
	instance = [None]

	def wrapper(*args, **kwargs):
		if instance[0] is None:
			instance[0] = cls(*args, **kwargs)
		return instance[0]

	return wrapper

# From browser_use/utils.py
def check_env_variables(keys: list[str], any_or_all=all) -> bool:
	"""Check if all required environment variables are set"""
	return any_or_all(os.getenv(key, '').strip() for key in keys)

# From browser_use/utils.py
def is_unsafe_pattern(pattern: str) -> bool:
	"""
	Check if a domain pattern has complex wildcards that could match too many domains.

	Args:
		pattern: The domain pattern to check

	Returns:
		bool: True if the pattern has unsafe wildcards, False otherwise
	"""
	# Extract domain part if there's a scheme
	if '://' in pattern:
		_, pattern = pattern.split('://', 1)

	# Remove safe patterns (*.domain and domain.*)
	bare_domain = pattern.replace('.*', '').replace('*.', '')

	# If there are still wildcards, it's potentially unsafe
	return '*' in bare_domain

# From browser_use/utils.py
def is_new_tab_page(url: str) -> bool:
	"""
	Check if a URL is a new tab page (about:blank or chrome://new-tab-page).

	Args:
		url: The URL to check

	Returns:
		bool: True if the URL is a new tab page, False otherwise
	"""
	return url in ('about:blank', 'chrome://new-tab-page/', 'chrome://new-tab-page')

# From browser_use/utils.py
def match_url_with_domain_pattern(url: str, domain_pattern: str, log_warnings: bool = False) -> bool:
	"""
	Check if a URL matches a domain pattern. SECURITY CRITICAL.

	Supports optional glob patterns and schemes:
	- *.example.com will match sub.example.com and example.com
	- *google.com will match google.com, agoogle.com, and www.google.com
	- http*://example.com will match http://example.com, https://example.com
	- chrome-extension://* will match chrome-extension://aaaaaaaaaaaa and chrome-extension://bbbbbbbbbbbbb

	When no scheme is specified, https is used by default for security.
	For example, 'example.com' will match 'https://example.com' but not 'http://example.com'.

	Note: New tab pages (about:blank, chrome://new-tab-page) must be handled at the callsite, not inside this function.

	Args:
		url: The URL to check
		domain_pattern: Domain pattern to match against
		log_warnings: Whether to log warnings about unsafe patterns

	Returns:
		bool: True if the URL matches the pattern, False otherwise
	"""
	try:
		# Note: new tab pages should be handled at the callsite, not here
		if is_new_tab_page(url):
			return False

		parsed_url = urlparse(url)

		# Extract only the hostname and scheme components
		scheme = parsed_url.scheme.lower() if parsed_url.scheme else ''
		domain = parsed_url.hostname.lower() if parsed_url.hostname else ''

		if not scheme or not domain:
			return False

		# Normalize the domain pattern
		domain_pattern = domain_pattern.lower()

		# Handle pattern with scheme
		if '://' in domain_pattern:
			pattern_scheme, pattern_domain = domain_pattern.split('://', 1)
		else:
			pattern_scheme = 'https'  # Default to matching only https for security
			pattern_domain = domain_pattern

		# Handle port in pattern (we strip ports from patterns since we already
		# extracted only the hostname from the URL)
		if ':' in pattern_domain and not pattern_domain.startswith(':'):
			pattern_domain = pattern_domain.split(':', 1)[0]

		# If scheme doesn't match, return False
		if not fnmatch(scheme, pattern_scheme):
			return False

		# Check for exact match
		if pattern_domain == '*' or domain == pattern_domain:
			return True

		# Handle glob patterns
		if '*' in pattern_domain:
			# Check for unsafe glob patterns
			# First, check for patterns like *.*.domain which are unsafe
			if pattern_domain.count('*.') > 1 or pattern_domain.count('.*') > 1:
				if log_warnings:
					logger = logging.getLogger(__name__)
					logger.error(f'⛔️ Multiple wildcards in pattern=[{domain_pattern}] are not supported')
				return False  # Don't match unsafe patterns

			# Check for wildcards in TLD part (example.*)
			if pattern_domain.endswith('.*'):
				if log_warnings:
					logger = logging.getLogger(__name__)
					logger.error(f'⛔️ Wildcard TLDs like in pattern=[{domain_pattern}] are not supported for security')
				return False  # Don't match unsafe patterns

			# Then check for embedded wildcards
			bare_domain = pattern_domain.replace('*.', '')
			if '*' in bare_domain:
				if log_warnings:
					logger = logging.getLogger(__name__)
					logger.error(f'⛔️ Only *.domain style patterns are supported, ignoring pattern=[{domain_pattern}]')
				return False  # Don't match unsafe patterns

			# Special handling so that *.google.com also matches bare google.com
			if pattern_domain.startswith('*.'):
				parent_domain = pattern_domain[2:]
				if domain == parent_domain or fnmatch(domain, parent_domain):
					return True

			# Normal case: match domain against pattern
			if fnmatch(domain, pattern_domain):
				return True

		return False
	except Exception as e:
		logger = logging.getLogger(__name__)
		logger.error(f'⛔️ Error matching URL {url} with pattern {domain_pattern}: {type(e).__name__}: {e}')
		return False

# From browser_use/utils.py
def merge_dicts(a: dict, b: dict, path: tuple[str, ...] = ()):
	for key in b:
		if key in a:
			if isinstance(a[key], dict) and isinstance(b[key], dict):
				merge_dicts(a[key], b[key], path + (str(key),))
			elif isinstance(a[key], list) and isinstance(b[key], list):
				a[key] = a[key] + b[key]
			elif a[key] != b[key]:
				raise Exception('Conflict at ' + '.'.join(path + (str(key),)))
		else:
			a[key] = b[key]
	return a

# From browser_use/utils.py
def get_browser_use_version() -> str:
	"""Get the browser-use package version using the same logic as Agent._set_browser_use_version_and_source"""
	try:
		package_root = Path(__file__).parent.parent
		pyproject_path = package_root / 'pyproject.toml'

		# Try to read version from pyproject.toml
		if pyproject_path.exists():
			import re

			with open(pyproject_path, encoding='utf-8') as f:
				content = f.read()
				match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
				if match:
					version = f'{match.group(1)}'
					os.environ['LIBRARY_VERSION'] = version  # used by bubus event_schema so all Event schemas include versioning
					return version

		# If pyproject.toml doesn't exist, try getting version from pip
		from importlib.metadata import version as get_version

		version = str(get_version('browser-use'))
		os.environ['LIBRARY_VERSION'] = version
		return version

	except Exception as e:
		logger.debug(f'Error detecting browser-use version: {type(e).__name__}: {e}')
		return 'unknown'

# From browser_use/utils.py
def get_git_info() -> dict[str, str] | None:
	"""Get git information if installed from git repository"""
	try:
		import subprocess

		package_root = Path(__file__).parent.parent
		git_dir = package_root / '.git'
		if not git_dir.exists():
			return None

		# Get git commit hash
		commit_hash = (
			subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=package_root, stderr=subprocess.DEVNULL).decode().strip()
		)

		# Get git branch
		branch = (
			subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=package_root, stderr=subprocess.DEVNULL)
			.decode()
			.strip()
		)

		# Get remote URL
		remote_url = (
			subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'], cwd=package_root, stderr=subprocess.DEVNULL)
			.decode()
			.strip()
		)

		# Get commit timestamp
		commit_timestamp = (
			subprocess.check_output(['git', 'show', '-s', '--format=%ci', 'HEAD'], cwd=package_root, stderr=subprocess.DEVNULL)
			.decode()
			.strip()
		)

		return {'commit_hash': commit_hash, 'branch': branch, 'remote_url': remote_url, 'commit_timestamp': commit_timestamp}
	except Exception as e:
		logger.debug(f'Error getting git info: {type(e).__name__}: {e}')
		return None

# From browser_use/utils.py
def register(self) -> None:
		"""Register signal handlers for SIGINT and SIGTERM."""
		try:
			if self.is_windows:
				# On Windows, use simple signal handling with immediate exit on Ctrl+C
				def windows_handler(sig, frame):
					print('\n\n🛑 Got Ctrl+C. Exiting immediately on Windows...\n', file=stderr)
					# Run the custom exit callback if provided
					if self.custom_exit_callback:
						self.custom_exit_callback()
					os._exit(0)

				self.original_sigint_handler = signal.signal(signal.SIGINT, windows_handler)
			else:
				# On Unix-like systems, use asyncio's signal handling for smoother experience
				self.original_sigint_handler = self.loop.add_signal_handler(signal.SIGINT, lambda: self.sigint_handler())
				self.original_sigterm_handler = self.loop.add_signal_handler(signal.SIGTERM, lambda: self.sigterm_handler())

		except Exception:
			# there are situations where signal handlers are not supported, e.g.
			# - when running in a thread other than the main thread
			# - some operating systems
			# - inside jupyter notebooks
			pass

# From browser_use/utils.py
def unregister(self) -> None:
		"""Unregister signal handlers and restore original handlers if possible."""
		try:
			if self.is_windows:
				# On Windows, just restore the original SIGINT handler
				if self.original_sigint_handler:
					signal.signal(signal.SIGINT, self.original_sigint_handler)
			else:
				# On Unix-like systems, use asyncio's signal handler removal
				self.loop.remove_signal_handler(signal.SIGINT)
				self.loop.remove_signal_handler(signal.SIGTERM)

				# Restore original handlers if available
				if self.original_sigint_handler:
					signal.signal(signal.SIGINT, self.original_sigint_handler)
				if self.original_sigterm_handler:
					signal.signal(signal.SIGTERM, self.original_sigterm_handler)
		except Exception as e:
			logger.warning(f'Error while unregistering signal handlers: {e}')

# From browser_use/utils.py
def sigint_handler(self) -> None:
		"""
		SIGINT (Ctrl+C) handler.

		First Ctrl+C: Cancel current step and pause.
		Second Ctrl+C: Exit immediately if exit_on_second_int is True.
		"""
		global _exiting

		if _exiting:
			# Already exiting, force exit immediately
			os._exit(0)

		if getattr(self.loop, 'ctrl_c_pressed', False):
			# If we're in the waiting for input state, let the pause method handle it
			if getattr(self.loop, 'waiting_for_input', False):
				return

			# Second Ctrl+C - exit immediately if configured to do so
			if self.exit_on_second_int:
				self._handle_second_ctrl_c()

		# Mark that Ctrl+C was pressed
		setattr(self.loop, 'ctrl_c_pressed', True)

		# Cancel current tasks that should be interruptible - this is crucial for immediate pausing
		self._cancel_interruptible_tasks()

		# Call pause callback if provided - this sets the paused flag
		if self.pause_callback:
			try:
				self.pause_callback()
			except Exception as e:
				logger.error(f'Error in pause callback: {e}')

		# Log pause message after pause_callback is called (not before)
		print('----------------------------------------------------------------------', file=stderr)

# From browser_use/utils.py
def sigterm_handler(self) -> None:
		"""
		SIGTERM handler.

		Always exits the program completely.
		"""
		global _exiting
		if not _exiting:
			_exiting = True
			print('\n\n🛑 SIGTERM received. Exiting immediately...\n\n', file=stderr)

			# Call custom exit callback if provided
			if self.custom_exit_callback:
				self.custom_exit_callback()

		os._exit(0)

# From browser_use/utils.py
def wait_for_resume(self) -> None:
		"""
		Wait for user input to resume or exit.

		This method should be called after handling the first Ctrl+C.
		It temporarily restores default signal handling to allow catching
		a second Ctrl+C directly.
		"""
		# Set flag to indicate we're waiting for input
		setattr(self.loop, 'waiting_for_input', True)

		# Temporarily restore default signal handling for SIGINT
		# This ensures KeyboardInterrupt will be raised during input()
		original_handler = signal.getsignal(signal.SIGINT)
		try:
			signal.signal(signal.SIGINT, signal.default_int_handler)
		except ValueError:
			# we are running in a thread other than the main thread
			# or signal handlers are not supported for some other reason
			pass

		green = '\x1b[32;1m'
		red = '\x1b[31m'
		blink = '\033[33;5m'
		unblink = '\033[0m'
		reset = '\x1b[0m'

		try:  # escape code is to blink the ...
			print(
				f'➡️  Press {green}[Enter]{reset} to resume or {red}[Ctrl+C]{reset} again to exit{blink}...{unblink} ',
				end='',
				flush=True,
				file=stderr,
			)
			input()  # This will raise KeyboardInterrupt on Ctrl+C

			# Call resume callback if provided
			if self.resume_callback:
				self.resume_callback()
		except KeyboardInterrupt:
			# Use the shared method to handle second Ctrl+C
			self._handle_second_ctrl_c()
		finally:
			try:
				# Restore our signal handler
				signal.signal(signal.SIGINT, original_handler)
				setattr(self.loop, 'waiting_for_input', False)
			except Exception:
				pass

# From browser_use/utils.py
def reset(self) -> None:
		"""Reset state after resuming."""
		# Clear the flags
		if hasattr(self.loop, 'ctrl_c_pressed'):
			setattr(self.loop, 'ctrl_c_pressed', False)
		if hasattr(self.loop, 'waiting_for_input'):
			setattr(self.loop, 'waiting_for_input', False)

# From browser_use/utils.py
def decorator(func: Callable[P, R]) -> Callable[P, R]:
		@wraps(func)
		def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
			start_time = time.time()
			result = func(*args, **kwargs)
			execution_time = time.time() - start_time
			# Only log if execution takes more than 0.25 seconds
			if execution_time > 0.25:
				self_has_logger = args and getattr(args[0], 'logger', None)
				if self_has_logger:
					logger = getattr(args[0], 'logger')
				elif 'agent' in kwargs:
					logger = getattr(kwargs['agent'], 'logger')
				elif 'browser_session' in kwargs:
					logger = getattr(kwargs['browser_session'], 'logger')
				else:
					logger = logging.getLogger(__name__)
				logger.debug(f'⏳ {additional_text.strip("-")}() took {execution_time:.2f}s')
			return result

		return wrapper

# From browser_use/utils.py
def wrapper(*args, **kwargs):
		if instance[0] is None:
			instance[0] = cls(*args, **kwargs)
		return instance[0]

# From browser_use/utils.py
def windows_handler(sig, frame):
					print('\n\n🛑 Got Ctrl+C. Exiting immediately on Windows...\n', file=stderr)
					# Run the custom exit callback if provided
					if self.custom_exit_callback:
						self.custom_exit_callback()
					os._exit(0)

import json
import sys
from browser_use.llm.anthropic.chat import ChatAnthropic
from browser_use.llm.google.chat import ChatGoogle
from browser_use.llm.openai.chat import ChatOpenAI
from browser_use import Agent
from browser_use import Controller
from browser_use.agent.views import AgentSettings
from browser_use.browser import BrowserProfile
from browser_use.browser import BrowserSession
from browser_use.config import CONFIG
from browser_use.logging_config import addLoggingLevel
from browser_use.telemetry import CLITelemetryEvent
from browser_use.telemetry import ProductTelemetry
from browser_use.utils import get_browser_use_version
import click
from textual import events
from textual.app import App
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.containers import HorizontalGroup
from textual.containers import VerticalScroll
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Input
from textual.widgets import Label
from textual.widgets import Link
from textual.widgets import RichLog
from textual.widgets import Static
import readline
from browser_use.logging_config import setup_logging
from browser_use.mcp.server import main
import traceback

# From browser_use/cli.py
class RichLogHandler(logging.Handler):
	"""Custom logging handler that redirects logs to a RichLog widget."""

	def __init__(self, rich_log: RichLog):
		super().__init__()
		self.rich_log = rich_log

	def emit(self, record):
		try:
			msg = self.format(record)
			self.rich_log.write(msg)
		except Exception:
			self.handleError(record)

# From browser_use/cli.py
class BrowserUseApp(App):
	"""Browser-use TUI application."""

	# Make it an inline app instead of fullscreen
	# MODES = {"light"}  # Ensure app is inline, not fullscreen

	CSS = """
	#main-container {
		height: 100%;
		layout: vertical;
	}
	
	#logo-panel, #links-panel, #paths-panel, #info-panels {
		border: solid $primary;
		margin: 0 0 0 0; 
		padding: 0;
	}
	
	#info-panels {
		display: none;
		layout: vertical;
		height: auto;
		min-height: 5;
	}
	
	#top-panels {
		layout: horizontal;
		height: auto;
		width: 100%;
		min-height: 5;
	}
	
	#browser-panel, #model-panel {
		width: 1fr;
		height: auto;
		border: solid $primary-darken-2;
		padding: 1;
		overflow: auto;
		margin: 0 1 0 0;
		padding: 1;
	}
	
	#tasks-panel {
		width: 100%;
		height: 1fr;
		min-height: 20;
		max-height: 60vh;
		border: solid $primary-darken-2;
		padding: 1;
		overflow-y: scroll;
		margin: 1 0 0 0;
	}
	
	#browser-panel {
		border-left: solid $primary-darken-2;
	}
	
	#results-container {
		display: none;
	}
	
	#logo-panel {
		width: 100%;
		height: auto;
		content-align: center middle;
		text-align: center;
	}
	
	#links-panel {
		width: 100%;
		padding: 1;
		border: solid $primary;
		height: auto;
	}
	
	.link-white {
		color: white;
	}
	
	.link-purple {
		color: purple;
	}
	
	.link-magenta {
		color: magenta;
	}
	
	.link-green {
		color: green;
	}

	HorizontalGroup {
		height: auto;
	}
	
	.link-label {
		width: auto;
	}
	
	.link-url {
		width: auto;
	}
	
	.link-row {
		width: 100%;
		height: auto;
	}
	
	#paths-panel {
		color: $text-muted;
	}
	
	#task-input-container {
		border: solid $accent;
		padding: 1;
		margin-bottom: 1;
		height: auto;
		dock: bottom;
	}
	
	#task-label {
		color: $accent;
		padding-bottom: 1;
	}
	
	#task-input {
		width: 100%;
	}
	
	#working-panel {
		border: solid $warning;
		padding: 1;
		margin: 1 0;
	}
	
	#completion-panel {
		border: solid $success;
		padding: 1;
		margin: 1 0;
	}
	
	#results-container {
		height: 1fr;
		overflow: auto;
		border: none;
	}
	
	#results-log {
		height: auto;
		overflow-y: scroll;
		background: $surface;
		color: $text;
		width: 100%;
	}
	
	.log-entry {
		margin: 0;
		padding: 0;
	}
	
	#browser-info, #model-info, #tasks-info {
		height: auto;
		margin: 0;
		padding: 0;
		background: transparent;
		overflow-y: auto;
		min-height: 5;
	}
	"""

	BINDINGS = [
		Binding('ctrl+c', 'quit', 'Quit', priority=True, show=True),
		Binding('ctrl+q', 'quit', 'Quit', priority=True),
		Binding('ctrl+d', 'quit', 'Quit', priority=True),
		Binding('up', 'input_history_prev', 'Previous command', show=False),
		Binding('down', 'input_history_next', 'Next command', show=False),
	]

	def __init__(self, config: dict[str, Any], *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.config = config
		self.browser_session: BrowserSession | None = None  # Will be set before app.run_async()
		self.controller: Controller | None = None  # Will be set before app.run_async()
		self.agent: Agent | None = None
		self.llm: Any | None = None  # Will be set before app.run_async()
		self.task_history = config.get('command_history', [])
		# Track current position in history for up/down navigation
		self.history_index = len(self.task_history)
		# Initialize telemetry
		self._telemetry = ProductTelemetry()

	def setup_richlog_logging(self) -> None:
		"""Set up logging to redirect to RichLog widget instead of stdout."""
		# Try to add RESULT level if it doesn't exist
		try:
			addLoggingLevel('RESULT', 35)
		except AttributeError:
			pass  # Level already exists, which is fine

		# Get the RichLog widget
		rich_log = self.query_one('#results-log', RichLog)

		# Create and set up the custom handler
		log_handler = RichLogHandler(rich_log)
		log_type = os.getenv('BROWSER_USE_LOGGING_LEVEL', 'result').lower()

		class BrowserUseFormatter(logging.Formatter):
			def format(self, record):
				# if isinstance(record.name, str) and record.name.startswith('browser_use.'):
				# 	record.name = record.name.split('.')[-2]
				return super().format(record)

		# Set up the formatter based on log type
		if log_type == 'result':
			log_handler.setLevel('RESULT')
			log_handler.setFormatter(BrowserUseFormatter('%(message)s'))
		else:
			log_handler.setFormatter(BrowserUseFormatter('%(levelname)-8s [%(name)s] %(message)s'))

		# Configure root logger - Replace ALL handlers, not just stdout handlers
		root = logging.getLogger()

		# Clear all existing handlers and add only our richlog handler
		root.handlers = []
		root.addHandler(log_handler)

		# Set log level based on environment variable
		if log_type == 'result':
			root.setLevel('RESULT')
		elif log_type == 'debug':
			root.setLevel(logging.DEBUG)
		else:
			root.setLevel(logging.INFO)

		# Configure browser_use logger
		browser_use_logger = logging.getLogger('browser_use')
		browser_use_logger.propagate = False  # Don't propagate to root logger
		browser_use_logger.handlers = [log_handler]  # Replace any existing handlers
		browser_use_logger.setLevel(root.level)

		# Silence third-party loggers
		for logger_name in [
			'WDM',
			'httpx',
			'selenium',
			'playwright',
			'urllib3',
			'asyncio',
			'openai',
			'httpcore',
			'charset_normalizer',
			'anthropic._base_client',
			'PIL.PngImagePlugin',
			'trafilatura.htmlprocessing',
			'trafilatura',
		]:
			third_party = logging.getLogger(logger_name)
			third_party.setLevel(logging.ERROR)
			third_party.propagate = False
			third_party.handlers = []  # Clear any existing handlers

	def on_mount(self) -> None:
		"""Set up components when app is mounted."""
		# We'll use a file logger since stdout is now controlled by Textual
		logger = logging.getLogger('browser_use.on_mount')
		logger.debug('on_mount() method started')

		# Step 1: Set up custom logging to RichLog
		logger.debug('Setting up RichLog logging...')
		try:
			self.setup_richlog_logging()
			logger.debug('RichLog logging set up successfully')
		except Exception as e:
			logger.error(f'Error setting up RichLog logging: {str(e)}', exc_info=True)
			raise RuntimeError(f'Failed to set up RichLog logging: {str(e)}')

		# Step 2: Set up input history
		logger.debug('Setting up readline history...')
		try:
			if READLINE_AVAILABLE and self.task_history:
				for item in self.task_history:
					readline.add_history(item)
				logger.debug(f'Added {len(self.task_history)} items to readline history')
			else:
				logger.debug('No readline history to set up')
		except Exception as e:
			logger.error(f'Error setting up readline history: {str(e)}', exc_info=False)
			# Non-critical, continue

		# Step 3: Focus the input field
		logger.debug('Focusing input field...')
		try:
			input_field = self.query_one('#task-input', Input)
			input_field.focus()
			logger.debug('Input field focused')
		except Exception as e:
			logger.error(f'Error focusing input field: {str(e)}', exc_info=True)
			# Non-critical, continue

		# Step 5: Start continuous info panel updates
		logger.debug('Starting info panel updates...')
		try:
			self.update_info_panels()
			logger.debug('Info panel updates started')
		except Exception as e:
			logger.error(f'Error starting info panel updates: {str(e)}', exc_info=True)
			# Non-critical, continue

		# Capture telemetry for CLI start
		self._telemetry.capture(
			CLITelemetryEvent(
				version=get_browser_use_version(),
				action='start',
				mode='interactive',
				model=self.llm.model if self.llm and hasattr(self.llm, 'model') else None,
				model_provider=self.llm.provider if self.llm and hasattr(self.llm, 'provider') else None,
			)
		)

		logger.debug('on_mount() completed successfully')

	def on_input_key_up(self, event: events.Key) -> None:
		"""Handle up arrow key in the input field."""
		# For textual key events, we need to check focus manually
		input_field = self.query_one('#task-input', Input)
		if not input_field.has_focus:
			return

		# Only process if we have history
		if not self.task_history:
			return

		# Move back in history if possible
		if self.history_index > 0:
			self.history_index -= 1
			task_input = self.query_one('#task-input', Input)
			task_input.value = self.task_history[self.history_index]
			# Move cursor to end of text
			task_input.cursor_position = len(task_input.value)

		# Prevent default behavior (cursor movement)
		event.prevent_default()
		event.stop()

	def on_input_key_down(self, event: events.Key) -> None:
		"""Handle down arrow key in the input field."""
		# For textual key events, we need to check focus manually
		input_field = self.query_one('#task-input', Input)
		if not input_field.has_focus:
			return

		# Only process if we have history
		if not self.task_history:
			return

		# Move forward in history or clear input if at the end
		if self.history_index < len(self.task_history) - 1:
			self.history_index += 1
			task_input = self.query_one('#task-input', Input)
			task_input.value = self.task_history[self.history_index]
			# Move cursor to end of text
			task_input.cursor_position = len(task_input.value)
		elif self.history_index == len(self.task_history) - 1:
			# At the end of history, go to "new line" state
			self.history_index += 1
			self.query_one('#task-input', Input).value = ''

		# Prevent default behavior (cursor movement)
		event.prevent_default()
		event.stop()

	async def on_key(self, event: events.Key) -> None:
		"""Handle key events at the app level to ensure graceful exit."""
		# Handle Ctrl+C, Ctrl+D, and Ctrl+Q for app exit
		if event.key == 'ctrl+c' or event.key == 'ctrl+d' or event.key == 'ctrl+q':
			await self.action_quit()
			event.stop()
			event.prevent_default()

	def on_input_submitted(self, event: Input.Submitted) -> None:
		"""Handle task input submission."""
		if event.input.id == 'task-input':
			task = event.input.value
			if not task.strip():
				return

			# Add to history if it's new
			if task.strip() and (not self.task_history or task != self.task_history[-1]):
				self.task_history.append(task)
				self.config['command_history'] = self.task_history
				save_user_config(self.config)

			# Reset history index to point past the end of history
			self.history_index = len(self.task_history)

			# Hide logo, links, and paths panels
			self.hide_intro_panels()

			# Process the task
			self.run_task(task)

			# Clear the input
			event.input.value = ''

	def hide_intro_panels(self) -> None:
		"""Hide the intro panels, show info panels, and expand the log view."""
		try:
			# Get the panels
			logo_panel = self.query_one('#logo-panel')
			links_panel = self.query_one('#links-panel')
			paths_panel = self.query_one('#paths-panel')
			info_panels = self.query_one('#info-panels')
			tasks_panel = self.query_one('#tasks-panel')
			# Hide intro panels if they're visible and show info panels
			if logo_panel.display:
				# Log for debugging
				logging.info('Hiding intro panels and showing info panels')

				logo_panel.display = False
				links_panel.display = False
				paths_panel.display = False

				# Show info panels
				info_panels.display = True
				tasks_panel.display = True

				# Make results container take full height
				results_container = self.query_one('#results-container')
				results_container.styles.height = '1fr'

				# Configure the log
				results_log = self.query_one('#results-log')
				results_log.styles.height = 'auto'

				logging.info('Panels should now be visible')
		except Exception as e:
			logging.error(f'Error in hide_intro_panels: {str(e)}')

	def update_info_panels(self) -> None:
		"""Update all information panels with current state."""
		try:
			# Update actual content
			self.update_browser_panel()
			self.update_model_panel()
			self.update_tasks_panel()
		except Exception as e:
			logging.error(f'Error in update_info_panels: {str(e)}')
		finally:
			# Always schedule the next update - will update at 1-second intervals
			# This ensures continuous updates even if agent state changes
			self.set_timer(1.0, self.update_info_panels)

	def update_browser_panel(self) -> None:
		"""Update browser information panel with details about the browser."""
		browser_info = self.query_one('#browser-info', RichLog)
		browser_info.clear()

		# Try to use the agent's browser session if available
		browser_session = self.browser_session
		if hasattr(self, 'agent') and self.agent and hasattr(self.agent, 'browser_session'):
			browser_session = self.agent.browser_session

		if browser_session:
			try:
				# Check if browser session has a browser context
				if not hasattr(browser_session, 'browser_context') or browser_session.browser_context is None:
					browser_info.write('[yellow]Browser session created, waiting for browser to launch...[/]')
					return

				# Update our reference if we're using the agent's session
				if browser_session != self.browser_session:
					self.browser_session = browser_session

				# Get basic browser info from browser_profile
				browser_type = 'Chromium'
				headless = browser_session.browser_profile.headless

				# Determine connection type based on config
				connection_type = 'playwright'  # Default
				if browser_session.cdp_url:
					connection_type = 'CDP'
				elif browser_session.wss_url:
					connection_type = 'WSS'
				elif browser_session.browser_profile.executable_path:
					connection_type = 'user-provided'

				# Get window size details from browser_profile
				window_width = None
				window_height = None
				if browser_session.browser_profile.viewport:
					window_width = browser_session.browser_profile.viewport.get('width')
					window_height = browser_session.browser_profile.viewport.get('height')

				# Try to get browser PID
				browser_pid = 'Unknown'
				connected = False
				browser_status = '[red]Disconnected[/]'

				try:
					# Check if browser PID is available
					if hasattr(browser_session, 'browser_pid') and browser_session.browser_pid:
						browser_pid = str(browser_session.browser_pid)
						connected = True
						browser_status = '[green]Connected[/]'
					# Otherwise just check if we have a browser context
					elif browser_session.browser_context is not None:
						connected = True
						browser_status = '[green]Connected[/]'
						browser_pid = 'N/A'
				except Exception as e:
					browser_pid = f'Error: {str(e)}'

				# Display browser information
				browser_info.write(f'[bold cyan]Chromium[/] Browser ({browser_status})')
				browser_info.write(
					f'Type: [yellow]{connection_type}[/] [{"green" if not headless else "red"}]{" (headless)" if headless else ""}[/]'
				)
				browser_info.write(f'PID: [dim]{browser_pid}[/]')
				browser_info.write(f'CDP Port: {browser_session.cdp_url}')

				if window_width and window_height:
					browser_info.write(f'Window: [blue]{window_width}[/] × [blue]{window_height}[/]')

				# Include additional information about the browser if needed
				if connected and hasattr(self, 'agent') and self.agent:
					try:
						# Show when the browser was connected
						timestamp = int(time.time())
						current_time = time.strftime('%H:%M:%S', time.localtime(timestamp))
						browser_info.write(f'Last updated: [dim]{current_time}[/]')
					except Exception:
						pass

					# Show the agent's current page URL if available
					if browser_session.agent_current_page:
						current_url = (
							browser_session.agent_current_page.url.replace('https://', '')
							.replace('http://', '')
							.replace('www.', '')[:36]
							+ '…'
						)
						browser_info.write(f'👁️  [green]{current_url}[/]')
			except Exception as e:
				browser_info.write(f'[red]Error updating browser info: {str(e)}[/]')
		else:
			browser_info.write('[red]Browser not initialized[/]')

	def update_model_panel(self) -> None:
		"""Update model information panel with details about the LLM."""
		model_info = self.query_one('#model-info', RichLog)
		model_info.clear()

		if self.llm:
			# Get model details
			model_name = 'Unknown'
			if hasattr(self.llm, 'model_name'):
				model_name = self.llm.model_name
			elif hasattr(self.llm, 'model'):
				model_name = self.llm.model

			# Show model name
			if self.agent:
				temp_str = f'{self.llm.temperature}ºC ' if self.llm.temperature else ''
				vision_str = '+ vision ' if self.agent.settings.use_vision else ''
				planner_str = '+ planner' if self.agent.settings.planner_llm else ''
				model_info.write(
					f'[white]LLM:[/] [blue]{self.llm.__class__.__name__} [yellow]{model_name}[/] {temp_str}{vision_str}{planner_str}'
				)
			else:
				model_info.write(f'[white]LLM:[/] [blue]{self.llm.__class__.__name__} [yellow]{model_name}[/]')

			# Show token usage statistics if agent exists and has history
			if self.agent and hasattr(self.agent, 'state') and hasattr(self.agent.state, 'history'):
				# Get total tokens used
				# total_tokens = self.agent.history.total_input_tokens()
				# model_info.write(f'[white]Input tokens:[/] [green]{total_tokens:,}[/]')

				# Calculate tokens per step
				num_steps = len(self.agent.history.history)
				# if num_steps > 0:
				# avg_tokens_per_step = total_tokens / num_steps
				# model_info.write(f'[white]Avg tokens/step:[/] [green]{avg_tokens_per_step:,.1f}[/]')

				# Get the last step metadata to show the most recent LLM response time
				if num_steps > 0 and self.agent.history.history[-1].metadata:
					last_step = self.agent.history.history[-1]
					if last_step.metadata:
						step_duration = last_step.metadata.duration_seconds
					else:
						step_duration = 0
					# step_tokens = last_step.metadata.input_tokens

					# if step_tokens > 0:
					# 	tokens_per_second = step_tokens / step_duration if step_duration > 0 else 0
					# 	model_info.write(f'[white]Avg tokens/sec:[/] [magenta]{tokens_per_second:.1f}[/]')

				# Show total duration
				total_duration = self.agent.history.total_duration_seconds()
				if total_duration > 0:
					model_info.write(f'[white]Total Duration:[/] [magenta]{total_duration:.2f}s[/]')

					# Calculate response time metrics
					model_info.write(f'[white]Last Step Duration:[/] [magenta]{step_duration:.2f}s[/]')

				# Add current state information
				if hasattr(self.agent, 'running'):
					if getattr(self.agent, 'running', False):
						model_info.write('[yellow]LLM is thinking[blink]...[/][/]')
					elif hasattr(self.agent, 'state') and hasattr(self.agent.state, 'paused') and self.agent.state.paused:
						model_info.write('[orange]LLM paused[/]')
		else:
			model_info.write('[red]Model not initialized[/]')

	def update_tasks_panel(self) -> None:
		"""Update tasks information panel with details about the tasks and steps hierarchy."""
		tasks_info = self.query_one('#tasks-info', RichLog)
		tasks_info.clear()

		if self.agent:
			# Check if agent has tasks
			task_history = []
			message_history = []

			# Try to extract tasks by looking at message history
			if hasattr(self.agent, '_message_manager') and self.agent._message_manager:
				message_history = self.agent._message_manager.state.history.get_messages()

				# Extract original task(s)
				original_tasks = []
				for msg in message_history:
					if hasattr(msg, 'content'):
						content = msg.content
						if isinstance(content, str) and 'Your ultimate task is:' in content:
							task_text = content.split('"""')[1].strip()
							original_tasks.append(task_text)

				if original_tasks:
					tasks_info.write('[bold green]TASK:[/]')
					for i, task in enumerate(original_tasks, 1):
						# Only show latest task if multiple task changes occurred
						if i == len(original_tasks):
							tasks_info.write(f'[white]{task}[/]')
					tasks_info.write('')

			# Get current state information
			current_step = self.agent.state.n_steps if hasattr(self.agent, 'state') else 0

			# Get all agent history items
			history_items = []
			if hasattr(self.agent, 'state') and hasattr(self.agent.state, 'history'):
				history_items = self.agent.history.history

				if history_items:
					tasks_info.write('[bold yellow]STEPS:[/]')

					for idx, item in enumerate(history_items, 1):
						# Determine step status
						step_style = '[green]✓[/]'

						# For the current step, show it as in progress
						if idx == current_step:
							step_style = '[yellow]⟳[/]'

						# Check if this step had an error
						if item.result and any(result.error for result in item.result):
							step_style = '[red]✗[/]'

						# Show step number
						tasks_info.write(f'{step_style} Step {idx}/{current_step}')

						# Show goal if available
						if item.model_output and hasattr(item.model_output, 'current_state'):
							# Show goal for this step
							goal = item.model_output.current_state.next_goal
							if goal:
								# Take just the first line for display
								goal_lines = goal.strip().split('\n')
								goal_summary = goal_lines[0]
								tasks_info.write(f'   [cyan]Goal:[/] {goal_summary}')

							# Show evaluation of previous goal (feedback)
							eval_prev = item.model_output.current_state.evaluation_previous_goal
							if eval_prev and idx > 1:  # Only show for steps after the first
								eval_lines = eval_prev.strip().split('\n')
								eval_summary = eval_lines[0]
								eval_summary = eval_summary.replace('Success', '✅ ').replace('Failed', '❌ ').strip()
								tasks_info.write(f'   [tan]Evaluation:[/] {eval_summary}')

						# Show actions taken in this step
						if item.model_output and item.model_output.action:
							tasks_info.write('   [purple]Actions:[/]')
							for action_idx, action in enumerate(item.model_output.action, 1):
								action_type = action.__class__.__name__
								if hasattr(action, 'model_dump'):
									# For proper actions, show the action type
									action_dict = action.model_dump(exclude_unset=True)
									if action_dict:
										action_name = list(action_dict.keys())[0]
										tasks_info.write(f'     {action_idx}. [blue]{action_name}[/]')

						# Show results or errors from this step
						if item.result:
							for result in item.result:
								if result.error:
									error_text = result.error
									tasks_info.write(f'   [red]Error:[/] {error_text}')
								elif result.extracted_content:
									content = result.extracted_content
									tasks_info.write(f'   [green]Result:[/] {content}')

						# Add a space between steps for readability
						tasks_info.write('')

			# If agent is actively running, show a status indicator
			if hasattr(self.agent, 'running') and getattr(self.agent, 'running', False):
				tasks_info.write('[yellow]Agent is actively working[blink]...[/][/]')
			elif hasattr(self.agent, 'state') and hasattr(self.agent.state, 'paused') and self.agent.state.paused:
				tasks_info.write('[orange]Agent is paused (press Enter to resume)[/]')
		else:
			tasks_info.write('[dim]Agent not initialized[/]')

		# Force scroll to bottom
		tasks_panel = self.query_one('#tasks-panel')
		tasks_panel.scroll_end(animate=False)

	def scroll_to_input(self) -> None:
		"""Scroll to the input field to ensure it's visible."""
		input_container = self.query_one('#task-input-container')
		input_container.scroll_visible()

	def run_task(self, task: str) -> None:
		"""Launch the task in a background worker."""
		# Create or update the agent
		agent_settings = AgentSettings.model_validate(self.config.get('agent', {}))

		# Get the logger
		logger = logging.getLogger('browser_use.app')

		# Make sure intro is hidden and log is ready
		self.hide_intro_panels()

		# Start continuous updates of all info panels
		self.update_info_panels()

		# Clear the log to start fresh
		rich_log = self.query_one('#results-log', RichLog)
		rich_log.clear()

		if self.agent is None:
			if not self.llm:
				raise RuntimeError('LLM not initialized')
			self.agent = Agent(
				task=task,
				llm=self.llm,
				controller=self.controller if self.controller else Controller(),
				browser_session=self.browser_session,
				source='cli',
				**agent_settings.model_dump(),
			)
			# Update our browser_session reference to point to the agent's
			if hasattr(self.agent, 'browser_session'):
				self.browser_session = self.agent.browser_session
		else:
			self.agent.add_new_task(task)

		# Let the agent run in the background
		async def agent_task_worker() -> None:
			logger.debug('\n🚀 Working on task: %s', task)

			# Set flags to indicate the agent is running
			if self.agent:
				self.agent.running = True  # type: ignore
				self.agent.last_response_time = 0  # type: ignore

			# Panel updates are already happening via the timer in update_info_panels

			task_start_time = time.time()
			error_msg = None

			try:
				# Capture telemetry for message sent
				self._telemetry.capture(
					CLITelemetryEvent(
						version=get_browser_use_version(),
						action='message_sent',
						mode='interactive',
						model=self.llm.model if self.llm and hasattr(self.llm, 'model') else None,
						model_provider=self.llm.provider if self.llm and hasattr(self.llm, 'provider') else None,
					)
				)

				# Run the agent task, redirecting output to RichLog through our handler
				if self.agent:
					await self.agent.run()
			except Exception as e:
				error_msg = str(e)
				logger.error('\nError running agent: %s', str(e))
			finally:
				# Clear the running flag
				if self.agent:
					self.agent.running = False  # type: ignore

				# No need to call update_info_panels() here as it's already updating via timer

				# Capture telemetry for task completion
				duration = time.time() - task_start_time
				self._telemetry.capture(
					CLITelemetryEvent(
						version=get_browser_use_version(),
						action='task_completed' if error_msg is None else 'error',
						mode='interactive',
						model=self.llm.model if self.llm and hasattr(self.llm, 'model') else None,
						model_provider=self.llm.provider if self.llm and hasattr(self.llm, 'provider') else None,
						duration_seconds=duration,
						error_message=error_msg,
					)
				)

				logger.debug('\n✅ Task completed!')

				# Make sure the task input container is visible
				task_input_container = self.query_one('#task-input-container')
				task_input_container.display = True

				# Refocus the input field
				input_field = self.query_one('#task-input', Input)
				input_field.focus()

				# Ensure the input is visible by scrolling to it
				self.call_after_refresh(self.scroll_to_input)

		# Run the worker
		self.run_worker(agent_task_worker, name='agent_task')

	def action_input_history_prev(self) -> None:
		"""Navigate to the previous item in command history."""
		# Only process if we have history and input is focused
		input_field = self.query_one('#task-input', Input)
		if not input_field.has_focus or not self.task_history:
			return

		# Move back in history if possible
		if self.history_index > 0:
			self.history_index -= 1
			input_field.value = self.task_history[self.history_index]
			# Move cursor to end of text
			input_field.cursor_position = len(input_field.value)

	def action_input_history_next(self) -> None:
		"""Navigate to the next item in command history or clear input."""
		# Only process if we have history and input is focused
		input_field = self.query_one('#task-input', Input)
		if not input_field.has_focus or not self.task_history:
			return

		# Move forward in history or clear input if at the end
		if self.history_index < len(self.task_history) - 1:
			self.history_index += 1
			input_field.value = self.task_history[self.history_index]
			# Move cursor to end of text
			input_field.cursor_position = len(input_field.value)
		elif self.history_index == len(self.task_history) - 1:
			# At the end of history, go to "new line" state
			self.history_index += 1
			input_field.value = ''

	async def action_quit(self) -> None:
		"""Quit the application and clean up resources."""
		# Note: We don't need to close the browser session here because:
		# 1. If an agent exists, it already called browser_session.stop() in its run() method
		# 2. If keep_alive=True (default), we want to leave the browser running anyway
		# This prevents the duplicate "stop() called" messages in the logs

		# Flush telemetry before exiting
		self._telemetry.flush()

		# Exit the application
		self.exit()
		print('\nTry running tasks on our cloud: https://browser-use.com')

	def compose(self) -> ComposeResult:
		"""Create the UI layout."""
		yield Header()

		# Main container for app content
		with Container(id='main-container'):
			# Logo panel
			yield Static(BROWSER_LOGO, id='logo-panel', markup=True)

			# Information panels (hidden by default)
			with Container(id='info-panels'):
				# Top row with browser and model panels side by side
				with Container(id='top-panels'):
					# Browser panel
					with Container(id='browser-panel'):
						yield RichLog(id='browser-info', markup=True, highlight=True, wrap=True)

					# Model panel
					with Container(id='model-panel'):
						yield RichLog(id='model-info', markup=True, highlight=True, wrap=True)

				# Tasks panel (full width, below browser and model)
				with VerticalScroll(id='tasks-panel'):
					yield RichLog(id='tasks-info', markup=True, highlight=True, wrap=True, auto_scroll=True)

			# Links panel with URLs
			with Container(id='links-panel'):
				with HorizontalGroup(classes='link-row'):
					yield Static('Run at scale on cloud:    [blink]☁️[/]  ', markup=True, classes='link-label')
					yield Link('https://browser-use.com', url='https://browser-use.com', classes='link-white link-url')

				yield Static('')  # Empty line

				with HorizontalGroup(classes='link-row'):
					yield Static('Chat & share on Discord:  🚀 ', markup=True, classes='link-label')
					yield Link(
						'https://discord.gg/ESAUZAdxXY', url='https://discord.gg/ESAUZAdxXY', classes='link-purple link-url'
					)

				with HorizontalGroup(classes='link-row'):
					yield Static('Get prompt inspiration:   🦸 ', markup=True, classes='link-label')
					yield Link(
						'https://github.com/browser-use/awesome-prompts',
						url='https://github.com/browser-use/awesome-prompts',
						classes='link-magenta link-url',
					)

				with HorizontalGroup(classes='link-row'):
					yield Static('[dim]Report any issues:[/]        🐛 ', markup=True, classes='link-label')
					yield Link(
						'https://github.com/browser-use/browser-use/issues',
						url='https://github.com/browser-use/browser-use/issues',
						classes='link-green link-url',
					)

			# Paths panel
			yield Static(
				f' ⚙️  Settings saved to:              {str(CONFIG.BROWSER_USE_CONFIG_FILE.resolve()).replace(str(Path.home()), "~")}\n'
				f' 📁 Outputs & recordings saved to:  {str(Path(".").resolve()).replace(str(Path.home()), "~")}',
				id='paths-panel',
				markup=True,
			)

			# Results view with scrolling (place this before input to make input sticky at bottom)
			with VerticalScroll(id='results-container'):
				yield RichLog(highlight=True, markup=True, id='results-log', wrap=True, auto_scroll=True)

			# Task input container (now at the bottom)
			with Container(id='task-input-container'):
				yield Label('🔍 What would you like me to do on the web?', id='task-label')
				yield Input(placeholder='Enter your task...', id='task-input')

		yield Footer()

# From browser_use/cli.py
class BrowserUseFormatter(logging.Formatter):
			def format(self, record):
				# if isinstance(record.name, str) and record.name.startswith('browser_use.'):
				# 	record.name = record.name.split('.')[-2]
				return super().format(record)

# From browser_use/cli.py
def get_default_config() -> dict[str, Any]:
	"""Return default configuration dictionary using the new config system."""
	# Load config from the new config system
	config_data = CONFIG.load_config()

	# Extract browser profile, llm, and agent configs
	browser_profile = config_data.get('browser_profile', {})
	llm_config = config_data.get('llm', {})
	agent_config = config_data.get('agent', {})

	return {
		'model': {
			'name': llm_config.get('model'),
			'temperature': llm_config.get('temperature', 0.0),
			'api_keys': {
				'OPENAI_API_KEY': llm_config.get('api_key', CONFIG.OPENAI_API_KEY),
				'ANTHROPIC_API_KEY': CONFIG.ANTHROPIC_API_KEY,
				'GOOGLE_API_KEY': CONFIG.GOOGLE_API_KEY,
				'DEEPSEEK_API_KEY': CONFIG.DEEPSEEK_API_KEY,
				'GROK_API_KEY': CONFIG.GROK_API_KEY,
			},
		},
		'agent': agent_config,
		'browser': {
			'headless': browser_profile.get('headless', True),
			'keep_alive': browser_profile.get('keep_alive', True),
			'ignore_https_errors': browser_profile.get('ignore_https_errors', False),
			'user_data_dir': browser_profile.get('user_data_dir'),
			'allowed_domains': browser_profile.get('allowed_domains'),
			'wait_between_actions': browser_profile.get('wait_between_actions'),
			'is_mobile': browser_profile.get('is_mobile'),
			'device_scale_factor': browser_profile.get('device_scale_factor'),
			'disable_security': browser_profile.get('disable_security'),
		},
		'command_history': [],
	}

# From browser_use/cli.py
def load_user_config() -> dict[str, Any]:
	"""Load user configuration using the new config system."""
	# Just get the default config which already loads from the new system
	config = get_default_config()

	# Load command history from a separate file if it exists
	history_file = CONFIG.BROWSER_USE_CONFIG_DIR / 'command_history.json'
	if history_file.exists():
		try:
			with open(history_file) as f:
				config['command_history'] = json.load(f)
		except (FileNotFoundError, json.JSONDecodeError):
			config['command_history'] = []

	return config

# From browser_use/cli.py
def save_user_config(config: dict[str, Any]) -> None:
	"""Save command history only (config is saved via the new system)."""
	# Only save command history to a separate file
	if 'command_history' in config and isinstance(config['command_history'], list):
		# Ensure command history doesn't exceed maximum length
		history = config['command_history']
		if len(history) > MAX_HISTORY_LENGTH:
			history = history[-MAX_HISTORY_LENGTH:]

		# Save to separate history file
		history_file = CONFIG.BROWSER_USE_CONFIG_DIR / 'command_history.json'
		with open(history_file, 'w') as f:
			json.dump(history, f, indent=2)

# From browser_use/cli.py
def update_config_with_click_args(config: dict[str, Any], ctx: click.Context) -> dict[str, Any]:
	"""Update configuration with command-line arguments."""
	# Ensure required sections exist
	if 'model' not in config:
		config['model'] = {}
	if 'browser' not in config:
		config['browser'] = {}

	# Update configuration with command-line args if provided
	if ctx.params.get('model'):
		config['model']['name'] = ctx.params['model']
	if ctx.params.get('headless') is not None:
		config['browser']['headless'] = ctx.params['headless']
	if ctx.params.get('window_width'):
		config['browser']['window_width'] = ctx.params['window_width']
	if ctx.params.get('window_height'):
		config['browser']['window_height'] = ctx.params['window_height']
	if ctx.params.get('user_data_dir'):
		config['browser']['user_data_dir'] = ctx.params['user_data_dir']
	if ctx.params.get('profile_directory'):
		config['browser']['profile_directory'] = ctx.params['profile_directory']
	if ctx.params.get('cdp_url'):
		config['browser']['cdp_url'] = ctx.params['cdp_url']

	return config

# From browser_use/cli.py
def setup_readline_history(history: list[str]) -> None:
	"""Set up readline with command history."""
	if not READLINE_AVAILABLE:
		return

	# Add history items to readline
	for item in history:
		readline.add_history(item)

# From browser_use/cli.py
def get_llm(config: dict[str, Any]):
	"""Get the language model based on config and available API keys."""
	model_config = config.get('model', {})
	model_name = model_config.get('name')
	temperature = model_config.get('temperature', 0.0)

	# Get API key from config or environment
	api_key = model_config.get('api_keys', {}).get('OPENAI_API_KEY') or CONFIG.OPENAI_API_KEY

	if model_name:
		if model_name.startswith('gpt'):
			if not api_key and not CONFIG.OPENAI_API_KEY:
				print('⚠️  OpenAI API key not found. Please update your config or set OPENAI_API_KEY environment variable.')
				sys.exit(1)
			return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key or CONFIG.OPENAI_API_KEY)
		elif model_name.startswith('claude'):
			if not CONFIG.ANTHROPIC_API_KEY:
				print('⚠️  Anthropic API key not found. Please update your config or set ANTHROPIC_API_KEY environment variable.')
				sys.exit(1)
			return ChatAnthropic(model=model_name, temperature=temperature)
		elif model_name.startswith('gemini'):
			if not CONFIG.GOOGLE_API_KEY:
				print('⚠️  Google API key not found. Please update your config or set GOOGLE_API_KEY environment variable.')
				sys.exit(1)
			return ChatGoogle(model=model_name, temperature=temperature)

	# Auto-detect based on available API keys
	if api_key or CONFIG.OPENAI_API_KEY:
		return ChatOpenAI(model='gpt-4o', temperature=temperature, api_key=api_key or CONFIG.OPENAI_API_KEY)
	elif CONFIG.ANTHROPIC_API_KEY:
		return ChatAnthropic(model='claude-3-5-sonnet-20241022', temperature=temperature)
	elif CONFIG.GOOGLE_API_KEY:
		return ChatGoogle(model='gemini-2.0-flash-exp', temperature=temperature)
	else:
		print(
			'⚠️  No API keys found. Please update your config or set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY.'
		)
		sys.exit(1)

# From browser_use/cli.py
def main(ctx: click.Context, debug: bool = False, **kwargs):
	"""Browser-Use Interactive TUI or Command Line Executor

	Use --user-data-dir to specify a local Chrome profile directory.
	Common Chrome profile locations:
	  macOS: ~/Library/Application Support/Google/Chrome
	  Linux: ~/.config/google-chrome
	  Windows: %LOCALAPPDATA%\\Google\\Chrome\\User Data

	Use --profile-directory to specify which profile within the user data directory.
	Examples: "Default", "Profile 1", "Profile 2", etc.
	"""

	if kwargs['version']:
		from importlib.metadata import version

		print(version('browser-use'))
		sys.exit(0)

	# Check if MCP server mode is activated
	if kwargs.get('mcp'):
		# Capture telemetry for MCP server mode via CLI
		telemetry = ProductTelemetry()
		telemetry.capture(
			CLITelemetryEvent(
				version=get_browser_use_version(),
				action='start',
				mode='mcp_server',
			)
		)
		# Run as MCP server
		from browser_use.mcp.server import main as mcp_main

		asyncio.run(mcp_main())
		return

	# Check if prompt mode is activated
	if kwargs.get('prompt'):
		# Set environment variable for prompt mode before running
		os.environ['BROWSER_USE_LOGGING_LEVEL'] = 'result'
		# Run in non-interactive mode
		asyncio.run(run_prompt_mode(kwargs['prompt'], ctx, debug))
		return

	# Configure console logging
	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S'))

	# Configure root logger
	root_logger = logging.getLogger()
	root_logger.setLevel(logging.INFO if not debug else logging.DEBUG)
	root_logger.addHandler(console_handler)

	logger = logging.getLogger('browser_use.startup')
	logger.info('Starting Browser-Use initialization')
	if debug:
		logger.debug(f'System info: Python {sys.version.split()[0]}, Platform: {sys.platform}')

	logger.debug('Loading environment variables from .env file...')
	load_dotenv()
	logger.debug('Environment variables loaded')

	# Load user configuration
	logger.debug('Loading user configuration...')
	try:
		config = load_user_config()
		logger.debug(f'User configuration loaded from {CONFIG.BROWSER_USE_CONFIG_FILE}')
	except Exception as e:
		logger.error(f'Error loading user configuration: {str(e)}', exc_info=True)
		print(f'Error loading configuration: {str(e)}')
		sys.exit(1)

	# Update config with command-line arguments
	logger.debug('Updating configuration with command line arguments...')
	try:
		config = update_config_with_click_args(config, ctx)
		logger.debug('Configuration updated')
	except Exception as e:
		logger.error(f'Error updating config with command line args: {str(e)}', exc_info=True)
		print(f'Error updating configuration: {str(e)}')
		sys.exit(1)

	# Save updated config
	logger.debug('Saving user configuration...')
	try:
		save_user_config(config)
		logger.debug('Configuration saved')
	except Exception as e:
		logger.error(f'Error saving user configuration: {str(e)}', exc_info=True)
		print(f'Error saving configuration: {str(e)}')
		sys.exit(1)

	# Setup handlers for console output before entering Textual UI
	logger.debug('Setting up handlers for Textual UI...')

	# Log browser and model configuration that will be used
	browser_type = 'Chromium'  # BrowserSession only supports Chromium
	model_name = config.get('model', {}).get('name', 'auto-detected')
	headless = config.get('browser', {}).get('headless', False)
	headless_str = 'headless' if headless else 'visible'

	logger.info(f'Preparing {browser_type} browser ({headless_str}) with {model_name} LLM')

	try:
		# Run the Textual UI interface - now all the initialization happens before we go fullscreen
		logger.debug('Starting Textual UI interface...')
		asyncio.run(textual_interface(config))
	except Exception as e:
		# Restore console logging for error reporting
		root_logger.setLevel(logging.INFO)
		for handler in root_logger.handlers:
			root_logger.removeHandler(handler)
		root_logger.addHandler(console_handler)

		logger.error(f'Error initializing Browser-Use: {str(e)}', exc_info=debug)
		print(f'\nError launching Browser-Use: {str(e)}')
		if debug:
			import traceback

			traceback.print_exc()
		sys.exit(1)

# From browser_use/cli.py
def emit(self, record):
		try:
			msg = self.format(record)
			self.rich_log.write(msg)
		except Exception:
			self.handleError(record)

# From browser_use/cli.py
def setup_richlog_logging(self) -> None:
		"""Set up logging to redirect to RichLog widget instead of stdout."""
		# Try to add RESULT level if it doesn't exist
		try:
			addLoggingLevel('RESULT', 35)
		except AttributeError:
			pass  # Level already exists, which is fine

		# Get the RichLog widget
		rich_log = self.query_one('#results-log', RichLog)

		# Create and set up the custom handler
		log_handler = RichLogHandler(rich_log)
		log_type = os.getenv('BROWSER_USE_LOGGING_LEVEL', 'result').lower()

		class BrowserUseFormatter(logging.Formatter):
			def format(self, record):
				# if isinstance(record.name, str) and record.name.startswith('browser_use.'):
				# 	record.name = record.name.split('.')[-2]
				return super().format(record)

		# Set up the formatter based on log type
		if log_type == 'result':
			log_handler.setLevel('RESULT')
			log_handler.setFormatter(BrowserUseFormatter('%(message)s'))
		else:
			log_handler.setFormatter(BrowserUseFormatter('%(levelname)-8s [%(name)s] %(message)s'))

		# Configure root logger - Replace ALL handlers, not just stdout handlers
		root = logging.getLogger()

		# Clear all existing handlers and add only our richlog handler
		root.handlers = []
		root.addHandler(log_handler)

		# Set log level based on environment variable
		if log_type == 'result':
			root.setLevel('RESULT')
		elif log_type == 'debug':
			root.setLevel(logging.DEBUG)
		else:
			root.setLevel(logging.INFO)

		# Configure browser_use logger
		browser_use_logger = logging.getLogger('browser_use')
		browser_use_logger.propagate = False  # Don't propagate to root logger
		browser_use_logger.handlers = [log_handler]  # Replace any existing handlers
		browser_use_logger.setLevel(root.level)

		# Silence third-party loggers
		for logger_name in [
			'WDM',
			'httpx',
			'selenium',
			'playwright',
			'urllib3',
			'asyncio',
			'openai',
			'httpcore',
			'charset_normalizer',
			'anthropic._base_client',
			'PIL.PngImagePlugin',
			'trafilatura.htmlprocessing',
			'trafilatura',
		]:
			third_party = logging.getLogger(logger_name)
			third_party.setLevel(logging.ERROR)
			third_party.propagate = False
			third_party.handlers = []

# From browser_use/cli.py
def on_mount(self) -> None:
		"""Set up components when app is mounted."""
		# We'll use a file logger since stdout is now controlled by Textual
		logger = logging.getLogger('browser_use.on_mount')
		logger.debug('on_mount() method started')

		# Step 1: Set up custom logging to RichLog
		logger.debug('Setting up RichLog logging...')
		try:
			self.setup_richlog_logging()
			logger.debug('RichLog logging set up successfully')
		except Exception as e:
			logger.error(f'Error setting up RichLog logging: {str(e)}', exc_info=True)
			raise RuntimeError(f'Failed to set up RichLog logging: {str(e)}')

		# Step 2: Set up input history
		logger.debug('Setting up readline history...')
		try:
			if READLINE_AVAILABLE and self.task_history:
				for item in self.task_history:
					readline.add_history(item)
				logger.debug(f'Added {len(self.task_history)} items to readline history')
			else:
				logger.debug('No readline history to set up')
		except Exception as e:
			logger.error(f'Error setting up readline history: {str(e)}', exc_info=False)
			# Non-critical, continue

		# Step 3: Focus the input field
		logger.debug('Focusing input field...')
		try:
			input_field = self.query_one('#task-input', Input)
			input_field.focus()
			logger.debug('Input field focused')
		except Exception as e:
			logger.error(f'Error focusing input field: {str(e)}', exc_info=True)
			# Non-critical, continue

		# Step 5: Start continuous info panel updates
		logger.debug('Starting info panel updates...')
		try:
			self.update_info_panels()
			logger.debug('Info panel updates started')
		except Exception as e:
			logger.error(f'Error starting info panel updates: {str(e)}', exc_info=True)
			# Non-critical, continue

		# Capture telemetry for CLI start
		self._telemetry.capture(
			CLITelemetryEvent(
				version=get_browser_use_version(),
				action='start',
				mode='interactive',
				model=self.llm.model if self.llm and hasattr(self.llm, 'model') else None,
				model_provider=self.llm.provider if self.llm and hasattr(self.llm, 'provider') else None,
			)
		)

		logger.debug('on_mount() completed successfully')

# From browser_use/cli.py
def on_input_key_up(self, event: events.Key) -> None:
		"""Handle up arrow key in the input field."""
		# For textual key events, we need to check focus manually
		input_field = self.query_one('#task-input', Input)
		if not input_field.has_focus:
			return

		# Only process if we have history
		if not self.task_history:
			return

		# Move back in history if possible
		if self.history_index > 0:
			self.history_index -= 1
			task_input = self.query_one('#task-input', Input)
			task_input.value = self.task_history[self.history_index]
			# Move cursor to end of text
			task_input.cursor_position = len(task_input.value)

		# Prevent default behavior (cursor movement)
		event.prevent_default()
		event.stop()

# From browser_use/cli.py
def on_input_key_down(self, event: events.Key) -> None:
		"""Handle down arrow key in the input field."""
		# For textual key events, we need to check focus manually
		input_field = self.query_one('#task-input', Input)
		if not input_field.has_focus:
			return

		# Only process if we have history
		if not self.task_history:
			return

		# Move forward in history or clear input if at the end
		if self.history_index < len(self.task_history) - 1:
			self.history_index += 1
			task_input = self.query_one('#task-input', Input)
			task_input.value = self.task_history[self.history_index]
			# Move cursor to end of text
			task_input.cursor_position = len(task_input.value)
		elif self.history_index == len(self.task_history) - 1:
			# At the end of history, go to "new line" state
			self.history_index += 1
			self.query_one('#task-input', Input).value = ''

		# Prevent default behavior (cursor movement)
		event.prevent_default()
		event.stop()

# From browser_use/cli.py
def on_input_submitted(self, event: Input.Submitted) -> None:
		"""Handle task input submission."""
		if event.input.id == 'task-input':
			task = event.input.value
			if not task.strip():
				return

			# Add to history if it's new
			if task.strip() and (not self.task_history or task != self.task_history[-1]):
				self.task_history.append(task)
				self.config['command_history'] = self.task_history
				save_user_config(self.config)

			# Reset history index to point past the end of history
			self.history_index = len(self.task_history)

			# Hide logo, links, and paths panels
			self.hide_intro_panels()

			# Process the task
			self.run_task(task)

			# Clear the input
			event.input.value = ''

# From browser_use/cli.py
def hide_intro_panels(self) -> None:
		"""Hide the intro panels, show info panels, and expand the log view."""
		try:
			# Get the panels
			logo_panel = self.query_one('#logo-panel')
			links_panel = self.query_one('#links-panel')
			paths_panel = self.query_one('#paths-panel')
			info_panels = self.query_one('#info-panels')
			tasks_panel = self.query_one('#tasks-panel')
			# Hide intro panels if they're visible and show info panels
			if logo_panel.display:
				# Log for debugging
				logging.info('Hiding intro panels and showing info panels')

				logo_panel.display = False
				links_panel.display = False
				paths_panel.display = False

				# Show info panels
				info_panels.display = True
				tasks_panel.display = True

				# Make results container take full height
				results_container = self.query_one('#results-container')
				results_container.styles.height = '1fr'

				# Configure the log
				results_log = self.query_one('#results-log')
				results_log.styles.height = 'auto'

				logging.info('Panels should now be visible')
		except Exception as e:
			logging.error(f'Error in hide_intro_panels: {str(e)}')

# From browser_use/cli.py
def update_info_panels(self) -> None:
		"""Update all information panels with current state."""
		try:
			# Update actual content
			self.update_browser_panel()
			self.update_model_panel()
			self.update_tasks_panel()
		except Exception as e:
			logging.error(f'Error in update_info_panels: {str(e)}')
		finally:
			# Always schedule the next update - will update at 1-second intervals
			# This ensures continuous updates even if agent state changes
			self.set_timer(1.0, self.update_info_panels)

# From browser_use/cli.py
def update_browser_panel(self) -> None:
		"""Update browser information panel with details about the browser."""
		browser_info = self.query_one('#browser-info', RichLog)
		browser_info.clear()

		# Try to use the agent's browser session if available
		browser_session = self.browser_session
		if hasattr(self, 'agent') and self.agent and hasattr(self.agent, 'browser_session'):
			browser_session = self.agent.browser_session

		if browser_session:
			try:
				# Check if browser session has a browser context
				if not hasattr(browser_session, 'browser_context') or browser_session.browser_context is None:
					browser_info.write('[yellow]Browser session created, waiting for browser to launch...[/]')
					return

				# Update our reference if we're using the agent's session
				if browser_session != self.browser_session:
					self.browser_session = browser_session

				# Get basic browser info from browser_profile
				browser_type = 'Chromium'
				headless = browser_session.browser_profile.headless

				# Determine connection type based on config
				connection_type = 'playwright'  # Default
				if browser_session.cdp_url:
					connection_type = 'CDP'
				elif browser_session.wss_url:
					connection_type = 'WSS'
				elif browser_session.browser_profile.executable_path:
					connection_type = 'user-provided'

				# Get window size details from browser_profile
				window_width = None
				window_height = None
				if browser_session.browser_profile.viewport:
					window_width = browser_session.browser_profile.viewport.get('width')
					window_height = browser_session.browser_profile.viewport.get('height')

				# Try to get browser PID
				browser_pid = 'Unknown'
				connected = False
				browser_status = '[red]Disconnected[/]'

				try:
					# Check if browser PID is available
					if hasattr(browser_session, 'browser_pid') and browser_session.browser_pid:
						browser_pid = str(browser_session.browser_pid)
						connected = True
						browser_status = '[green]Connected[/]'
					# Otherwise just check if we have a browser context
					elif browser_session.browser_context is not None:
						connected = True
						browser_status = '[green]Connected[/]'
						browser_pid = 'N/A'
				except Exception as e:
					browser_pid = f'Error: {str(e)}'

				# Display browser information
				browser_info.write(f'[bold cyan]Chromium[/] Browser ({browser_status})')
				browser_info.write(
					f'Type: [yellow]{connection_type}[/] [{"green" if not headless else "red"}]{" (headless)" if headless else ""}[/]'
				)
				browser_info.write(f'PID: [dim]{browser_pid}[/]')
				browser_info.write(f'CDP Port: {browser_session.cdp_url}')

				if window_width and window_height:
					browser_info.write(f'Window: [blue]{window_width}[/] × [blue]{window_height}[/]')

				# Include additional information about the browser if needed
				if connected and hasattr(self, 'agent') and self.agent:
					try:
						# Show when the browser was connected
						timestamp = int(time.time())
						current_time = time.strftime('%H:%M:%S', time.localtime(timestamp))
						browser_info.write(f'Last updated: [dim]{current_time}[/]')
					except Exception:
						pass

					# Show the agent's current page URL if available
					if browser_session.agent_current_page:
						current_url = (
							browser_session.agent_current_page.url.replace('https://', '')
							.replace('http://', '')
							.replace('www.', '')[:36]
							+ '…'
						)
						browser_info.write(f'👁️  [green]{current_url}[/]')
			except Exception as e:
				browser_info.write(f'[red]Error updating browser info: {str(e)}[/]')
		else:
			browser_info.write('[red]Browser not initialized[/]')

# From browser_use/cli.py
def update_model_panel(self) -> None:
		"""Update model information panel with details about the LLM."""
		model_info = self.query_one('#model-info', RichLog)
		model_info.clear()

		if self.llm:
			# Get model details
			model_name = 'Unknown'
			if hasattr(self.llm, 'model_name'):
				model_name = self.llm.model_name
			elif hasattr(self.llm, 'model'):
				model_name = self.llm.model

			# Show model name
			if self.agent:
				temp_str = f'{self.llm.temperature}ºC ' if self.llm.temperature else ''
				vision_str = '+ vision ' if self.agent.settings.use_vision else ''
				planner_str = '+ planner' if self.agent.settings.planner_llm else ''
				model_info.write(
					f'[white]LLM:[/] [blue]{self.llm.__class__.__name__} [yellow]{model_name}[/] {temp_str}{vision_str}{planner_str}'
				)
			else:
				model_info.write(f'[white]LLM:[/] [blue]{self.llm.__class__.__name__} [yellow]{model_name}[/]')

			# Show token usage statistics if agent exists and has history
			if self.agent and hasattr(self.agent, 'state') and hasattr(self.agent.state, 'history'):
				# Get total tokens used
				# total_tokens = self.agent.history.total_input_tokens()
				# model_info.write(f'[white]Input tokens:[/] [green]{total_tokens:,}[/]')

				# Calculate tokens per step
				num_steps = len(self.agent.history.history)
				# if num_steps > 0:
				# avg_tokens_per_step = total_tokens / num_steps
				# model_info.write(f'[white]Avg tokens/step:[/] [green]{avg_tokens_per_step:,.1f}[/]')

				# Get the last step metadata to show the most recent LLM response time
				if num_steps > 0 and self.agent.history.history[-1].metadata:
					last_step = self.agent.history.history[-1]
					if last_step.metadata:
						step_duration = last_step.metadata.duration_seconds
					else:
						step_duration = 0
					# step_tokens = last_step.metadata.input_tokens

					# if step_tokens > 0:
					# 	tokens_per_second = step_tokens / step_duration if step_duration > 0 else 0
					# 	model_info.write(f'[white]Avg tokens/sec:[/] [magenta]{tokens_per_second:.1f}[/]')

				# Show total duration
				total_duration = self.agent.history.total_duration_seconds()
				if total_duration > 0:
					model_info.write(f'[white]Total Duration:[/] [magenta]{total_duration:.2f}s[/]')

					# Calculate response time metrics
					model_info.write(f'[white]Last Step Duration:[/] [magenta]{step_duration:.2f}s[/]')

				# Add current state information
				if hasattr(self.agent, 'running'):
					if getattr(self.agent, 'running', False):
						model_info.write('[yellow]LLM is thinking[blink]...[/][/]')
					elif hasattr(self.agent, 'state') and hasattr(self.agent.state, 'paused') and self.agent.state.paused:
						model_info.write('[orange]LLM paused[/]')
		else:
			model_info.write('[red]Model not initialized[/]')

# From browser_use/cli.py
def update_tasks_panel(self) -> None:
		"""Update tasks information panel with details about the tasks and steps hierarchy."""
		tasks_info = self.query_one('#tasks-info', RichLog)
		tasks_info.clear()

		if self.agent:
			# Check if agent has tasks
			task_history = []
			message_history = []

			# Try to extract tasks by looking at message history
			if hasattr(self.agent, '_message_manager') and self.agent._message_manager:
				message_history = self.agent._message_manager.state.history.get_messages()

				# Extract original task(s)
				original_tasks = []
				for msg in message_history:
					if hasattr(msg, 'content'):
						content = msg.content
						if isinstance(content, str) and 'Your ultimate task is:' in content:
							task_text = content.split('"""')[1].strip()
							original_tasks.append(task_text)

				if original_tasks:
					tasks_info.write('[bold green]TASK:[/]')
					for i, task in enumerate(original_tasks, 1):
						# Only show latest task if multiple task changes occurred
						if i == len(original_tasks):
							tasks_info.write(f'[white]{task}[/]')
					tasks_info.write('')

			# Get current state information
			current_step = self.agent.state.n_steps if hasattr(self.agent, 'state') else 0

			# Get all agent history items
			history_items = []
			if hasattr(self.agent, 'state') and hasattr(self.agent.state, 'history'):
				history_items = self.agent.history.history

				if history_items:
					tasks_info.write('[bold yellow]STEPS:[/]')

					for idx, item in enumerate(history_items, 1):
						# Determine step status
						step_style = '[green]✓[/]'

						# For the current step, show it as in progress
						if idx == current_step:
							step_style = '[yellow]⟳[/]'

						# Check if this step had an error
						if item.result and any(result.error for result in item.result):
							step_style = '[red]✗[/]'

						# Show step number
						tasks_info.write(f'{step_style} Step {idx}/{current_step}')

						# Show goal if available
						if item.model_output and hasattr(item.model_output, 'current_state'):
							# Show goal for this step
							goal = item.model_output.current_state.next_goal
							if goal:
								# Take just the first line for display
								goal_lines = goal.strip().split('\n')
								goal_summary = goal_lines[0]
								tasks_info.write(f'   [cyan]Goal:[/] {goal_summary}')

							# Show evaluation of previous goal (feedback)
							eval_prev = item.model_output.current_state.evaluation_previous_goal
							if eval_prev and idx > 1:  # Only show for steps after the first
								eval_lines = eval_prev.strip().split('\n')
								eval_summary = eval_lines[0]
								eval_summary = eval_summary.replace('Success', '✅ ').replace('Failed', '❌ ').strip()
								tasks_info.write(f'   [tan]Evaluation:[/] {eval_summary}')

						# Show actions taken in this step
						if item.model_output and item.model_output.action:
							tasks_info.write('   [purple]Actions:[/]')
							for action_idx, action in enumerate(item.model_output.action, 1):
								action_type = action.__class__.__name__
								if hasattr(action, 'model_dump'):
									# For proper actions, show the action type
									action_dict = action.model_dump(exclude_unset=True)
									if action_dict:
										action_name = list(action_dict.keys())[0]
										tasks_info.write(f'     {action_idx}. [blue]{action_name}[/]')

						# Show results or errors from this step
						if item.result:
							for result in item.result:
								if result.error:
									error_text = result.error
									tasks_info.write(f'   [red]Error:[/] {error_text}')
								elif result.extracted_content:
									content = result.extracted_content
									tasks_info.write(f'   [green]Result:[/] {content}')

						# Add a space between steps for readability
						tasks_info.write('')

			# If agent is actively running, show a status indicator
			if hasattr(self.agent, 'running') and getattr(self.agent, 'running', False):
				tasks_info.write('[yellow]Agent is actively working[blink]...[/][/]')
			elif hasattr(self.agent, 'state') and hasattr(self.agent.state, 'paused') and self.agent.state.paused:
				tasks_info.write('[orange]Agent is paused (press Enter to resume)[/]')
		else:
			tasks_info.write('[dim]Agent not initialized[/]')

		# Force scroll to bottom
		tasks_panel = self.query_one('#tasks-panel')
		tasks_panel.scroll_end(animate=False)

# From browser_use/cli.py
def scroll_to_input(self) -> None:
		"""Scroll to the input field to ensure it's visible."""
		input_container = self.query_one('#task-input-container')
		input_container.scroll_visible()

# From browser_use/cli.py
def run_task(self, task: str) -> None:
		"""Launch the task in a background worker."""
		# Create or update the agent
		agent_settings = AgentSettings.model_validate(self.config.get('agent', {}))

		# Get the logger
		logger = logging.getLogger('browser_use.app')

		# Make sure intro is hidden and log is ready
		self.hide_intro_panels()

		# Start continuous updates of all info panels
		self.update_info_panels()

		# Clear the log to start fresh
		rich_log = self.query_one('#results-log', RichLog)
		rich_log.clear()

		if self.agent is None:
			if not self.llm:
				raise RuntimeError('LLM not initialized')
			self.agent = Agent(
				task=task,
				llm=self.llm,
				controller=self.controller if self.controller else Controller(),
				browser_session=self.browser_session,
				source='cli',
				**agent_settings.model_dump(),
			)
			# Update our browser_session reference to point to the agent's
			if hasattr(self.agent, 'browser_session'):
				self.browser_session = self.agent.browser_session
		else:
			self.agent.add_new_task(task)

		# Let the agent run in the background
		async def agent_task_worker() -> None:
			logger.debug('\n🚀 Working on task: %s', task)

			# Set flags to indicate the agent is running
			if self.agent:
				self.agent.running = True  # type: ignore
				self.agent.last_response_time = 0  # type: ignore

			# Panel updates are already happening via the timer in update_info_panels

			task_start_time = time.time()
			error_msg = None

			try:
				# Capture telemetry for message sent
				self._telemetry.capture(
					CLITelemetryEvent(
						version=get_browser_use_version(),
						action='message_sent',
						mode='interactive',
						model=self.llm.model if self.llm and hasattr(self.llm, 'model') else None,
						model_provider=self.llm.provider if self.llm and hasattr(self.llm, 'provider') else None,
					)
				)

				# Run the agent task, redirecting output to RichLog through our handler
				if self.agent:
					await self.agent.run()
			except Exception as e:
				error_msg = str(e)
				logger.error('\nError running agent: %s', str(e))
			finally:
				# Clear the running flag
				if self.agent:
					self.agent.running = False  # type: ignore

				# No need to call update_info_panels() here as it's already updating via timer

				# Capture telemetry for task completion
				duration = time.time() - task_start_time
				self._telemetry.capture(
					CLITelemetryEvent(
						version=get_browser_use_version(),
						action='task_completed' if error_msg is None else 'error',
						mode='interactive',
						model=self.llm.model if self.llm and hasattr(self.llm, 'model') else None,
						model_provider=self.llm.provider if self.llm and hasattr(self.llm, 'provider') else None,
						duration_seconds=duration,
						error_message=error_msg,
					)
				)

				logger.debug('\n✅ Task completed!')

				# Make sure the task input container is visible
				task_input_container = self.query_one('#task-input-container')
				task_input_container.display = True

				# Refocus the input field
				input_field = self.query_one('#task-input', Input)
				input_field.focus()

				# Ensure the input is visible by scrolling to it
				self.call_after_refresh(self.scroll_to_input)

		# Run the worker
		self.run_worker(agent_task_worker, name='agent_task')

# From browser_use/cli.py
def action_input_history_prev(self) -> None:
		"""Navigate to the previous item in command history."""
		# Only process if we have history and input is focused
		input_field = self.query_one('#task-input', Input)
		if not input_field.has_focus or not self.task_history:
			return

		# Move back in history if possible
		if self.history_index > 0:
			self.history_index -= 1
			input_field.value = self.task_history[self.history_index]
			# Move cursor to end of text
			input_field.cursor_position = len(input_field.value)

# From browser_use/cli.py
def action_input_history_next(self) -> None:
		"""Navigate to the next item in command history or clear input."""
		# Only process if we have history and input is focused
		input_field = self.query_one('#task-input', Input)
		if not input_field.has_focus or not self.task_history:
			return

		# Move forward in history or clear input if at the end
		if self.history_index < len(self.task_history) - 1:
			self.history_index += 1
			input_field.value = self.task_history[self.history_index]
			# Move cursor to end of text
			input_field.cursor_position = len(input_field.value)
		elif self.history_index == len(self.task_history) - 1:
			# At the end of history, go to "new line" state
			self.history_index += 1
			input_field.value = ''

# From browser_use/cli.py
def compose(self) -> ComposeResult:
		"""Create the UI layout."""
		yield Header()

		# Main container for app content
		with Container(id='main-container'):
			# Logo panel
			yield Static(BROWSER_LOGO, id='logo-panel', markup=True)

			# Information panels (hidden by default)
			with Container(id='info-panels'):
				# Top row with browser and model panels side by side
				with Container(id='top-panels'):
					# Browser panel
					with Container(id='browser-panel'):
						yield RichLog(id='browser-info', markup=True, highlight=True, wrap=True)

					# Model panel
					with Container(id='model-panel'):
						yield RichLog(id='model-info', markup=True, highlight=True, wrap=True)

				# Tasks panel (full width, below browser and model)
				with VerticalScroll(id='tasks-panel'):
					yield RichLog(id='tasks-info', markup=True, highlight=True, wrap=True, auto_scroll=True)

			# Links panel with URLs
			with Container(id='links-panel'):
				with HorizontalGroup(classes='link-row'):
					yield Static('Run at scale on cloud:    [blink]☁️[/]  ', markup=True, classes='link-label')
					yield Link('https://browser-use.com', url='https://browser-use.com', classes='link-white link-url')

				yield Static('')  # Empty line

				with HorizontalGroup(classes='link-row'):
					yield Static('Chat & share on Discord:  🚀 ', markup=True, classes='link-label')
					yield Link(
						'https://discord.gg/ESAUZAdxXY', url='https://discord.gg/ESAUZAdxXY', classes='link-purple link-url'
					)

				with HorizontalGroup(classes='link-row'):
					yield Static('Get prompt inspiration:   🦸 ', markup=True, classes='link-label')
					yield Link(
						'https://github.com/browser-use/awesome-prompts',
						url='https://github.com/browser-use/awesome-prompts',
						classes='link-magenta link-url',
					)

				with HorizontalGroup(classes='link-row'):
					yield Static('[dim]Report any issues:[/]        🐛 ', markup=True, classes='link-label')
					yield Link(
						'https://github.com/browser-use/browser-use/issues',
						url='https://github.com/browser-use/browser-use/issues',
						classes='link-green link-url',
					)

			# Paths panel
			yield Static(
				f' ⚙️  Settings saved to:              {str(CONFIG.BROWSER_USE_CONFIG_FILE.resolve()).replace(str(Path.home()), "~")}\n'
				f' 📁 Outputs & recordings saved to:  {str(Path(".").resolve()).replace(str(Path.home()), "~")}',
				id='paths-panel',
				markup=True,
			)

			# Results view with scrolling (place this before input to make input sticky at bottom)
			with VerticalScroll(id='results-container'):
				yield RichLog(highlight=True, markup=True, id='results-log', wrap=True, auto_scroll=True)

			# Task input container (now at the bottom)
			with Container(id='task-input-container'):
				yield Label('🔍 What would you like me to do on the web?', id='task-label')
				yield Input(placeholder='Enter your task...', id='task-input')

		yield Footer()

# From browser_use/cli.py
def setup_textual_logging():
		# Replace all handlers with null handler
		root_logger = logging.getLogger()
		for handler in root_logger.handlers:
			root_logger.removeHandler(handler)

		# Add null handler to ensure no output to stdout/stderr
		null_handler = logging.NullHandler()
		root_logger.addHandler(null_handler)
		logger.debug('Logging configured for Textual UI')

# From browser_use/cli.py
def format(self, record):
				# if isinstance(record.name, str) and record.name.startswith('browser_use.'):
				# 	record.name = record.name.split('.')[-2]
				return super().format(record)

from typing import Literal
from typing import cast
from lmnr import observe

# From browser_use/observability.py
def observe(
	name: str | None = None,
	ignore_input: bool = False,
	ignore_output: bool = False,
	metadata: dict[str, Any] | None = None,
	span_type: Literal['DEFAULT', 'LLM', 'TOOL'] = 'DEFAULT',
	**kwargs: Any,
) -> Callable[[F], F]:
	"""
	Observability decorator that traces function execution when lmnr is available.

	This decorator will use lmnr's observe decorator if lmnr is installed,
	otherwise it will be a no-op that accepts the same parameters.

	Args:
	    name: Name of the span/trace
	    ignore_input: Whether to ignore function input parameters in tracing
	    ignore_output: Whether to ignore function output in tracing
	    metadata: Additional metadata to attach to the span
	    **kwargs: Additional parameters passed to lmnr observe

	Returns:
	    Decorated function that may be traced depending on lmnr availability

	Example:
	    @observe(name="my_function", metadata={"version": "1.0"})
	    def my_function(param1, param2):
	        return param1 + param2
	"""
	kwargs = {
		'name': name,
		'ignore_input': ignore_input,
		'ignore_output': ignore_output,
		'metadata': metadata,
		'span_type': span_type,
		**kwargs,
	}

	if _LMNR_AVAILABLE and _lmnr_observe:
		# Use the real lmnr observe decorator
		return cast(Callable[[F], F], _lmnr_observe(**kwargs))
	else:
		# Use no-op decorator
		return _create_no_op_decorator(**kwargs)

# From browser_use/observability.py
def observe_debug(
	name: str | None = None,
	ignore_input: bool = False,
	ignore_output: bool = False,
	metadata: dict[str, Any] | None = None,
	span_type: Literal['DEFAULT', 'LLM', 'TOOL'] = 'DEFAULT',
	**kwargs: Any,
) -> Callable[[F], F]:
	"""
	Debug-only observability decorator that only traces when in debug mode.

	This decorator will use lmnr's observe decorator if both lmnr is installed
	AND we're in debug mode, otherwise it will be a no-op.

	Debug mode is determined by:
	- DEBUG environment variable set to 1/true/yes/on
	- BROWSER_USE_DEBUG environment variable set to 1/true/yes/on
	- Root logging level set to DEBUG or lower

	Args:
	    name: Name of the span/trace
	    ignore_input: Whether to ignore function input parameters in tracing
	    ignore_output: Whether to ignore function output in tracing
	    metadata: Additional metadata to attach to the span
	    **kwargs: Additional parameters passed to lmnr observe

	Returns:
	    Decorated function that may be traced only in debug mode

	Example:
	    @observe_debug(ignore_input=True, ignore_output=True,name="debug_function", metadata={"debug": True})
	    def debug_function(param1, param2):
	        return param1 + param2
	"""
	kwargs = {
		'name': name,
		'ignore_input': ignore_input,
		'ignore_output': ignore_output,
		'metadata': metadata,
		'span_type': span_type,
		**kwargs,
	}

	if _LMNR_AVAILABLE and _lmnr_observe and _is_debug_mode():
		# Use the real lmnr observe decorator only in debug mode
		return cast(Callable[[F], F], _lmnr_observe(**kwargs))
	else:
		# Use no-op decorator (either not in debug mode or lmnr not available)
		return _create_no_op_decorator(**kwargs)

# From browser_use/observability.py
def is_lmnr_available() -> bool:
	"""Check if lmnr is available for tracing."""
	return _LMNR_AVAILABLE

# From browser_use/observability.py
def is_debug_mode() -> bool:
	"""Check if we're currently in debug mode."""
	return _is_debug_mode()

# From browser_use/observability.py
def get_observability_status() -> dict[str, bool]:
	"""Get the current status of observability features."""
	return {
		'lmnr_available': _LMNR_AVAILABLE,
		'debug_mode': _is_debug_mode(),
		'observe_active': _LMNR_AVAILABLE,
		'observe_debug_active': _LMNR_AVAILABLE and _is_debug_mode(),
	}


# From browser_use/exceptions.py
class LLMException(Exception):
	def __init__(self, status_code, message):
		self.status_code = status_code
		self.message = message
		super().__init__(f'Error {status_code}: {message}')


# From browser_use/logging_config.py
def addLoggingLevel(levelName, levelNum, methodName=None):
	"""
	Comprehensively adds a new logging level to the `logging` module and the
	currently configured logging class.

	`levelName` becomes an attribute of the `logging` module with the value
	`levelNum`. `methodName` becomes a convenience method for both `logging`
	itself and the class returned by `logging.getLoggerClass()` (usually just
	`logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
	used.

	To avoid accidental clobberings of existing attributes, this method will
	raise an `AttributeError` if the level name is already an attribute of the
	`logging` module or if the method name is already present

	Example
	-------
	>>> addLoggingLevel('TRACE', logging.DEBUG - 5)
	>>> logging.getLogger(__name__).setLevel('TRACE')
	>>> logging.getLogger(__name__).trace('that worked')
	>>> logging.trace('so did this')
	>>> logging.TRACE
	5

	"""
	if not methodName:
		methodName = levelName.lower()

	if hasattr(logging, levelName):
		raise AttributeError(f'{levelName} already defined in logging module')
	if hasattr(logging, methodName):
		raise AttributeError(f'{methodName} already defined in logging module')
	if hasattr(logging.getLoggerClass(), methodName):
		raise AttributeError(f'{methodName} already defined in logger class')

	# This method was inspired by the answers to Stack Overflow post
	# http://stackoverflow.com/q/2183233/2988730, especially
	# http://stackoverflow.com/a/13638084/2988730
	def logForLevel(self, message, *args, **kwargs):
		if self.isEnabledFor(levelNum):
			self._log(levelNum, message, args, **kwargs)

	def logToRoot(message, *args, **kwargs):
		logging.log(levelNum, message, *args, **kwargs)

	logging.addLevelName(levelNum, levelName)
	setattr(logging, levelName, levelNum)
	setattr(logging.getLoggerClass(), methodName, logForLevel)
	setattr(logging, methodName, logToRoot)

# From browser_use/logging_config.py
def setup_logging(stream=None, log_level=None, force_setup=False):
	"""Setup logging configuration for browser-use.

	Args:
		stream: Output stream for logs (default: sys.stdout). Can be sys.stderr for MCP mode.
		log_level: Override log level (default: uses CONFIG.BROWSER_USE_LOGGING_LEVEL)
		force_setup: Force reconfiguration even if handlers already exist
	"""
	# Try to add RESULT level, but ignore if it already exists
	try:
		addLoggingLevel('RESULT', 35)  # This allows ERROR, FATAL and CRITICAL
	except AttributeError:
		pass  # Level already exists, which is fine

	log_type = log_level or CONFIG.BROWSER_USE_LOGGING_LEVEL

	# Check if handlers are already set up
	if logging.getLogger().hasHandlers() and not force_setup:
		return logging.getLogger('browser_use')

	# Clear existing handlers
	root = logging.getLogger()
	root.handlers = []

	class BrowserUseFormatter(logging.Formatter):
		def format(self, record):
			# if isinstance(record.name, str) and record.name.startswith('browser_use.'):
			# 	record.name = record.name.split('.')[-2]
			return super().format(record)

	# Setup single handler for all loggers
	console = logging.StreamHandler(stream or sys.stdout)

	# adittional setLevel here to filter logs
	if log_type == 'result':
		console.setLevel('RESULT')
		console.setFormatter(BrowserUseFormatter('%(message)s'))
	else:
		console.setFormatter(BrowserUseFormatter('%(levelname)-8s [%(name)s] %(message)s'))

	# Configure root logger only
	root.addHandler(console)

	# switch cases for log_type
	if log_type == 'result':
		root.setLevel('RESULT')  # string usage to avoid syntax error
	elif log_type == 'debug':
		root.setLevel(logging.DEBUG)
	else:
		root.setLevel(logging.INFO)

	# Configure browser_use logger
	browser_use_logger = logging.getLogger('browser_use')
	browser_use_logger.propagate = False  # Don't propagate to root logger
	browser_use_logger.addHandler(console)
	browser_use_logger.setLevel(root.level)  # Set same level as root logger

	logger = logging.getLogger('browser_use')
	# logger.info('BrowserUse logging setup complete with level %s', log_type)
	# Silence or adjust third-party loggers
	third_party_loggers = [
		'WDM',
		'httpx',
		'selenium',
		'playwright',
		'urllib3',
		'asyncio',
		'langsmith',
		'langsmith.client',
		'openai',
		'httpcore',
		'charset_normalizer',
		'anthropic._base_client',
		'PIL.PngImagePlugin',
		'trafilatura.htmlprocessing',
		'trafilatura',
		'groq',
		'portalocker',
		'portalocker.utils',
	]
	for logger_name in third_party_loggers:
		third_party = logging.getLogger(logger_name)
		third_party.setLevel(logging.ERROR)
		third_party.propagate = False

	return logger

# From browser_use/logging_config.py
def logForLevel(self, message, *args, **kwargs):
		if self.isEnabledFor(levelNum):
			self._log(levelNum, message, args, **kwargs)

# From browser_use/logging_config.py
def logToRoot(message, *args, **kwargs):
		logging.log(levelNum, message, *args, **kwargs)

from datetime import datetime
from uuid import uuid4
import psutil
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

# From browser_use/config.py
class OldConfig:
	"""Original lazy-loading configuration class for environment variables."""

	# Cache for directory creation tracking
	_dirs_created = False

	@property
	def BROWSER_USE_LOGGING_LEVEL(self) -> str:
		return os.getenv('BROWSER_USE_LOGGING_LEVEL', 'info').lower()

	@property
	def ANONYMIZED_TELEMETRY(self) -> bool:
		return os.getenv('ANONYMIZED_TELEMETRY', 'true').lower()[:1] in 'ty1'

	@property
	def BROWSER_USE_CLOUD_SYNC(self) -> bool:
		return os.getenv('BROWSER_USE_CLOUD_SYNC', str(self.ANONYMIZED_TELEMETRY)).lower()[:1] in 'ty1'

	@property
	def BROWSER_USE_CLOUD_API_URL(self) -> str:
		url = os.getenv('BROWSER_USE_CLOUD_API_URL', 'https://api.browser-use.com')
		assert '://' in url, 'BROWSER_USE_CLOUD_API_URL must be a valid URL'
		return url

	@property
	def BROWSER_USE_CLOUD_UI_URL(self) -> str:
		url = os.getenv('BROWSER_USE_CLOUD_UI_URL', '')
		# Allow empty string as default, only validate if set
		if url and '://' not in url:
			raise AssertionError('BROWSER_USE_CLOUD_UI_URL must be a valid URL if set')
		return url

	# Path configuration
	@property
	def XDG_CACHE_HOME(self) -> Path:
		return Path(os.getenv('XDG_CACHE_HOME', '~/.cache')).expanduser().resolve()

	@property
	def XDG_CONFIG_HOME(self) -> Path:
		return Path(os.getenv('XDG_CONFIG_HOME', '~/.config')).expanduser().resolve()

	@property
	def BROWSER_USE_CONFIG_DIR(self) -> Path:
		path = Path(os.getenv('BROWSER_USE_CONFIG_DIR', str(self.XDG_CONFIG_HOME / 'browseruse'))).expanduser().resolve()
		self._ensure_dirs()
		return path

	@property
	def BROWSER_USE_CONFIG_FILE(self) -> Path:
		return self.BROWSER_USE_CONFIG_DIR / 'config.json'

	@property
	def BROWSER_USE_PROFILES_DIR(self) -> Path:
		path = self.BROWSER_USE_CONFIG_DIR / 'profiles'
		self._ensure_dirs()
		return path

	@property
	def BROWSER_USE_DEFAULT_USER_DATA_DIR(self) -> Path:
		return self.BROWSER_USE_PROFILES_DIR / 'default'

	@property
	def BROWSER_USE_EXTENSIONS_DIR(self) -> Path:
		path = self.BROWSER_USE_CONFIG_DIR / 'extensions'
		self._ensure_dirs()
		return path

	def _ensure_dirs(self) -> None:
		"""Create directories if they don't exist (only once)"""
		if not self._dirs_created:
			config_dir = (
				Path(os.getenv('BROWSER_USE_CONFIG_DIR', str(self.XDG_CONFIG_HOME / 'browseruse'))).expanduser().resolve()
			)
			config_dir.mkdir(parents=True, exist_ok=True)
			(config_dir / 'profiles').mkdir(parents=True, exist_ok=True)
			(config_dir / 'extensions').mkdir(parents=True, exist_ok=True)
			self._dirs_created = True

	# LLM API key configuration
	@property
	def OPENAI_API_KEY(self) -> str:
		return os.getenv('OPENAI_API_KEY', '')

	@property
	def ANTHROPIC_API_KEY(self) -> str:
		return os.getenv('ANTHROPIC_API_KEY', '')

	@property
	def GOOGLE_API_KEY(self) -> str:
		return os.getenv('GOOGLE_API_KEY', '')

	@property
	def DEEPSEEK_API_KEY(self) -> str:
		return os.getenv('DEEPSEEK_API_KEY', '')

	@property
	def GROK_API_KEY(self) -> str:
		return os.getenv('GROK_API_KEY', '')

	@property
	def NOVITA_API_KEY(self) -> str:
		return os.getenv('NOVITA_API_KEY', '')

	@property
	def AZURE_OPENAI_ENDPOINT(self) -> str:
		return os.getenv('AZURE_OPENAI_ENDPOINT', '')

	@property
	def AZURE_OPENAI_KEY(self) -> str:
		return os.getenv('AZURE_OPENAI_KEY', '')

	@property
	def SKIP_LLM_API_KEY_VERIFICATION(self) -> bool:
		return os.getenv('SKIP_LLM_API_KEY_VERIFICATION', 'false').lower()[:1] in 'ty1'

	# Runtime hints
	@property
	def IN_DOCKER(self) -> bool:
		return os.getenv('IN_DOCKER', 'false').lower()[:1] in 'ty1' or is_running_in_docker()

	@property
	def IS_IN_EVALS(self) -> bool:
		return os.getenv('IS_IN_EVALS', 'false').lower()[:1] in 'ty1'

	@property
	def WIN_FONT_DIR(self) -> str:
		return os.getenv('WIN_FONT_DIR', 'C:\\Windows\\Fonts')

# From browser_use/config.py
class FlatEnvConfig(BaseSettings):
	"""All environment variables in a flat namespace."""

	model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', case_sensitive=True, extra='allow')

	# Logging and telemetry
	BROWSER_USE_LOGGING_LEVEL: str = Field(default='info')
	ANONYMIZED_TELEMETRY: bool = Field(default=True)
	BROWSER_USE_CLOUD_SYNC: bool | None = Field(default=None)
	BROWSER_USE_CLOUD_API_URL: str = Field(default='https://api.browser-use.com')
	BROWSER_USE_CLOUD_UI_URL: str = Field(default='')

	# Path configuration
	XDG_CACHE_HOME: str = Field(default='~/.cache')
	XDG_CONFIG_HOME: str = Field(default='~/.config')
	BROWSER_USE_CONFIG_DIR: str | None = Field(default=None)

	# LLM API keys
	OPENAI_API_KEY: str = Field(default='')
	ANTHROPIC_API_KEY: str = Field(default='')
	GOOGLE_API_KEY: str = Field(default='')
	DEEPSEEK_API_KEY: str = Field(default='')
	GROK_API_KEY: str = Field(default='')
	NOVITA_API_KEY: str = Field(default='')
	AZURE_OPENAI_ENDPOINT: str = Field(default='')
	AZURE_OPENAI_KEY: str = Field(default='')
	SKIP_LLM_API_KEY_VERIFICATION: bool = Field(default=False)

	# Runtime hints
	IN_DOCKER: bool | None = Field(default=None)
	IS_IN_EVALS: bool = Field(default=False)
	WIN_FONT_DIR: str = Field(default='C:\\Windows\\Fonts')

	# MCP-specific env vars
	BROWSER_USE_CONFIG_PATH: str | None = Field(default=None)
	BROWSER_USE_HEADLESS: bool | None = Field(default=None)
	BROWSER_USE_ALLOWED_DOMAINS: str | None = Field(default=None)
	BROWSER_USE_LLM_MODEL: str | None = Field(default=None)

# From browser_use/config.py
class DBStyleEntry(BaseModel):
	"""Database-style entry with UUID and metadata."""

	id: str = Field(default_factory=lambda: str(uuid4()))
	default: bool = Field(default=False)
	created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

# From browser_use/config.py
class BrowserProfileEntry(DBStyleEntry):
	"""Browser profile configuration entry - accepts any BrowserProfile fields."""

	model_config = ConfigDict(extra='allow')

	# Common browser profile fields for reference
	headless: bool | None = None
	user_data_dir: str | None = None
	allowed_domains: list[str] | None = None
	downloads_path: str | None = None

# From browser_use/config.py
class LLMEntry(DBStyleEntry):
	"""LLM configuration entry."""

	api_key: str | None = None
	model: str | None = None
	temperature: float | None = None
	max_tokens: int | None = None

# From browser_use/config.py
class AgentEntry(DBStyleEntry):
	"""Agent configuration entry."""

	max_steps: int | None = None
	use_vision: bool | None = None
	system_prompt: str | None = None

# From browser_use/config.py
class DBStyleConfigJSON(BaseModel):
	"""New database-style configuration format."""

	browser_profile: dict[str, BrowserProfileEntry] = Field(default_factory=dict)
	llm: dict[str, LLMEntry] = Field(default_factory=dict)
	agent: dict[str, AgentEntry] = Field(default_factory=dict)

# From browser_use/config.py
class Config:
	"""Backward-compatible configuration class that merges all config sources.

	Re-reads environment variables on every access to maintain compatibility.
	"""

	def __init__(self):
		# Cache for directory creation tracking only
		self._dirs_created = False

	def __getattr__(self, name: str) -> Any:
		"""Dynamically proxy all attributes to fresh instances.

		This ensures env vars are re-read on every access.
		"""
		# Special handling for internal attributes
		if name.startswith('_'):
			raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

		# Create fresh instances on every access
		old_config = OldConfig()

		# Always use old config for all attributes (it handles env vars with proper transformations)
		if hasattr(old_config, name):
			return getattr(old_config, name)

		# For new MCP-specific attributes not in old config
		env_config = FlatEnvConfig()
		if hasattr(env_config, name):
			return getattr(env_config, name)

		# Handle special methods
		if name == 'get_default_profile':
			return lambda: self._get_default_profile()
		elif name == 'get_default_llm':
			return lambda: self._get_default_llm()
		elif name == 'get_default_agent':
			return lambda: self._get_default_agent()
		elif name == 'load_config':
			return lambda: self._load_config()
		elif name == '_ensure_dirs':
			return lambda: old_config._ensure_dirs()

		raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

	def _get_config_path(self) -> Path:
		"""Get config path from fresh env config."""
		env_config = FlatEnvConfig()
		if env_config.BROWSER_USE_CONFIG_PATH:
			return Path(env_config.BROWSER_USE_CONFIG_PATH).expanduser()
		elif env_config.BROWSER_USE_CONFIG_DIR:
			return Path(env_config.BROWSER_USE_CONFIG_DIR).expanduser() / 'config.json'
		else:
			xdg_config = Path(env_config.XDG_CONFIG_HOME).expanduser()
			return xdg_config / 'browseruse' / 'config.json'

	def _get_db_config(self) -> DBStyleConfigJSON:
		"""Load and migrate config.json."""
		config_path = self._get_config_path()
		return load_and_migrate_config(config_path)

	def _get_default_profile(self) -> dict[str, Any]:
		"""Get the default browser profile configuration."""
		db_config = self._get_db_config()
		for profile in db_config.browser_profile.values():
			if profile.default:
				return profile.model_dump(exclude_none=True)

		# Return first profile if no default
		if db_config.browser_profile:
			return next(iter(db_config.browser_profile.values())).model_dump(exclude_none=True)

		return {}

	def _get_default_llm(self) -> dict[str, Any]:
		"""Get the default LLM configuration."""
		db_config = self._get_db_config()
		for llm in db_config.llm.values():
			if llm.default:
				return llm.model_dump(exclude_none=True)

		# Return first LLM if no default
		if db_config.llm:
			return next(iter(db_config.llm.values())).model_dump(exclude_none=True)

		return {}

	def _get_default_agent(self) -> dict[str, Any]:
		"""Get the default agent configuration."""
		db_config = self._get_db_config()
		for agent in db_config.agent.values():
			if agent.default:
				return agent.model_dump(exclude_none=True)

		# Return first agent if no default
		if db_config.agent:
			return next(iter(db_config.agent.values())).model_dump(exclude_none=True)

		return {}

	def _load_config(self) -> dict[str, Any]:
		"""Load configuration with env var overrides for MCP components."""
		config = {
			'browser_profile': self._get_default_profile(),
			'llm': self._get_default_llm(),
			'agent': self._get_default_agent(),
		}

		# Fresh env config for overrides
		env_config = FlatEnvConfig()

		# Apply MCP-specific env var overrides
		if env_config.BROWSER_USE_HEADLESS is not None:
			config['browser_profile']['headless'] = env_config.BROWSER_USE_HEADLESS

		if env_config.BROWSER_USE_ALLOWED_DOMAINS:
			domains = [d.strip() for d in env_config.BROWSER_USE_ALLOWED_DOMAINS.split(',') if d.strip()]
			config['browser_profile']['allowed_domains'] = domains

		if env_config.OPENAI_API_KEY:
			config['llm']['api_key'] = env_config.OPENAI_API_KEY

		if env_config.BROWSER_USE_LLM_MODEL:
			config['llm']['model'] = env_config.BROWSER_USE_LLM_MODEL

		return config

# From browser_use/config.py
def is_running_in_docker() -> bool:
	"""Detect if we are running in a docker container, for the purpose of optimizing chrome launch flags (dev shm usage, gpu settings, etc.)"""
	try:
		if Path('/.dockerenv').exists() or 'docker' in Path('/proc/1/cgroup').read_text().lower():
			return True
	except Exception:
		pass

	try:
		# if init proc (PID 1) looks like uvicorn/python/uv/etc. then we're in Docker
		# if init proc (PID 1) looks like bash/systemd/init/etc. then we're probably NOT in Docker
		init_cmd = ' '.join(psutil.Process(1).cmdline())
		if ('py' in init_cmd) or ('uv' in init_cmd) or ('app' in init_cmd):
			return True
	except Exception:
		pass

	try:
		# if less than 10 total running procs, then we're almost certainly in a container
		if len(psutil.pids()) < 10:
			return True
	except Exception:
		pass

	return False

# From browser_use/config.py
def create_default_config() -> DBStyleConfigJSON:
	"""Create a fresh default configuration."""
	logger.info('Creating fresh default config.json')

	new_config = DBStyleConfigJSON()

	# Generate default IDs
	profile_id = str(uuid4())
	llm_id = str(uuid4())
	agent_id = str(uuid4())

	# Create default browser profile entry
	new_config.browser_profile[profile_id] = BrowserProfileEntry(id=profile_id, default=True, headless=False, user_data_dir=None)

	# Create default LLM entry
	new_config.llm[llm_id] = LLMEntry(id=llm_id, default=True, model='gpt-4o', api_key='your-openai-api-key-here')

	# Create default agent entry
	new_config.agent[agent_id] = AgentEntry(id=agent_id, default=True)

	return new_config

# From browser_use/config.py
def load_and_migrate_config(config_path: Path) -> DBStyleConfigJSON:
	"""Load config.json or create fresh one if old format detected."""
	if not config_path.exists():
		# Create fresh config with defaults
		config_path.parent.mkdir(parents=True, exist_ok=True)
		new_config = create_default_config()
		with open(config_path, 'w') as f:
			json.dump(new_config.model_dump(), f, indent=2)
		return new_config

	try:
		with open(config_path) as f:
			data = json.load(f)

		# Check if it's already in DB-style format
		if all(key in data for key in ['browser_profile', 'llm', 'agent']) and all(
			isinstance(data.get(key, {}), dict) for key in ['browser_profile', 'llm', 'agent']
		):
			# Check if the values are DB-style entries (have UUIDs as keys)
			if data.get('browser_profile') and all(isinstance(v, dict) and 'id' in v for v in data['browser_profile'].values()):
				# Already in new format
				return DBStyleConfigJSON(**data)

		# Old format detected - delete it and create fresh config
		logger.info(f'Old config format detected at {config_path}, creating fresh config')
		new_config = create_default_config()

		# Overwrite with new config
		with open(config_path, 'w') as f:
			json.dump(new_config.model_dump(), f, indent=2)

		logger.info(f'Created fresh config.json at {config_path}')
		return new_config

	except Exception as e:
		logger.error(f'Failed to load config from {config_path}: {e}, creating fresh config')
		# On any error, create fresh config
		new_config = create_default_config()
		try:
			with open(config_path, 'w') as f:
				json.dump(new_config.model_dump(), f, indent=2)
		except Exception as write_error:
			logger.error(f'Failed to write fresh config: {write_error}')
		return new_config

# From browser_use/config.py
def load_browser_use_config() -> dict[str, Any]:
	"""Load browser-use configuration for MCP components."""
	return CONFIG.load_config()

# From browser_use/config.py
def get_default_profile(config: dict[str, Any]) -> dict[str, Any]:
	"""Get default browser profile from config dict."""
	return config.get('browser_profile', {})

# From browser_use/config.py
def get_default_llm(config: dict[str, Any]) -> dict[str, Any]:
	"""Get default LLM config from config dict."""
	return config.get('llm', {})

# From browser_use/config.py
def BROWSER_USE_LOGGING_LEVEL(self) -> str:
		return os.getenv('BROWSER_USE_LOGGING_LEVEL', 'info').lower()

# From browser_use/config.py
def ANONYMIZED_TELEMETRY(self) -> bool:
		return os.getenv('ANONYMIZED_TELEMETRY', 'true').lower()[:1] in 'ty1'

# From browser_use/config.py
def BROWSER_USE_CLOUD_SYNC(self) -> bool:
		return os.getenv('BROWSER_USE_CLOUD_SYNC', str(self.ANONYMIZED_TELEMETRY)).lower()[:1] in 'ty1'

# From browser_use/config.py
def BROWSER_USE_CLOUD_API_URL(self) -> str:
		url = os.getenv('BROWSER_USE_CLOUD_API_URL', 'https://api.browser-use.com')
		assert '://' in url, 'BROWSER_USE_CLOUD_API_URL must be a valid URL'
		return url

# From browser_use/config.py
def BROWSER_USE_CLOUD_UI_URL(self) -> str:
		url = os.getenv('BROWSER_USE_CLOUD_UI_URL', '')
		# Allow empty string as default, only validate if set
		if url and '://' not in url:
			raise AssertionError('BROWSER_USE_CLOUD_UI_URL must be a valid URL if set')
		return url

# From browser_use/config.py
def XDG_CACHE_HOME(self) -> Path:
		return Path(os.getenv('XDG_CACHE_HOME', '~/.cache')).expanduser().resolve()

# From browser_use/config.py
def XDG_CONFIG_HOME(self) -> Path:
		return Path(os.getenv('XDG_CONFIG_HOME', '~/.config')).expanduser().resolve()

# From browser_use/config.py
def BROWSER_USE_CONFIG_DIR(self) -> Path:
		path = Path(os.getenv('BROWSER_USE_CONFIG_DIR', str(self.XDG_CONFIG_HOME / 'browseruse'))).expanduser().resolve()
		self._ensure_dirs()
		return path

# From browser_use/config.py
def BROWSER_USE_CONFIG_FILE(self) -> Path:
		return self.BROWSER_USE_CONFIG_DIR / 'config.json'

# From browser_use/config.py
def BROWSER_USE_PROFILES_DIR(self) -> Path:
		path = self.BROWSER_USE_CONFIG_DIR / 'profiles'
		self._ensure_dirs()
		return path

# From browser_use/config.py
def BROWSER_USE_DEFAULT_USER_DATA_DIR(self) -> Path:
		return self.BROWSER_USE_PROFILES_DIR / 'default'

# From browser_use/config.py
def BROWSER_USE_EXTENSIONS_DIR(self) -> Path:
		path = self.BROWSER_USE_CONFIG_DIR / 'extensions'
		self._ensure_dirs()
		return path

# From browser_use/config.py
def OPENAI_API_KEY(self) -> str:
		return os.getenv('OPENAI_API_KEY', '')

# From browser_use/config.py
def ANTHROPIC_API_KEY(self) -> str:
		return os.getenv('ANTHROPIC_API_KEY', '')

# From browser_use/config.py
def GOOGLE_API_KEY(self) -> str:
		return os.getenv('GOOGLE_API_KEY', '')

# From browser_use/config.py
def DEEPSEEK_API_KEY(self) -> str:
		return os.getenv('DEEPSEEK_API_KEY', '')

# From browser_use/config.py
def GROK_API_KEY(self) -> str:
		return os.getenv('GROK_API_KEY', '')

# From browser_use/config.py
def NOVITA_API_KEY(self) -> str:
		return os.getenv('NOVITA_API_KEY', '')

# From browser_use/config.py
def AZURE_OPENAI_ENDPOINT(self) -> str:
		return os.getenv('AZURE_OPENAI_ENDPOINT', '')

# From browser_use/config.py
def AZURE_OPENAI_KEY(self) -> str:
		return os.getenv('AZURE_OPENAI_KEY', '')

# From browser_use/config.py
def SKIP_LLM_API_KEY_VERIFICATION(self) -> bool:
		return os.getenv('SKIP_LLM_API_KEY_VERIFICATION', 'false').lower()[:1] in 'ty1'

# From browser_use/config.py
def IN_DOCKER(self) -> bool:
		return os.getenv('IN_DOCKER', 'false').lower()[:1] in 'ty1' or is_running_in_docker()

# From browser_use/config.py
def IS_IN_EVALS(self) -> bool:
		return os.getenv('IS_IN_EVALS', 'false').lower()[:1] in 'ty1'

# From browser_use/config.py
def WIN_FONT_DIR(self) -> str:
		return os.getenv('WIN_FONT_DIR', 'C:\\Windows\\Fonts')


import shutil
from abc import ABC
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from markdown_pdf import MarkdownPdf
from markdown_pdf import Section
import anyio
import pypdf

# From filesystem/file_system.py
class FileSystemError(Exception):
	"""Custom exception for file system operations that should be shown to LLM"""

	pass

# From filesystem/file_system.py
class BaseFile(BaseModel, ABC):
	"""Base class for all file types"""

	name: str
	content: str = ''

	# --- Subclass must define this ---
	@property
	@abstractmethod
	def extension(self) -> str:
		"""File extension (e.g. 'txt', 'md')"""
		pass

	def write_file_content(self, content: str) -> None:
		"""Update internal content (formatted)"""
		self.update_content(content)

	def append_file_content(self, content: str) -> None:
		"""Append content to internal content"""
		self.update_content(self.content + content)

	# --- These are shared and implemented here ---

	def update_content(self, content: str) -> None:
		self.content = content

	def sync_to_disk_sync(self, path: Path) -> None:
		file_path = path / self.full_name
		file_path.write_text(self.content)

	async def sync_to_disk(self, path: Path) -> None:
		file_path = path / self.full_name
		with ThreadPoolExecutor() as executor:
			await asyncio.get_event_loop().run_in_executor(executor, lambda: file_path.write_text(self.content))

	async def write(self, content: str, path: Path) -> None:
		self.write_file_content(content)
		await self.sync_to_disk(path)

	async def append(self, content: str, path: Path) -> None:
		self.append_file_content(content)
		await self.sync_to_disk(path)

	def read(self) -> str:
		return self.content

	@property
	def full_name(self) -> str:
		return f'{self.name}.{self.extension}'

	@property
	def get_size(self) -> int:
		return len(self.content)

	@property
	def get_line_count(self) -> int:
		return len(self.content.splitlines())

# From filesystem/file_system.py
class MarkdownFile(BaseFile):
	"""Markdown file implementation"""

	@property
	def extension(self) -> str:
		return 'md'

# From filesystem/file_system.py
class TxtFile(BaseFile):
	"""Plain text file implementation"""

	@property
	def extension(self) -> str:
		return 'txt'

# From filesystem/file_system.py
class JsonFile(BaseFile):
	"""JSON file implementation"""

	@property
	def extension(self) -> str:
		return 'json'

# From filesystem/file_system.py
class CsvFile(BaseFile):
	"""CSV file implementation"""

	@property
	def extension(self) -> str:
		return 'csv'

# From filesystem/file_system.py
class PdfFile(BaseFile):
	"""PDF file implementation"""

	@property
	def extension(self) -> str:
		return 'pdf'

	def sync_to_disk_sync(self, path: Path) -> None:
		file_path = path / self.full_name
		try:
			md_pdf = MarkdownPdf()
			md_pdf.add_section(Section(self.content))
			md_pdf.save(file_path)
		except Exception as e:
			raise FileSystemError(f"Error: Could not write to file '{self.full_name}'. {str(e)}")

	async def sync_to_disk(self, path: Path) -> None:
		with ThreadPoolExecutor() as executor:
			await asyncio.get_event_loop().run_in_executor(executor, lambda: self.sync_to_disk_sync(path))

# From filesystem/file_system.py
class FileSystemState(BaseModel):
	"""Serializable state of the file system"""

	files: dict[str, dict[str, Any]] = Field(default_factory=dict)  # full filename -> file data
	base_dir: str
	extracted_content_count: int = 0

# From filesystem/file_system.py
class FileSystem:
	"""Enhanced file system with in-memory storage and multiple file type support"""

	def __init__(self, base_dir: str | Path, create_default_files: bool = True):
		# Handle the Path conversion before calling super().__init__
		self.base_dir = Path(base_dir) if isinstance(base_dir, str) else base_dir
		self.base_dir.mkdir(parents=True, exist_ok=True)

		# Create and use a dedicated subfolder for all operations
		self.data_dir = self.base_dir / DEFAULT_FILE_SYSTEM_PATH
		if self.data_dir.exists():
			# clean the data directory
			shutil.rmtree(self.data_dir)
		self.data_dir.mkdir(exist_ok=True)

		self._file_types: dict[str, type[BaseFile]] = {
			'md': MarkdownFile,
			'txt': TxtFile,
			'json': JsonFile,
			'csv': CsvFile,
			'pdf': PdfFile,
		}

		self.files = {}
		if create_default_files:
			self.default_files = ['todo.md']
			self._create_default_files()

		self.extracted_content_count = 0

	def get_allowed_extensions(self) -> list[str]:
		"""Get allowed extensions"""
		return list(self._file_types.keys())

	def _get_file_type_class(self, extension: str) -> type[BaseFile] | None:
		"""Get the appropriate file class for an extension."""
		return self._file_types.get(extension.lower(), None)

	def _create_default_files(self) -> None:
		"""Create default results and todo files"""
		for full_filename in self.default_files:
			name_without_ext, extension = self._parse_filename(full_filename)
			file_class = self._get_file_type_class(extension)
			if not file_class:
				raise ValueError(f"Error: Invalid file extension '{extension}' for file '{full_filename}'.")

			file_obj = file_class(name=name_without_ext)
			self.files[full_filename] = file_obj  # Use full filename as key
			file_obj.sync_to_disk_sync(self.data_dir)

	def _is_valid_filename(self, file_name: str) -> bool:
		"""Check if filename matches the required pattern: name.extension"""
		# Build extensions pattern from _file_types
		extensions = '|'.join(self._file_types.keys())
		pattern = rf'^[a-zA-Z0-9_\-]+\.({extensions})$'
		return bool(re.match(pattern, file_name))

	def _parse_filename(self, filename: str) -> tuple[str, str]:
		"""Parse filename into name and extension. Always check _is_valid_filename first."""
		name, extension = filename.rsplit('.', 1)
		return name, extension.lower()

	def get_dir(self) -> Path:
		"""Get the file system directory"""
		return self.data_dir

	def get_file(self, full_filename: str) -> BaseFile | None:
		"""Get a file object by full filename"""
		if not self._is_valid_filename(full_filename):
			return None

		# Use full filename as key
		return self.files.get(full_filename)

	def list_files(self) -> list[str]:
		"""List all files in the system"""
		return [file_obj.full_name for file_obj in self.files.values()]

	def display_file(self, full_filename: str) -> str | None:
		"""Display file content using file-specific display method"""
		if not self._is_valid_filename(full_filename):
			return None

		file_obj = self.get_file(full_filename)
		if not file_obj:
			return None

		return file_obj.read()

	async def read_file(self, full_filename: str, external_file: bool = False) -> str:
		"""Read file content using file-specific read method and return appropriate message to LLM"""
		if external_file:
			try:
				try:
					_, extension = self._parse_filename(full_filename)
				except Exception:
					return f'Error: Invalid filename format {full_filename}. Must be alphanumeric with a supported extension.'
				if extension in ['md', 'txt', 'json', 'csv']:
					import anyio

					async with await anyio.open_file(full_filename, 'r') as f:
						content = await f.read()
						return f'Read from file {full_filename}.\n<content>\n{content}\n</content>'
				elif extension == 'pdf':
					import pypdf

					reader = pypdf.PdfReader(full_filename)
					num_pages = len(reader.pages)
					MAX_PDF_PAGES = 10
					extra_pages = num_pages - MAX_PDF_PAGES
					extracted_text = ''
					for page in reader.pages[:MAX_PDF_PAGES]:
						extracted_text += page.extract_text()
					extra_pages_text = f'{extra_pages} more pages...' if extra_pages > 0 else ''
					return f'Read from file {full_filename}.\n<content>\n{extracted_text}\n{extra_pages_text}</content>'
				else:
					return f'Error: Cannot read file {full_filename} as {extension} extension is not supported.'
			except FileNotFoundError:
				return f"Error: File '{full_filename}' not found."
			except PermissionError:
				return f"Error: Permission denied to read file '{full_filename}'."
			except Exception as e:
				return f"Error: Could not read file '{full_filename}'."

		if not self._is_valid_filename(full_filename):
			return INVALID_FILENAME_ERROR_MESSAGE

		file_obj = self.get_file(full_filename)
		if not file_obj:
			return f"File '{full_filename}' not found."

		try:
			content = file_obj.read()
			return f'Read from file {full_filename}.\n<content>\n{content}\n</content>'
		except FileSystemError as e:
			return str(e)
		except Exception:
			return f"Error: Could not read file '{full_filename}'."

	async def write_file(self, full_filename: str, content: str) -> str:
		"""Write content to file using file-specific write method"""
		if not self._is_valid_filename(full_filename):
			return INVALID_FILENAME_ERROR_MESSAGE

		try:
			name_without_ext, extension = self._parse_filename(full_filename)
			file_class = self._get_file_type_class(extension)
			if not file_class:
				raise ValueError(f"Error: Invalid file extension '{extension}' for file '{full_filename}'.")

			# Create or get existing file using full filename as key
			if full_filename in self.files:
				file_obj = self.files[full_filename]
			else:
				file_obj = file_class(name=name_without_ext)
				self.files[full_filename] = file_obj  # Use full filename as key

			# Use file-specific write method
			await file_obj.write(content, self.data_dir)
			return f'Data written to file {full_filename} successfully.'
		except FileSystemError as e:
			return str(e)
		except Exception as e:
			return f"Error: Could not write to file '{full_filename}'. {str(e)}"

	async def append_file(self, full_filename: str, content: str) -> str:
		"""Append content to file using file-specific append method"""
		if not self._is_valid_filename(full_filename):
			return INVALID_FILENAME_ERROR_MESSAGE

		file_obj = self.get_file(full_filename)
		if not file_obj:
			return f"File '{full_filename}' not found."

		try:
			await file_obj.append(content, self.data_dir)
			return f'Data appended to file {full_filename} successfully.'
		except FileSystemError as e:
			return str(e)
		except Exception as e:
			return f"Error: Could not append to file '{full_filename}'. {str(e)}"

	async def replace_file_str(self, full_filename: str, old_str: str, new_str: str) -> str:
		"""Replace old_str with new_str in file_name"""
		if not self._is_valid_filename(full_filename):
			return INVALID_FILENAME_ERROR_MESSAGE

		if not old_str:
			return 'Error: Cannot replace empty string. Please provide a non-empty string to replace.'

		file_obj = self.get_file(full_filename)
		if not file_obj:
			return f"File '{full_filename}' not found."

		try:
			content = file_obj.read()
			content = content.replace(old_str, new_str)
			await file_obj.write(content, self.data_dir)
			return f'Successfully replaced all occurrences of "{old_str}" with "{new_str}" in file {full_filename}'
		except FileSystemError as e:
			return str(e)
		except Exception as e:
			return f"Error: Could not replace string in file '{full_filename}'. {str(e)}"

	async def save_extracted_content(self, content: str) -> str:
		"""Save extracted content to a numbered file"""
		initial_filename = f'extracted_content_{self.extracted_content_count}'
		extracted_filename = f'{initial_filename}.md'
		file_obj = MarkdownFile(name=initial_filename)
		await file_obj.write(content, self.data_dir)
		self.files[extracted_filename] = file_obj
		self.extracted_content_count += 1
		return f'Extracted content saved to file {extracted_filename} successfully.'

	def describe(self) -> str:
		"""List all files with their content information using file-specific display methods"""
		DISPLAY_CHARS = 400
		description = ''

		for file_obj in self.files.values():
			# Skip todo.md from description
			if file_obj.full_name == 'todo.md':
				continue

			content = file_obj.read()

			# Handle empty files
			if not content:
				description += f'<file>\n{file_obj.full_name} - [empty file]\n</file>\n'
				continue

			lines = content.splitlines()
			line_count = len(lines)

			# For small files, display the entire content
			whole_file_description = (
				f'<file>\n{file_obj.full_name} - {line_count} lines\n<content>\n{content}\n</content>\n</file>\n'
			)
			if len(content) < int(1.5 * DISPLAY_CHARS):
				description += whole_file_description
				continue

			# For larger files, display start and end previews
			half_display_chars = DISPLAY_CHARS // 2

			# Get start preview
			start_preview = ''
			start_line_count = 0
			chars_count = 0
			for line in lines:
				if chars_count + len(line) + 1 > half_display_chars:
					break
				start_preview += line + '\n'
				chars_count += len(line) + 1
				start_line_count += 1

			# Get end preview
			end_preview = ''
			end_line_count = 0
			chars_count = 0
			for line in reversed(lines):
				if chars_count + len(line) + 1 > half_display_chars:
					break
				end_preview = line + '\n' + end_preview
				chars_count += len(line) + 1
				end_line_count += 1

			# Calculate lines in between
			middle_line_count = line_count - start_line_count - end_line_count
			if middle_line_count <= 0:
				description += whole_file_description
				continue

			start_preview = start_preview.strip('\n').rstrip()
			end_preview = end_preview.strip('\n').rstrip()

			# Format output
			if not (start_preview or end_preview):
				description += f'<file>\n{file_obj.full_name} - {line_count} lines\n<content>\n{middle_line_count} lines...\n</content>\n</file>\n'
			else:
				description += f'<file>\n{file_obj.full_name} - {line_count} lines\n<content>\n{start_preview}\n'
				description += f'... {middle_line_count} more lines ...\n'
				description += f'{end_preview}\n'
				description += '</content>\n</file>\n'

		return description.strip('\n')

	def get_todo_contents(self) -> str:
		"""Get todo file contents"""
		todo_file = self.get_file('todo.md')
		return todo_file.read() if todo_file else ''

	def get_state(self) -> FileSystemState:
		"""Get serializable state of the file system"""
		files_data = {}
		for full_filename, file_obj in self.files.items():
			files_data[full_filename] = {'type': file_obj.__class__.__name__, 'data': file_obj.model_dump()}

		return FileSystemState(
			files=files_data, base_dir=str(self.base_dir), extracted_content_count=self.extracted_content_count
		)

	def nuke(self) -> None:
		"""Delete the file system directory"""
		shutil.rmtree(self.data_dir)

	@classmethod
	def from_state(cls, state: FileSystemState) -> 'FileSystem':
		"""Restore file system from serializable state at the exact same location"""
		# Create file system without default files
		fs = cls(base_dir=Path(state.base_dir), create_default_files=False)
		fs.extracted_content_count = state.extracted_content_count

		# Restore all files
		for full_filename, file_data in state.files.items():
			file_type = file_data['type']
			file_info = file_data['data']

			# Create the appropriate file object based on type
			if file_type == 'MarkdownFile':
				file_obj = MarkdownFile(**file_info)
			elif file_type == 'TxtFile':
				file_obj = TxtFile(**file_info)
			elif file_type == 'JsonFile':
				file_obj = JsonFile(**file_info)
			elif file_type == 'CsvFile':
				file_obj = CsvFile(**file_info)
			elif file_type == 'PdfFile':
				file_obj = PdfFile(**file_info)
			else:
				# Skip unknown file types
				continue

			# Add to files dict and sync to disk
			fs.files[full_filename] = file_obj
			file_obj.sync_to_disk_sync(fs.data_dir)

		return fs

# From filesystem/file_system.py
def extension(self) -> str:
		"""File extension (e.g. 'txt', 'md')"""
		pass

# From filesystem/file_system.py
def write_file_content(self, content: str) -> None:
		"""Update internal content (formatted)"""
		self.update_content(content)

# From filesystem/file_system.py
def append_file_content(self, content: str) -> None:
		"""Append content to internal content"""
		self.update_content(self.content + content)

# From filesystem/file_system.py
def update_content(self, content: str) -> None:
		self.content = content

# From filesystem/file_system.py
def sync_to_disk_sync(self, path: Path) -> None:
		file_path = path / self.full_name
		file_path.write_text(self.content)

# From filesystem/file_system.py
def read(self) -> str:
		return self.content

# From filesystem/file_system.py
def full_name(self) -> str:
		return f'{self.name}.{self.extension}'

# From filesystem/file_system.py
def get_size(self) -> int:
		return len(self.content)

# From filesystem/file_system.py
def get_line_count(self) -> int:
		return len(self.content.splitlines())

# From filesystem/file_system.py
def get_allowed_extensions(self) -> list[str]:
		"""Get allowed extensions"""
		return list(self._file_types.keys())

# From filesystem/file_system.py
def get_dir(self) -> Path:
		"""Get the file system directory"""
		return self.data_dir

# From filesystem/file_system.py
def get_file(self, full_filename: str) -> BaseFile | None:
		"""Get a file object by full filename"""
		if not self._is_valid_filename(full_filename):
			return None

		# Use full filename as key
		return self.files.get(full_filename)

# From filesystem/file_system.py
def list_files(self) -> list[str]:
		"""List all files in the system"""
		return [file_obj.full_name for file_obj in self.files.values()]

# From filesystem/file_system.py
def display_file(self, full_filename: str) -> str | None:
		"""Display file content using file-specific display method"""
		if not self._is_valid_filename(full_filename):
			return None

		file_obj = self.get_file(full_filename)
		if not file_obj:
			return None

		return file_obj.read()

# From filesystem/file_system.py
def describe(self) -> str:
		"""List all files with their content information using file-specific display methods"""
		DISPLAY_CHARS = 400
		description = ''

		for file_obj in self.files.values():
			# Skip todo.md from description
			if file_obj.full_name == 'todo.md':
				continue

			content = file_obj.read()

			# Handle empty files
			if not content:
				description += f'<file>\n{file_obj.full_name} - [empty file]\n</file>\n'
				continue

			lines = content.splitlines()
			line_count = len(lines)

			# For small files, display the entire content
			whole_file_description = (
				f'<file>\n{file_obj.full_name} - {line_count} lines\n<content>\n{content}\n</content>\n</file>\n'
			)
			if len(content) < int(1.5 * DISPLAY_CHARS):
				description += whole_file_description
				continue

			# For larger files, display start and end previews
			half_display_chars = DISPLAY_CHARS // 2

			# Get start preview
			start_preview = ''
			start_line_count = 0
			chars_count = 0
			for line in lines:
				if chars_count + len(line) + 1 > half_display_chars:
					break
				start_preview += line + '\n'
				chars_count += len(line) + 1
				start_line_count += 1

			# Get end preview
			end_preview = ''
			end_line_count = 0
			chars_count = 0
			for line in reversed(lines):
				if chars_count + len(line) + 1 > half_display_chars:
					break
				end_preview = line + '\n' + end_preview
				chars_count += len(line) + 1
				end_line_count += 1

			# Calculate lines in between
			middle_line_count = line_count - start_line_count - end_line_count
			if middle_line_count <= 0:
				description += whole_file_description
				continue

			start_preview = start_preview.strip('\n').rstrip()
			end_preview = end_preview.strip('\n').rstrip()

			# Format output
			if not (start_preview or end_preview):
				description += f'<file>\n{file_obj.full_name} - {line_count} lines\n<content>\n{middle_line_count} lines...\n</content>\n</file>\n'
			else:
				description += f'<file>\n{file_obj.full_name} - {line_count} lines\n<content>\n{start_preview}\n'
				description += f'... {middle_line_count} more lines ...\n'
				description += f'{end_preview}\n'
				description += '</content>\n</file>\n'

		return description.strip('\n')

# From filesystem/file_system.py
def get_todo_contents(self) -> str:
		"""Get todo file contents"""
		todo_file = self.get_file('todo.md')
		return todo_file.read() if todo_file else ''

# From filesystem/file_system.py
def get_state(self) -> FileSystemState:
		"""Get serializable state of the file system"""
		files_data = {}
		for full_filename, file_obj in self.files.items():
			files_data[full_filename] = {'type': file_obj.__class__.__name__, 'data': file_obj.model_dump()}

		return FileSystemState(
			files=files_data, base_dir=str(self.base_dir), extracted_content_count=self.extracted_content_count
		)

# From filesystem/file_system.py
def nuke(self) -> None:
		"""Delete the file system directory"""
		shutil.rmtree(self.data_dir)

# From filesystem/file_system.py
def from_state(cls, state: FileSystemState) -> 'FileSystem':
		"""Restore file system from serializable state at the exact same location"""
		# Create file system without default files
		fs = cls(base_dir=Path(state.base_dir), create_default_files=False)
		fs.extracted_content_count = state.extracted_content_count

		# Restore all files
		for full_filename, file_data in state.files.items():
			file_type = file_data['type']
			file_info = file_data['data']

			# Create the appropriate file object based on type
			if file_type == 'MarkdownFile':
				file_obj = MarkdownFile(**file_info)
			elif file_type == 'TxtFile':
				file_obj = TxtFile(**file_info)
			elif file_type == 'JsonFile':
				file_obj = JsonFile(**file_info)
			elif file_type == 'CsvFile':
				file_obj = CsvFile(**file_info)
			elif file_type == 'PdfFile':
				file_obj = PdfFile(**file_info)
			else:
				# Skip unknown file types
				continue

			# Add to files dict and sync to disk
			fs.files[full_filename] = file_obj
			file_obj.sync_to_disk_sync(fs.data_dir)

		return fs

from datetime import timedelta
import aiofiles
import httpx
from browser_use.llm.base import BaseChatModel
from browser_use.llm.views import ChatInvokeUsage
from browser_use.tokens.views import CachedPricingData
from browser_use.tokens.views import ModelPricing
from browser_use.tokens.views import ModelUsageStats
from browser_use.tokens.views import ModelUsageTokens
from browser_use.tokens.views import TokenCostCalculated
from browser_use.tokens.views import TokenUsageEntry
from browser_use.tokens.views import UsageSummary

# From tokens/service.py
class TokenCost:
	"""Service for tracking token usage and calculating costs"""

	CACHE_DIR_NAME = 'browser_use/token_cost'
	CACHE_DURATION = timedelta(days=1)
	PRICING_URL = 'https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json'

	def __init__(self, include_cost: bool = False):
		self.include_cost = include_cost or os.getenv('BROWSER_USE_CALCULATE_COST', 'false').lower() == 'true'

		self.usage_history: list[TokenUsageEntry] = []
		self.registered_llms: dict[str, BaseChatModel] = {}
		self._pricing_data: dict[str, Any] | None = None
		self._initialized = False
		self._cache_dir = xdg_cache_home() / self.CACHE_DIR_NAME

	async def initialize(self) -> None:
		"""Initialize the service by loading pricing data"""
		if not self._initialized:
			if self.include_cost:
				await self._load_pricing_data()
			self._initialized = True

	async def _load_pricing_data(self) -> None:
		"""Load pricing data from cache or fetch from GitHub"""
		# Try to find a valid cache file
		cache_file = await self._find_valid_cache()

		if cache_file:
			await self._load_from_cache(cache_file)
		else:
			await self._fetch_and_cache_pricing_data()

	async def _find_valid_cache(self) -> Path | None:
		"""Find the most recent valid cache file"""
		try:
			# Ensure cache directory exists
			self._cache_dir.mkdir(parents=True, exist_ok=True)

			# List all JSON files in the cache directory
			cache_files = list(self._cache_dir.glob('*.json'))

			if not cache_files:
				return None

			# Sort by modification time (most recent first)
			cache_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

			# Check each file until we find a valid one
			for cache_file in cache_files:
				if await self._is_cache_valid(cache_file):
					return cache_file
				else:
					# Clean up old cache files
					try:
						os.remove(cache_file)
					except Exception:
						pass

			return None
		except Exception:
			return None

	async def _is_cache_valid(self, cache_file: Path) -> bool:
		"""Check if a specific cache file is valid and not expired"""
		try:
			if not cache_file.exists():
				return False

			# Read the cached data
			async with aiofiles.open(cache_file, 'r') as f:
				content = await f.read()
				cached = CachedPricingData.model_validate_json(content)

			# Check if cache is still valid
			return datetime.now() - cached.timestamp < self.CACHE_DURATION
		except Exception:
			return False

	async def _load_from_cache(self, cache_file: Path) -> None:
		"""Load pricing data from a specific cache file"""
		try:
			async with aiofiles.open(cache_file, 'r') as f:
				content = await f.read()
				cached = CachedPricingData.model_validate_json(content)
				self._pricing_data = cached.data
		except Exception as e:
			print(f'Error loading cached pricing data from {cache_file}: {e}')
			# Fall back to fetching
			await self._fetch_and_cache_pricing_data()

	async def _fetch_and_cache_pricing_data(self) -> None:
		"""Fetch pricing data from LiteLLM GitHub and cache it with timestamp"""
		try:
			async with httpx.AsyncClient() as client:
				response = await client.get(self.PRICING_URL, timeout=30)
				response.raise_for_status()

				self._pricing_data = response.json()

			# Create cache object with timestamp
			cached = CachedPricingData(timestamp=datetime.now(), data=self._pricing_data or {})

			# Ensure cache directory exists
			self._cache_dir.mkdir(parents=True, exist_ok=True)

			# Create cache file with timestamp in filename
			timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
			cache_file = self._cache_dir / f'pricing_{timestamp_str}.json'

			async with aiofiles.open(cache_file, 'w') as f:
				await f.write(cached.model_dump_json(indent=2))

		except Exception as e:
			print(f'Error fetching pricing data: {e}')
			# Fall back to empty pricing data
			self._pricing_data = {}

	async def get_model_pricing(self, model_name: str) -> ModelPricing | None:
		"""Get pricing information for a specific model"""
		# Ensure we're initialized
		if not self._initialized:
			await self.initialize()

		if not self._pricing_data or model_name not in self._pricing_data:
			return None

		data = self._pricing_data[model_name]
		return ModelPricing(
			model=model_name,
			input_cost_per_token=data.get('input_cost_per_token'),
			output_cost_per_token=data.get('output_cost_per_token'),
			max_tokens=data.get('max_tokens'),
			max_input_tokens=data.get('max_input_tokens'),
			max_output_tokens=data.get('max_output_tokens'),
			cache_read_input_token_cost=data.get('cache_read_input_token_cost'),
			cache_creation_input_token_cost=data.get('cache_creation_input_token_cost'),
		)

	async def calculate_cost(self, model: str, usage: ChatInvokeUsage) -> TokenCostCalculated | None:
		if not self.include_cost:
			return None

		data = await self.get_model_pricing(model)
		if data is None:
			return None

		uncached_prompt_tokens = usage.prompt_tokens - (usage.prompt_cached_tokens or 0)

		return TokenCostCalculated(
			new_prompt_tokens=usage.prompt_tokens,
			new_prompt_cost=uncached_prompt_tokens * (data.input_cost_per_token or 0),
			# Cached tokens
			prompt_read_cached_tokens=usage.prompt_cached_tokens,
			prompt_read_cached_cost=usage.prompt_cached_tokens * data.cache_read_input_token_cost
			if usage.prompt_cached_tokens and data.cache_read_input_token_cost
			else None,
			# Cache creation tokens
			prompt_cached_creation_tokens=usage.prompt_cache_creation_tokens,
			prompt_cache_creation_cost=usage.prompt_cache_creation_tokens * data.cache_creation_input_token_cost
			if data.cache_creation_input_token_cost and usage.prompt_cache_creation_tokens
			else None,
			# Completion tokens
			completion_tokens=usage.completion_tokens,
			completion_cost=usage.completion_tokens * float(data.output_cost_per_token or 0),
		)

	def add_usage(self, model: str, usage: ChatInvokeUsage) -> TokenUsageEntry:
		"""Add token usage entry to history (without calculating cost)"""
		entry = TokenUsageEntry(
			model=model,
			timestamp=datetime.now(),
			usage=usage,
		)

		self.usage_history.append(entry)

		return entry

	# async def _log_non_usage_llm(self, llm: BaseChatModel) -> None:
	# 	"""Log non-usage to the logger"""
	# 	C_CYAN = '\033[96m'
	# 	C_RESET = '\033[0m'

	# 	cost_logger.info(f'🧠 llm : {C_CYAN}{llm.model}{C_RESET} (no usage found)')

	async def _log_usage(self, model: str, usage: TokenUsageEntry) -> None:
		"""Log usage to the logger"""
		if not self._initialized:
			await self.initialize()

		# ANSI color codes
		C_CYAN = '\033[96m'
		C_YELLOW = '\033[93m'
		C_GREEN = '\033[92m'
		C_BLUE = '\033[94m'
		C_RESET = '\033[0m'

		# Always get cost breakdown for token details (even if not showing costs)
		cost = await self.calculate_cost(model, usage.usage)

		# Build input tokens breakdown
		input_part = self._build_input_tokens_display(usage.usage, cost)

		# Build output tokens display
		completion_tokens_fmt = self._format_tokens(usage.usage.completion_tokens)
		if self.include_cost and cost and cost.completion_cost > 0:
			output_part = f'📤 {C_GREEN}{completion_tokens_fmt} (${cost.completion_cost:.4f}){C_RESET}'
		else:
			output_part = f'📤 {C_GREEN}{completion_tokens_fmt}{C_RESET}'

		cost_logger.info(f'🧠 {C_CYAN}{model}{C_RESET} | {input_part} | {output_part}')

	def _build_input_tokens_display(self, usage: ChatInvokeUsage, cost: TokenCostCalculated | None) -> str:
		"""Build a clear display of input tokens breakdown with emojis and optional costs"""
		C_YELLOW = '\033[93m'
		C_BLUE = '\033[94m'
		C_RESET = '\033[0m'

		parts = []

		# Always show token breakdown if we have cache information, regardless of cost tracking
		if usage.prompt_cached_tokens or usage.prompt_cache_creation_tokens:
			# Calculate actual new tokens (non-cached)
			new_tokens = usage.prompt_tokens - (usage.prompt_cached_tokens or 0)

			if new_tokens > 0:
				new_tokens_fmt = self._format_tokens(new_tokens)
				if self.include_cost and cost and cost.new_prompt_cost > 0:
					parts.append(f'🆕 {C_YELLOW}{new_tokens_fmt} (${cost.new_prompt_cost:.4f}){C_RESET}')
				else:
					parts.append(f'🆕 {C_YELLOW}{new_tokens_fmt}{C_RESET}')

			if usage.prompt_cached_tokens:
				cached_tokens_fmt = self._format_tokens(usage.prompt_cached_tokens)
				if self.include_cost and cost and cost.prompt_read_cached_cost:
					parts.append(f'💾 {C_BLUE}{cached_tokens_fmt} (${cost.prompt_read_cached_cost:.4f}){C_RESET}')
				else:
					parts.append(f'💾 {C_BLUE}{cached_tokens_fmt}{C_RESET}')

			if usage.prompt_cache_creation_tokens:
				creation_tokens_fmt = self._format_tokens(usage.prompt_cache_creation_tokens)
				if self.include_cost and cost and cost.prompt_cache_creation_cost:
					parts.append(f'🔧 {C_BLUE}{creation_tokens_fmt} (${cost.prompt_cache_creation_cost:.4f}){C_RESET}')
				else:
					parts.append(f'🔧 {C_BLUE}{creation_tokens_fmt}{C_RESET}')

		if not parts:
			# Fallback to simple display when no cache information available
			total_tokens_fmt = self._format_tokens(usage.prompt_tokens)
			if self.include_cost and cost and cost.new_prompt_cost > 0:
				parts.append(f'📥 {C_YELLOW}{total_tokens_fmt} (${cost.new_prompt_cost:.4f}){C_RESET}')
			else:
				parts.append(f'📥 {C_YELLOW}{total_tokens_fmt}{C_RESET}')

		return ' + '.join(parts)

	def register_llm(self, llm: BaseChatModel) -> BaseChatModel:
		"""
		Register an LLM to automatically track its token usage

		@dev Guarantees that the same instance is not registered multiple times
		"""
		# Use instance ID as key to avoid collisions between multiple instances
		instance_id = str(id(llm))

		# Check if this exact instance is already registered
		if instance_id in self.registered_llms:
			logger.debug(f'LLM instance {instance_id} ({llm.provider}_{llm.model}) is already registered')
			return llm

		self.registered_llms[instance_id] = llm

		# Store the original method
		original_ainvoke = llm.ainvoke
		# Store reference to self for use in the closure
		token_cost_service = self

		# Create a wrapped version that tracks usage
		async def tracked_ainvoke(messages, output_format=None):
			# Call the original method
			result = await original_ainvoke(messages, output_format)

			# Track usage if available (no await needed since add_usage is now sync)
			if result.usage:
				usage = token_cost_service.add_usage(llm.model, result.usage)

				logger.debug(f'Token cost service: {usage}')

				asyncio.create_task(token_cost_service._log_usage(llm.model, usage))

			# else:
			# 	await token_cost_service._log_non_usage_llm(llm)

			return result

		# Replace the method with our tracked version
		# Using setattr to avoid type checking issues with overloaded methods
		setattr(llm, 'ainvoke', tracked_ainvoke)

		return llm

	def get_usage_tokens_for_model(self, model: str) -> ModelUsageTokens:
		"""Get usage tokens for a specific model"""
		filtered_usage = [u for u in self.usage_history if u.model == model]

		return ModelUsageTokens(
			model=model,
			prompt_tokens=sum(u.usage.prompt_tokens for u in filtered_usage),
			prompt_cached_tokens=sum(u.usage.prompt_cached_tokens or 0 for u in filtered_usage),
			completion_tokens=sum(u.usage.completion_tokens for u in filtered_usage),
			total_tokens=sum(u.usage.prompt_tokens + u.usage.completion_tokens for u in filtered_usage),
		)

	async def get_usage_summary(self, model: str | None = None, since: datetime | None = None) -> UsageSummary:
		"""Get summary of token usage and costs (costs calculated on-the-fly)"""
		filtered_usage = self.usage_history

		if model:
			filtered_usage = [u for u in filtered_usage if u.model == model]

		if since:
			filtered_usage = [u for u in filtered_usage if u.timestamp >= since]

		if not filtered_usage:
			return UsageSummary(
				total_prompt_tokens=0,
				total_prompt_cost=0.0,
				total_prompt_cached_tokens=0,
				total_prompt_cached_cost=0.0,
				total_completion_tokens=0,
				total_completion_cost=0.0,
				total_tokens=0,
				total_cost=0.0,
				entry_count=0,
			)

		# Calculate totals
		total_prompt = sum(u.usage.prompt_tokens for u in filtered_usage)
		total_completion = sum(u.usage.completion_tokens for u in filtered_usage)
		total_tokens = total_prompt + total_completion
		total_prompt_cached = sum(u.usage.prompt_cached_tokens or 0 for u in filtered_usage)
		models = list({u.model for u in filtered_usage})

		# Calculate per-model stats with record-by-record cost calculation
		model_stats: dict[str, ModelUsageStats] = {}
		total_prompt_cost = 0.0
		total_completion_cost = 0.0
		total_prompt_cached_cost = 0.0

		for entry in filtered_usage:
			if entry.model not in model_stats:
				model_stats[entry.model] = ModelUsageStats(model=entry.model)

			stats = model_stats[entry.model]
			stats.prompt_tokens += entry.usage.prompt_tokens
			stats.completion_tokens += entry.usage.completion_tokens
			stats.total_tokens += entry.usage.prompt_tokens + entry.usage.completion_tokens
			stats.invocations += 1

			if self.include_cost:
				# Calculate cost record by record using the updated calculate_cost function
				cost = await self.calculate_cost(entry.model, entry.usage)
				if cost:
					stats.cost += cost.total_cost
					total_prompt_cost += cost.prompt_cost
					total_completion_cost += cost.completion_cost
					total_prompt_cached_cost += cost.prompt_read_cached_cost or 0

		# Calculate averages
		for stats in model_stats.values():
			if stats.invocations > 0:
				stats.average_tokens_per_invocation = stats.total_tokens / stats.invocations

		return UsageSummary(
			total_prompt_tokens=total_prompt,
			total_prompt_cost=total_prompt_cost,
			total_prompt_cached_tokens=total_prompt_cached,
			total_prompt_cached_cost=total_prompt_cached_cost,
			total_completion_tokens=total_completion,
			total_completion_cost=total_completion_cost,
			total_tokens=total_tokens,
			total_cost=total_prompt_cost + total_completion_cost + total_prompt_cached_cost,
			entry_count=len(filtered_usage),
			by_model=model_stats,
		)

	def _format_tokens(self, tokens: int) -> str:
		"""Format token count with k suffix for thousands"""
		if tokens >= 1000000000:
			return f'{tokens / 1000000000:.1f}B'
		if tokens >= 1000000:
			return f'{tokens / 1000000:.1f}M'
		if tokens >= 1000:
			return f'{tokens / 1000:.1f}k'
		return str(tokens)

	async def log_usage_summary(self) -> None:
		"""Log a comprehensive usage summary per model with colors and nice formatting"""
		if not self.usage_history:
			return

		summary = await self.get_usage_summary()

		if summary.entry_count == 0:
			return

		# ANSI color codes
		C_CYAN = '\033[96m'
		C_YELLOW = '\033[93m'
		C_GREEN = '\033[92m'
		C_BLUE = '\033[94m'
		C_MAGENTA = '\033[95m'
		C_RESET = '\033[0m'
		C_BOLD = '\033[1m'

		# Log overall summary
		total_tokens_fmt = self._format_tokens(summary.total_tokens)
		prompt_tokens_fmt = self._format_tokens(summary.total_prompt_tokens)
		completion_tokens_fmt = self._format_tokens(summary.total_completion_tokens)

		# Format cost breakdowns for input and output (only if cost tracking is enabled)
		if self.include_cost and summary.total_cost > 0:
			total_cost_part = f' (${C_MAGENTA}{summary.total_cost:.4f}{C_RESET})'
			prompt_cost_part = f' (${summary.total_prompt_cost:.4f})'
			completion_cost_part = f' (${summary.total_completion_cost:.4f})'
		else:
			total_cost_part = ''
			prompt_cost_part = ''
			completion_cost_part = ''

		if len(summary.by_model) > 1:
			cost_logger.info(
				f'💲 {C_BOLD}Total Usage Summary{C_RESET}: {C_BLUE}{total_tokens_fmt} tokens{C_RESET}{total_cost_part} | '
				f'⬅️ {C_YELLOW}{prompt_tokens_fmt}{prompt_cost_part}{C_RESET} | ➡️ {C_GREEN}{completion_tokens_fmt}{completion_cost_part}{C_RESET}'
			)

		# Log per-model breakdown
		cost_logger.info(f'📊 {C_BOLD}Per-Model Usage Breakdown{C_RESET}:')

		for model, stats in summary.by_model.items():
			# Format tokens
			model_total_fmt = self._format_tokens(stats.total_tokens)
			model_prompt_fmt = self._format_tokens(stats.prompt_tokens)
			model_completion_fmt = self._format_tokens(stats.completion_tokens)
			avg_tokens_fmt = self._format_tokens(int(stats.average_tokens_per_invocation))

			# Format cost display (only if cost tracking is enabled)
			if self.include_cost:
				# Calculate per-model costs on-the-fly
				total_model_cost = 0.0
				model_prompt_cost = 0.0
				model_completion_cost = 0.0

				# Calculate costs for this model
				for entry in self.usage_history:
					if entry.model == model:
						cost = await self.calculate_cost(entry.model, entry.usage)
						if cost:
							model_prompt_cost += cost.prompt_cost
							model_completion_cost += cost.completion_cost

				total_model_cost = model_prompt_cost + model_completion_cost

				if total_model_cost > 0:
					cost_part = f' (${C_MAGENTA}{total_model_cost:.4f}{C_RESET})'
					prompt_part = f'{C_YELLOW}{model_prompt_fmt} (${model_prompt_cost:.4f}){C_RESET}'
					completion_part = f'{C_GREEN}{model_completion_fmt} (${model_completion_cost:.4f}){C_RESET}'
				else:
					cost_part = ''
					prompt_part = f'{C_YELLOW}{model_prompt_fmt}{C_RESET}'
					completion_part = f'{C_GREEN}{model_completion_fmt}{C_RESET}'
			else:
				cost_part = ''
				prompt_part = f'{C_YELLOW}{model_prompt_fmt}{C_RESET}'
				completion_part = f'{C_GREEN}{model_completion_fmt}{C_RESET}'

			cost_logger.info(
				f'  🤖 {C_CYAN}{model}{C_RESET}: {C_BLUE}{model_total_fmt} tokens{C_RESET}{cost_part} | '
				f'⬅️ {prompt_part} | ➡️ {completion_part} | '
				f'📞 {stats.invocations} calls | 📈 {avg_tokens_fmt}/call'
			)

	async def get_cost_by_model(self) -> dict[str, ModelUsageStats]:
		"""Get cost breakdown by model"""
		summary = await self.get_usage_summary()
		return summary.by_model

	def clear_history(self) -> None:
		"""Clear usage history"""
		self.usage_history = []

	async def refresh_pricing_data(self) -> None:
		"""Force refresh of pricing data from GitHub"""
		if self.include_cost:
			await self._fetch_and_cache_pricing_data()

	async def clean_old_caches(self, keep_count: int = 3) -> None:
		"""Clean up old cache files, keeping only the most recent ones"""
		try:
			# List all JSON files in the cache directory
			cache_files = list(self._cache_dir.glob('*.json'))

			if len(cache_files) <= keep_count:
				return

			# Sort by modification time (oldest first)
			cache_files.sort(key=lambda f: f.stat().st_mtime)

			# Remove all but the most recent files
			for cache_file in cache_files[:-keep_count]:
				try:
					os.remove(cache_file)
				except Exception:
					pass
		except Exception as e:
			print(f'Error cleaning old cache files: {e}')

	async def ensure_pricing_loaded(self) -> None:
		"""Ensure pricing data is loaded in the background. Call this after creating the service."""
		if not self._initialized and self.include_cost:
			# This will run in the background and won't block
			await self.initialize()

# From tokens/service.py
def xdg_cache_home() -> Path:
	default = Path.home() / '.cache'
	if CONFIG.XDG_CACHE_HOME and (path := Path(CONFIG.XDG_CACHE_HOME)).is_absolute():
		return path
	return default

# From tokens/service.py
def add_usage(self, model: str, usage: ChatInvokeUsage) -> TokenUsageEntry:
		"""Add token usage entry to history (without calculating cost)"""
		entry = TokenUsageEntry(
			model=model,
			timestamp=datetime.now(),
			usage=usage,
		)

		self.usage_history.append(entry)

		return entry

# From tokens/service.py
def register_llm(self, llm: BaseChatModel) -> BaseChatModel:
		"""
		Register an LLM to automatically track its token usage

		@dev Guarantees that the same instance is not registered multiple times
		"""
		# Use instance ID as key to avoid collisions between multiple instances
		instance_id = str(id(llm))

		# Check if this exact instance is already registered
		if instance_id in self.registered_llms:
			logger.debug(f'LLM instance {instance_id} ({llm.provider}_{llm.model}) is already registered')
			return llm

		self.registered_llms[instance_id] = llm

		# Store the original method
		original_ainvoke = llm.ainvoke
		# Store reference to self for use in the closure
		token_cost_service = self

		# Create a wrapped version that tracks usage
		async def tracked_ainvoke(messages, output_format=None):
			# Call the original method
			result = await original_ainvoke(messages, output_format)

			# Track usage if available (no await needed since add_usage is now sync)
			if result.usage:
				usage = token_cost_service.add_usage(llm.model, result.usage)

				logger.debug(f'Token cost service: {usage}')

				asyncio.create_task(token_cost_service._log_usage(llm.model, usage))

			# else:
			# 	await token_cost_service._log_non_usage_llm(llm)

			return result

		# Replace the method with our tracked version
		# Using setattr to avoid type checking issues with overloaded methods
		setattr(llm, 'ainvoke', tracked_ainvoke)

		return llm

# From tokens/service.py
def get_usage_tokens_for_model(self, model: str) -> ModelUsageTokens:
		"""Get usage tokens for a specific model"""
		filtered_usage = [u for u in self.usage_history if u.model == model]

		return ModelUsageTokens(
			model=model,
			prompt_tokens=sum(u.usage.prompt_tokens for u in filtered_usage),
			prompt_cached_tokens=sum(u.usage.prompt_cached_tokens or 0 for u in filtered_usage),
			completion_tokens=sum(u.usage.completion_tokens for u in filtered_usage),
			total_tokens=sum(u.usage.prompt_tokens + u.usage.completion_tokens for u in filtered_usage),
		)

# From tokens/service.py
def clear_history(self) -> None:
		"""Clear usage history"""
		self.usage_history = []


# From tokens/views.py
class TokenUsageEntry(BaseModel):
	"""Single token usage entry"""

	model: str
	timestamp: datetime
	usage: ChatInvokeUsage

# From tokens/views.py
class TokenCostCalculated(BaseModel):
	"""Token cost"""

	new_prompt_tokens: int
	new_prompt_cost: float

	prompt_read_cached_tokens: int | None
	prompt_read_cached_cost: float | None

	prompt_cached_creation_tokens: int | None
	prompt_cache_creation_cost: float | None
	"""Anthropic only: The cost of creating the cache."""

	completion_tokens: int
	completion_cost: float

	@property
	def prompt_cost(self) -> float:
		return self.new_prompt_cost + (self.prompt_read_cached_cost or 0) + (self.prompt_cache_creation_cost or 0)

	@property
	def total_cost(self) -> float:
		return (
			self.new_prompt_cost
			+ (self.prompt_read_cached_cost or 0)
			+ (self.prompt_cache_creation_cost or 0)
			+ self.completion_cost
		)

# From tokens/views.py
class ModelPricing(BaseModel):
	"""Pricing information for a model"""

	model: str
	input_cost_per_token: float | None
	output_cost_per_token: float | None

	cache_read_input_token_cost: float | None
	cache_creation_input_token_cost: float | None

	max_tokens: int | None
	max_input_tokens: int | None
	max_output_tokens: int | None

# From tokens/views.py
class CachedPricingData(BaseModel):
	"""Cached pricing data with timestamp"""

	timestamp: datetime
	data: dict[str, Any]

# From tokens/views.py
class ModelUsageStats(BaseModel):
	"""Usage statistics for a single model"""

	model: str
	prompt_tokens: int = 0
	completion_tokens: int = 0
	total_tokens: int = 0
	cost: float = 0.0
	invocations: int = 0
	average_tokens_per_invocation: float = 0.0

# From tokens/views.py
class ModelUsageTokens(BaseModel):
	"""Usage tokens for a single model"""

	model: str
	prompt_tokens: int
	prompt_cached_tokens: int
	completion_tokens: int
	total_tokens: int

# From tokens/views.py
class UsageSummary(BaseModel):
	"""Summary of token usage and costs"""

	total_prompt_tokens: int
	total_prompt_cost: float

	total_prompt_cached_tokens: int
	total_prompt_cached_cost: float

	total_completion_tokens: int
	total_completion_cost: float
	total_tokens: int
	total_cost: float
	entry_count: int

	by_model: dict[str, ModelUsageStats] = Field(default_factory=dict)

# From tokens/views.py
def prompt_cost(self) -> float:
		return self.new_prompt_cost + (self.prompt_read_cached_cost or 0) + (self.prompt_cache_creation_cost or 0)

# From tokens/views.py
def total_cost(self) -> float:
		return (
			self.new_prompt_cost
			+ (self.prompt_read_cached_cost or 0)
			+ (self.prompt_cache_creation_cost or 0)
			+ self.completion_cost
		)

import base64

# From screenshots/service.py
class ScreenshotService:
	"""Simple screenshot storage service that saves screenshots to disk"""

	def __init__(self, agent_directory: str | Path):
		"""Initialize with agent directory path"""
		self.agent_directory = Path(agent_directory) if isinstance(agent_directory, str) else agent_directory

		# Create screenshots subdirectory
		self.screenshots_dir = self.agent_directory / 'screenshots'
		self.screenshots_dir.mkdir(parents=True, exist_ok=True)

	async def store_screenshot(self, screenshot_b64: str, step_number: int) -> str:
		"""Store screenshot to disk and return the full path as string"""
		screenshot_filename = f'step_{step_number}.png'
		screenshot_path = self.screenshots_dir / screenshot_filename

		# Decode base64 and save to disk
		screenshot_data = base64.b64decode(screenshot_b64)

		async with await anyio.open_file(screenshot_path, 'wb') as f:
			await f.write(screenshot_data)

		return str(screenshot_path)

	async def get_screenshot(self, screenshot_path: str) -> str | None:
		"""Load screenshot from disk path and return as base64"""
		if not screenshot_path:
			return None

		path = Path(screenshot_path)
		if not path.exists():
			return None

		# Load from disk and encode to base64
		async with await anyio.open_file(path, 'rb') as f:
			screenshot_data = await f.read()

		return base64.b64encode(screenshot_data).decode('utf-8')

from browser_use import ActionModel
from browser_use.config import get_default_llm
from browser_use.config import get_default_profile
from browser_use.config import load_browser_use_config
from browser_use.controller.service import Controller
from browser_use.filesystem.file_system import FileSystem
from browser_use.telemetry import MCPServerTelemetryEvent
import mcp.server.stdio
import mcp.types
from mcp.server import NotificationOptions
from mcp.server import Server
from mcp.server.models import InitializationOptions
from pydantic import create_model

# From mcp/server.py
class BrowserUseServer:
	"""MCP Server for browser-use capabilities."""

	def __init__(self):
		# Ensure all logging goes to stderr (in case new loggers were created)
		_ensure_all_loggers_use_stderr()

		self.server = Server('browser-use')
		self.config = load_browser_use_config()
		self.agent: Agent | None = None
		self.browser_session: BrowserSession | None = None
		self.controller: Controller | None = None
		self.llm: ChatOpenAI | None = None
		self.file_system: FileSystem | None = None
		self._telemetry = ProductTelemetry()
		self._start_time = time.time()

		# Setup handlers
		self._setup_handlers()

	def _setup_handlers(self):
		"""Setup MCP server handlers."""

		@self.server.list_tools()
		async def handle_list_tools() -> list[types.Tool]:
			"""List all available browser-use tools."""
			return [
				# Agent tools
				# Direct browser control tools
				types.Tool(
					name='browser_navigate',
					description='Navigate to a URL in the browser',
					inputSchema={
						'type': 'object',
						'properties': {
							'url': {'type': 'string', 'description': 'The URL to navigate to'},
							'new_tab': {'type': 'boolean', 'description': 'Whether to open in a new tab', 'default': False},
						},
						'required': ['url'],
					},
				),
				types.Tool(
					name='browser_click',
					description='Click an element on the page by its index',
					inputSchema={
						'type': 'object',
						'properties': {
							'index': {
								'type': 'integer',
								'description': 'The index of the link or element to click (from browser_get_state)',
							},
							'new_tab': {
								'type': 'boolean',
								'description': 'Whether to open any resulting navigation in a new tab',
								'default': False,
							},
						},
						'required': ['index'],
					},
				),
				types.Tool(
					name='browser_type',
					description='Type text into an input field',
					inputSchema={
						'type': 'object',
						'properties': {
							'index': {
								'type': 'integer',
								'description': 'The index of the input element (from browser_get_state)',
							},
							'text': {'type': 'string', 'description': 'The text to type'},
						},
						'required': ['index', 'text'],
					},
				),
				types.Tool(
					name='browser_get_state',
					description='Get the current state of the page including all interactive elements',
					inputSchema={
						'type': 'object',
						'properties': {
							'include_screenshot': {
								'type': 'boolean',
								'description': 'Whether to include a screenshot of the current page',
								'default': False,
							}
						},
					},
				),
				types.Tool(
					name='browser_extract_content',
					description='Extract structured content from the current page based on a query',
					inputSchema={
						'type': 'object',
						'properties': {
							'query': {'type': 'string', 'description': 'What information to extract from the page'},
							'extract_links': {
								'type': 'boolean',
								'description': 'Whether to include links in the extraction',
								'default': False,
							},
						},
						'required': ['query'],
					},
				),
				types.Tool(
					name='browser_scroll',
					description='Scroll the page',
					inputSchema={
						'type': 'object',
						'properties': {
							'direction': {
								'type': 'string',
								'enum': ['up', 'down'],
								'description': 'Direction to scroll',
								'default': 'down',
							}
						},
					},
				),
				types.Tool(
					name='browser_go_back',
					description='Go back to the previous page',
					inputSchema={'type': 'object', 'properties': {}},
				),
				# Tab management
				types.Tool(
					name='browser_list_tabs', description='List all open tabs', inputSchema={'type': 'object', 'properties': {}}
				),
				types.Tool(
					name='browser_switch_tab',
					description='Switch to a different tab',
					inputSchema={
						'type': 'object',
						'properties': {'tab_index': {'type': 'integer', 'description': 'Index of the tab to switch to'}},
						'required': ['tab_index'],
					},
				),
				types.Tool(
					name='browser_close_tab',
					description='Close a tab',
					inputSchema={
						'type': 'object',
						'properties': {'tab_index': {'type': 'integer', 'description': 'Index of the tab to close'}},
						'required': ['tab_index'],
					},
				),
				# types.Tool(
				# 	name="browser_close",
				# 	description="Close the browser session",
				# 	inputSchema={
				# 		"type": "object",
				# 		"properties": {}
				# 	}
				# ),
				types.Tool(
					name='retry_with_browser_use_agent',
					description='Retry a task using the browser-use agent. Only use this as a last resort if you fail to interact with a page multiple times.',
					inputSchema={
						'type': 'object',
						'properties': {
							'task': {
								'type': 'string',
								'description': 'The high-level goal and detailed step-by-step description of the task the AI browser agent needs to attempt, along with any relevant data needed to complete the task and info about previous attempts.',
							},
							'max_steps': {
								'type': 'integer',
								'description': 'Maximum number of steps the agent can take',
								'default': 100,
							},
							'model': {
								'type': 'string',
								'description': 'LLM model to use (e.g., gpt-4o, claude-3-opus-20240229)',
								'default': 'gpt-4o',
							},
							'allowed_domains': {
								'type': 'array',
								'items': {'type': 'string'},
								'description': 'List of domains the agent is allowed to visit (security feature)',
								'default': [],
							},
							'use_vision': {
								'type': 'boolean',
								'description': 'Whether to use vision capabilities (screenshots) for the agent',
								'default': True,
							},
						},
						'required': ['task'],
					},
				),
			]

		@self.server.call_tool()
		async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
			"""Handle tool execution."""
			start_time = time.time()
			error_msg = None
			try:
				result = await self._execute_tool(name, arguments or {})
				return [types.TextContent(type='text', text=result)]
			except Exception as e:
				error_msg = str(e)
				logger.error(f'Tool execution failed: {e}', exc_info=True)
				return [types.TextContent(type='text', text=f'Error: {str(e)}')]
			finally:
				# Capture telemetry for tool calls
				duration = time.time() - start_time
				self._telemetry.capture(
					MCPServerTelemetryEvent(
						version=get_browser_use_version(),
						action='tool_call',
						tool_name=name,
						duration_seconds=duration,
						error_message=error_msg,
					)
				)

	async def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
		"""Execute a browser-use tool."""

		# Agent-based tools
		if tool_name == 'retry_with_browser_use_agent':
			return await self._retry_with_browser_use_agent(
				task=arguments['task'],
				max_steps=arguments.get('max_steps', 100),
				model=arguments.get('model', 'gpt-4o'),
				allowed_domains=arguments.get('allowed_domains', []),
				use_vision=arguments.get('use_vision', True),
			)

		# Direct browser control tools (require active session)
		if tool_name.startswith('browser_'):
			# Ensure browser session exists
			if not self.browser_session:
				await self._init_browser_session()

			if tool_name == 'browser_navigate':
				return await self._navigate(arguments['url'], arguments.get('new_tab', False))

			elif tool_name == 'browser_click':
				return await self._click(arguments['index'], arguments.get('new_tab', False))

			elif tool_name == 'browser_type':
				return await self._type_text(arguments['index'], arguments['text'])

			elif tool_name == 'browser_get_state':
				return await self._get_browser_state(arguments.get('include_screenshot', False))

			elif tool_name == 'browser_extract_content':
				return await self._extract_content(arguments['query'], arguments.get('extract_links', False))

			elif tool_name == 'browser_scroll':
				return await self._scroll(arguments.get('direction', 'down'))

			elif tool_name == 'browser_go_back':
				return await self._go_back()

			elif tool_name == 'browser_close':
				return await self._close_browser()

			elif tool_name == 'browser_list_tabs':
				return await self._list_tabs()

			elif tool_name == 'browser_switch_tab':
				return await self._switch_tab(arguments['tab_index'])

			elif tool_name == 'browser_close_tab':
				return await self._close_tab(arguments['tab_index'])

		return f'Unknown tool: {tool_name}'

	async def _init_browser_session(self, allowed_domains: list[str] | None = None, **kwargs):
		"""Initialize browser session using config"""
		if self.browser_session:
			return

		# Ensure all logging goes to stderr before browser initialization
		_ensure_all_loggers_use_stderr()

		logger.debug('Initializing browser session...')

		# Get profile config
		profile_config = get_default_profile(self.config)

		# Merge profile config with defaults and overrides
		profile_data = {
			'downloads_path': str(Path.home() / 'Downloads' / 'browser-use-mcp'),
			'wait_between_actions': 0.5,
			'keep_alive': True,
			'user_data_dir': '~/.config/browseruse/profiles/default',
			'is_mobile': False,
			'device_scale_factor': 1.0,
			'disable_security': False,
			'headless': False,
			**profile_config,  # Config values override defaults
		}

		# Tool parameter overrides (highest priority)
		if allowed_domains is not None:
			profile_data['allowed_domains'] = allowed_domains

		# Merge any additional kwargs that are valid BrowserProfile fields
		for key, value in kwargs.items():
			profile_data[key] = value

		# Create browser profile
		profile = BrowserProfile(**profile_data)

		# Create browser session
		self.browser_session = BrowserSession(browser_profile=profile)
		await self.browser_session.start()

		# Create controller for direct actions
		self.controller = Controller()

		# Initialize LLM from config
		llm_config = get_default_llm(self.config)
		if api_key := llm_config.get('api_key'):
			self.llm = ChatOpenAI(
				model=llm_config.get('model', 'gpt-4o-mini'),
				api_key=api_key,
				temperature=llm_config.get('temperature', 0.7),
				# max_tokens=llm_config.get('max_tokens'),
			)

		# Initialize FileSystem for extraction actions
		file_system_path = profile_config.get('file_system_path', '~/.browser-use-mcp')
		self.file_system = FileSystem(base_dir=Path(file_system_path).expanduser())

		logger.debug('Browser session initialized')

	async def _retry_with_browser_use_agent(
		self,
		task: str,
		max_steps: int = 100,
		model: str = 'gpt-4o',
		allowed_domains: list[str] | None = None,
		use_vision: bool = True,
	) -> str:
		"""Run an autonomous agent task."""
		logger.debug(f'Running agent task: {task}')

		# Get LLM config
		llm_config = get_default_llm(self.config)
		api_key = llm_config.get('api_key') or os.getenv('OPENAI_API_KEY')
		if not api_key:
			return 'Error: OPENAI_API_KEY not set in config or environment'

		# Override model if provided in tool call
		if model != llm_config.get('model', 'gpt-4o'):
			llm_model = model
		else:
			llm_model = llm_config.get('model', 'gpt-4o')

		llm = ChatOpenAI(
			model=llm_model,
			api_key=api_key,
			temperature=llm_config.get('temperature', 0.7),
		)

		# Get profile config and merge with tool parameters
		profile_config = get_default_profile(self.config)

		# Override allowed_domains if provided in tool call
		if allowed_domains is not None:
			profile_config['allowed_domains'] = allowed_domains

		# Create browser profile using config
		profile = BrowserProfile(**profile_config)

		# Create and run agent
		agent = Agent(
			task=task,
			llm=llm,
			browser_profile=profile,
			use_vision=use_vision,
		)

		try:
			history = await agent.run(max_steps=max_steps)

			# Format results
			results = []
			results.append(f'Task completed in {len(history.history)} steps')
			results.append(f'Success: {history.is_successful()}')

			# Get final result if available
			final_result = history.final_result()
			if final_result:
				results.append(f'\nFinal result:\n{final_result}')

			# Include any errors
			errors = history.errors()
			if errors:
				results.append(f'\nErrors encountered:\n{json.dumps(errors, indent=2)}')

			# Include URLs visited
			urls = history.urls()
			if urls:
				# Filter out None values and convert to strings
				valid_urls = [str(url) for url in urls if url is not None]
				if valid_urls:
					results.append(f'\nURLs visited: {", ".join(valid_urls)}')

			return '\n'.join(results)

		except Exception as e:
			logger.error(f'Agent task failed: {e}', exc_info=True)
			return f'Agent task failed: {str(e)}'
		finally:
			# Clean up
			await agent.close()

	async def _navigate(self, url: str, new_tab: bool = False) -> str:
		"""Navigate to a URL."""
		if not self.browser_session:
			return 'Error: No browser session active'

		if new_tab:
			page = await self.browser_session.create_new_tab(url)
			tab_idx = self.browser_session.tabs.index(page)
			return f'Opened new tab #{tab_idx} with URL: {url}'
		else:
			await self.browser_session.navigate_to(url)
			return f'Navigated to: {url}'

	async def _click(self, index: int, new_tab: bool = False) -> str:
		"""Click an element by index."""
		if not self.browser_session:
			return 'Error: No browser session active'

		# Get the element
		element = await self.browser_session.get_dom_element_by_index(index)
		if not element:
			return f'Element with index {index} not found'

		if new_tab:
			# For links, extract href and open in new tab
			href = element.attributes.get('href')
			if href:
				# Convert relative href to absolute URL
				current_page = await self.browser_session.get_current_page()
				if href.startswith('/'):
					# Relative URL - construct full URL
					from urllib.parse import urlparse

					parsed = urlparse(current_page.url)
					full_url = f'{parsed.scheme}://{parsed.netloc}{href}'
				else:
					full_url = href

				# Open link in new tab
				page = await self.browser_session.create_new_tab(full_url)
				tab_idx = self.browser_session.tabs.index(page)
				return f'Clicked element {index} and opened in new tab #{tab_idx}'
			else:
				# For non-link elements, try Cmd/Ctrl+Click
				page = await self.browser_session.get_current_page()
				element_handle = await self.browser_session.get_locate_element(element)
				if element_handle:
					# Use playwright's click with modifiers
					modifier: Literal['Meta', 'Control'] = 'Meta' if sys.platform == 'darwin' else 'Control'
					await element_handle.click(modifiers=[modifier])
					# Wait a bit for potential new tab
					await asyncio.sleep(0.5)
					return f'Clicked element {index} with {modifier} key (new tab if supported)'
				else:
					return f'Could not locate element {index} for modified click'
		else:
			# Normal click
			await self.browser_session._click_element_node(element)
			return f'Clicked element {index}'

	async def _type_text(self, index: int, text: str) -> str:
		"""Type text into an element."""
		if not self.browser_session:
			return 'Error: No browser session active'

		element = await self.browser_session.get_dom_element_by_index(index)
		if not element:
			return f'Element with index {index} not found'

		await self.browser_session._input_text_element_node(element, text)
		return f"Typed '{text}' into element {index}"

	async def _get_browser_state(self, include_screenshot: bool = False) -> str:
		"""Get current browser state."""
		if not self.browser_session:
			return 'Error: No browser session active'

		state = await self.browser_session.get_browser_state_with_recovery(cache_clickable_elements_hashes=False)

		result = {
			'url': state.url,
			'title': state.title,
			'tabs': [{'url': tab.url, 'title': tab.title} for tab in state.tabs],
			'interactive_elements': [],
		}

		# Add interactive elements with their indices
		for index, element in state.selector_map.items():
			elem_info = {
				'index': index,
				'tag': element.tag_name,
				'text': element.get_all_text_till_next_clickable_element(max_depth=2)[:100],
			}
			if element.attributes.get('placeholder'):
				elem_info['placeholder'] = element.attributes['placeholder']
			if element.attributes.get('href'):
				elem_info['href'] = element.attributes['href']
			result['interactive_elements'].append(elem_info)

		if include_screenshot and state.screenshot:
			result['screenshot'] = state.screenshot

		return json.dumps(result, indent=2)

	async def _extract_content(self, query: str, extract_links: bool = False) -> str:
		"""Extract content from current page."""
		if not self.llm:
			return 'Error: LLM not initialized (set OPENAI_API_KEY)'

		if not self.file_system:
			return 'Error: FileSystem not initialized'

		if not self.browser_session:
			return 'Error: No browser session active'

		if not self.controller:
			return 'Error: Controller not initialized'

		page = await self.browser_session.get_current_page()

		# Use the extract_structured_data action
		# Create a dynamic action model that matches the controller's expectations
		from pydantic import create_model

		# Create action model dynamically
		ExtractAction = create_model(
			'ExtractAction',
			__base__=ActionModel,
			extract_structured_data=(dict[str, Any], {'query': query, 'extract_links': extract_links}),
		)

		action = ExtractAction()
		action_result = await self.controller.act(
			action=action,
			browser_session=self.browser_session,
			page_extraction_llm=self.llm,
			file_system=self.file_system,
		)

		return action_result.extracted_content or 'No content extracted'

	async def _scroll(self, direction: str = 'down') -> str:
		"""Scroll the page."""
		if not self.browser_session:
			return 'Error: No browser session active'

		page = await self.browser_session.get_current_page()

		# Get viewport height
		viewport_height = await page.evaluate('() => window.innerHeight')
		dy = viewport_height if direction == 'down' else -viewport_height

		await page.evaluate('(y) => window.scrollBy(0, y)', dy)
		return f'Scrolled {direction}'

	async def _go_back(self) -> str:
		"""Go back in browser history."""
		if not self.browser_session:
			return 'Error: No browser session active'

		await self.browser_session.go_back()
		return 'Navigated back'

	async def _close_browser(self) -> str:
		"""Close the browser session."""
		if self.browser_session:
			await self.browser_session.stop()
			self.browser_session = None
			self.controller = None
			return 'Browser closed'
		return 'No browser session to close'

	async def _list_tabs(self) -> str:
		"""List all open tabs."""
		if not self.browser_session:
			return 'Error: No browser session active'

		tabs = []
		for i, tab in enumerate(self.browser_session.tabs):
			tabs.append({'index': i, 'url': tab.url, 'title': await tab.title() if not tab.is_closed() else 'Closed'})
		return json.dumps(tabs, indent=2)

	async def _switch_tab(self, tab_index: int) -> str:
		"""Switch to a different tab."""
		if not self.browser_session:
			return 'Error: No browser session active'

		await self.browser_session.switch_to_tab(tab_index)
		page = await self.browser_session.get_current_page()
		return f'Switched to tab {tab_index}: {page.url}'

	async def _close_tab(self, tab_index: int) -> str:
		"""Close a specific tab."""
		if not self.browser_session:
			return 'Error: No browser session active'

		if 0 <= tab_index < len(self.browser_session.tabs):
			tab = self.browser_session.tabs[tab_index]
			url = tab.url
			await tab.close()
			return f'Closed tab {tab_index}: {url}'
		return f'Invalid tab index: {tab_index}'

	async def run(self):
		"""Run the MCP server."""
		async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
			await self.server.run(
				read_stream,
				write_stream,
				InitializationOptions(
					server_name='browser-use',
					server_version='0.1.0',
					capabilities=self.server.get_capabilities(
						notification_options=NotificationOptions(),
						experimental_capabilities={},
					),
				),
			)

# From mcp/server.py
def get_parent_process_cmdline() -> str | None:
	"""Get the command line of all parent processes up the chain."""
	if not PSUTIL_AVAILABLE:
		return None

	try:
		cmdlines = []
		current_process = psutil.Process()
		parent = current_process.parent()

		while parent:
			try:
				cmdline = parent.cmdline()
				if cmdline:
					cmdlines.append(' '.join(cmdline))
			except (psutil.AccessDenied, psutil.NoSuchProcess):
				# Skip processes we can't access (like system processes)
				pass

			try:
				parent = parent.parent()
			except (psutil.AccessDenied, psutil.NoSuchProcess):
				# Can't go further up the chain
				break

		return ';'.join(cmdlines) if cmdlines else None
	except Exception:
		# If we can't get parent process info, just return None
		return None

from browser_use.agent.views import ActionResult
from browser_use.controller.registry.service import Registry
from browser_use.telemetry import MCPClientTelemetryEvent
from browser_use.utils import is_new_tab_page
from mcp import ClientSession
from mcp import StdioServerParameters
from mcp import types
from mcp.client.stdio import stdio_client

# From mcp/client.py
class MCPClient:
	"""Client for connecting to MCP servers and exposing their tools as browser-use actions."""

	def __init__(
		self,
		server_name: str,
		command: str,
		args: list[str] | None = None,
		env: dict[str, str] | None = None,
	):
		"""Initialize MCP client.

		Args:
			server_name: Name of the MCP server (for logging and identification)
			command: Command to start the MCP server (e.g., "npx", "python")
			args: Arguments for the command (e.g., ["@playwright/mcp@latest"])
			env: Environment variables for the server process
		"""
		self.server_name = server_name
		self.command = command
		self.args = args or []
		self.env = env

		self.session: ClientSession | None = None
		self._stdio_task = None
		self._read_stream = None
		self._write_stream = None
		self._tools: dict[str, types.Tool] = {}
		self._registered_actions: set[str] = set()
		self._connected = False
		self._disconnect_event = asyncio.Event()
		self._telemetry = ProductTelemetry()

	async def connect(self) -> None:
		"""Connect to the MCP server and discover available tools."""
		if self._connected:
			logger.debug(f'Already connected to {self.server_name}')
			return

		start_time = time.time()
		error_msg = None

		try:
			logger.info(f"🔌 Connecting to MCP server '{self.server_name}': {self.command} {' '.join(self.args)}")

			# Create server parameters
			server_params = StdioServerParameters(command=self.command, args=self.args, env=self.env)

			# Start stdio client in background task
			self._stdio_task = asyncio.create_task(self._run_stdio_client(server_params))

			# Wait for connection to be established
			retries = 0
			max_retries = 100  # 10 second timeout (increased for parallel test execution)
			while not self._connected and retries < max_retries:
				await asyncio.sleep(0.1)
				retries += 1

			if not self._connected:
				error_msg = f"Failed to connect to MCP server '{self.server_name}' after {max_retries * 0.1} seconds"
				raise RuntimeError(error_msg)

			logger.info(f"📦 Discovered {len(self._tools)} tools from '{self.server_name}': {list(self._tools.keys())}")

		except Exception as e:
			error_msg = str(e)
			raise
		finally:
			# Capture telemetry for connect action
			duration = time.time() - start_time
			self._telemetry.capture(
				MCPClientTelemetryEvent(
					server_name=self.server_name,
					command=self.command,
					tools_discovered=len(self._tools),
					version=get_browser_use_version(),
					action='connect',
					duration_seconds=duration,
					error_message=error_msg,
				)
			)

	async def _run_stdio_client(self, server_params: StdioServerParameters):
		"""Run the stdio client connection in a background task."""
		try:
			async with stdio_client(server_params) as (read_stream, write_stream):
				self._read_stream = read_stream
				self._write_stream = write_stream

				# Create and initialize session
				async with ClientSession(read_stream, write_stream) as session:
					self.session = session

					# Initialize the connection
					await session.initialize()

					# Discover available tools
					tools_response = await session.list_tools()
					self._tools = {tool.name: tool for tool in tools_response.tools}

					# Mark as connected
					self._connected = True

					# Keep the connection alive until disconnect is called
					await self._disconnect_event.wait()

		except Exception as e:
			logger.error(f'MCP server connection error: {e}')
			self._connected = False
			raise
		finally:
			self._connected = False
			self.session = None

	async def disconnect(self) -> None:
		"""Disconnect from the MCP server."""
		if not self._connected:
			return

		start_time = time.time()
		error_msg = None

		try:
			logger.info(f"🔌 Disconnecting from MCP server '{self.server_name}'")

			# Signal disconnect
			self._connected = False
			self._disconnect_event.set()

			# Wait for stdio task to finish
			if self._stdio_task:
				try:
					await asyncio.wait_for(self._stdio_task, timeout=2.0)
				except TimeoutError:
					logger.warning(f"Timeout waiting for MCP server '{self.server_name}' to disconnect")
					self._stdio_task.cancel()
					try:
						await self._stdio_task
					except asyncio.CancelledError:
						pass

			self._tools.clear()
			self._registered_actions.clear()

		except Exception as e:
			error_msg = str(e)
			logger.error(f'Error disconnecting from MCP server: {e}')
		finally:
			# Capture telemetry for disconnect action
			duration = time.time() - start_time
			self._telemetry.capture(
				MCPClientTelemetryEvent(
					server_name=self.server_name,
					command=self.command,
					tools_discovered=0,  # Tools cleared on disconnect
					version=get_browser_use_version(),
					action='disconnect',
					duration_seconds=duration,
					error_message=error_msg,
				)
			)
			self._telemetry.flush()

	async def register_to_controller(
		self,
		controller: Controller,
		tool_filter: list[str] | None = None,
		prefix: str | None = None,
	) -> None:
		"""Register MCP tools as actions in the browser-use controller.

		Args:
			controller: Browser-use controller to register actions to
			tool_filter: Optional list of tool names to register (None = all tools)
			prefix: Optional prefix to add to action names (e.g., "playwright_")
		"""
		if not self._connected:
			await self.connect()

		registry = controller.registry

		for tool_name, tool in self._tools.items():
			# Skip if not in filter
			if tool_filter and tool_name not in tool_filter:
				continue

			# Apply prefix if specified
			action_name = f'{prefix}{tool_name}' if prefix else tool_name

			# Skip if already registered
			if action_name in self._registered_actions:
				continue

			# Register the tool as an action
			self._register_tool_as_action(registry, action_name, tool)
			self._registered_actions.add(action_name)

		logger.info(f"✅ Registered {len(self._registered_actions)} MCP tools from '{self.server_name}' as browser-use actions")

	def _register_tool_as_action(self, registry: Registry, action_name: str, tool: Any) -> None:
		"""Register a single MCP tool as a browser-use action.

		Args:
			registry: Browser-use registry to register action to
			action_name: Name for the registered action
			tool: MCP Tool object with schema information
		"""
		# Parse tool parameters to create Pydantic model
		param_fields = {}

		if tool.inputSchema:
			# MCP tools use JSON Schema for parameters
			properties = tool.inputSchema.get('properties', {})
			required = set(tool.inputSchema.get('required', []))

			for param_name, param_schema in properties.items():
				# Convert JSON Schema type to Python type
				param_type = self._json_schema_to_python_type(param_schema, f'{action_name}_{param_name}')

				# Determine if field is required and handle defaults
				if param_name in required:
					default = ...  # Required field
				else:
					# Optional field - make type optional and handle default
					param_type = param_type | None
					if 'default' in param_schema:
						default = param_schema['default']
					else:
						default = None

				# Add field with description if available
				field_kwargs = {}
				if 'description' in param_schema:
					field_kwargs['description'] = param_schema['description']

				param_fields[param_name] = (param_type, Field(default, **field_kwargs))

		# Create Pydantic model for the tool parameters
		if param_fields:
			# Create a BaseModel class with proper configuration
			class ConfiguredBaseModel(BaseModel):
				model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

			param_model = create_model(f'{action_name}_Params', __base__=ConfiguredBaseModel, **param_fields)
		else:
			# No parameters - create empty model
			param_model = None

		# Determine if this is a browser-specific tool
		is_browser_tool = tool.name.startswith('browser_') or 'page' in tool.name.lower()

		# Set up action filters
		domains = None
		page_filter = None

		if is_browser_tool:
			# Browser tools should only be available when on a web page
			page_filter = lambda page: page and not is_new_tab_page(page.url)

		# Create async wrapper function for the MCP tool
		# Need to define function with explicit parameters to satisfy registry validation
		if param_model:
			# Type 1: Function takes param model as first parameter
			async def mcp_action_wrapper(params: param_model) -> ActionResult:  # type: ignore[no-redef]
				"""Wrapper function that calls the MCP tool."""
				if not self.session or not self._connected:
					return ActionResult(error=f"MCP server '{self.server_name}' not connected", success=False)

				# Convert pydantic model to dict for MCP call
				tool_params = params.model_dump(exclude_none=True)

				logger.debug(f"🔧 Calling MCP tool '{tool.name}' with params: {tool_params}")

				start_time = time.time()
				error_msg = None

				try:
					# Call the MCP tool
					result = await self.session.call_tool(tool.name, tool_params)

					# Convert MCP result to ActionResult
					extracted_content = self._format_mcp_result(result)

					return ActionResult(
						extracted_content=extracted_content,
						long_term_memory=f"Used MCP tool '{tool.name}' from {self.server_name}",
					)

				except Exception as e:
					error_msg = f"MCP tool '{tool.name}' failed: {str(e)}"
					logger.error(error_msg)
					return ActionResult(error=error_msg, success=False)
				finally:
					# Capture telemetry for tool call
					duration = time.time() - start_time
					self._telemetry.capture(
						MCPClientTelemetryEvent(
							server_name=self.server_name,
							command=self.command,
							tools_discovered=len(self._tools),
							version=get_browser_use_version(),
							action='tool_call',
							tool_name=tool.name,
							duration_seconds=duration,
							error_message=error_msg,
						)
					)
		else:
			# No parameters - empty function signature
			async def mcp_action_wrapper() -> ActionResult:  # type: ignore[no-redef]
				"""Wrapper function that calls the MCP tool."""
				if not self.session or not self._connected:
					return ActionResult(error=f"MCP server '{self.server_name}' not connected", success=False)

				logger.debug(f"🔧 Calling MCP tool '{tool.name}' with no params")

				start_time = time.time()
				error_msg = None

				try:
					# Call the MCP tool with empty params
					result = await self.session.call_tool(tool.name, {})

					# Convert MCP result to ActionResult
					extracted_content = self._format_mcp_result(result)

					return ActionResult(
						extracted_content=extracted_content,
						long_term_memory=f"Used MCP tool '{tool.name}' from {self.server_name}",
					)

				except Exception as e:
					error_msg = f"MCP tool '{tool.name}' failed: {str(e)}"
					logger.error(error_msg)
					return ActionResult(error=error_msg, success=False)
				finally:
					# Capture telemetry for tool call
					duration = time.time() - start_time
					self._telemetry.capture(
						MCPClientTelemetryEvent(
							server_name=self.server_name,
							command=self.command,
							tools_discovered=len(self._tools),
							version=get_browser_use_version(),
							action='tool_call',
							tool_name=tool.name,
							duration_seconds=duration,
							error_message=error_msg,
						)
					)

		# Set function metadata for better debugging
		mcp_action_wrapper.__name__ = action_name
		mcp_action_wrapper.__qualname__ = f'mcp.{self.server_name}.{action_name}'

		# Register the action with browser-use
		description = tool.description or f'MCP tool from {self.server_name}: {tool.name}'

		# Use the registry's action decorator
		registry.action(description=description, param_model=param_model, domains=domains, page_filter=page_filter)(
			mcp_action_wrapper
		)

		logger.debug(f"✅ Registered MCP tool '{tool.name}' as action '{action_name}'")

	def _format_mcp_result(self, result: Any) -> str:
		"""Format MCP tool result into a string for ActionResult.

		Args:
			result: Raw result from MCP tool call

		Returns:
			Formatted string representation of the result
		"""
		# Handle different MCP result formats
		if hasattr(result, 'content'):
			# Structured content response
			if isinstance(result.content, list):
				# Multiple content items
				parts = []
				for item in result.content:
					if hasattr(item, 'text'):
						parts.append(item.text)
					elif hasattr(item, 'type') and item.type == 'text':
						parts.append(str(item))
					else:
						parts.append(str(item))
				return '\n'.join(parts)
			else:
				return str(result.content)
		elif isinstance(result, list):
			# List of content items
			parts = []
			for item in result:
				if hasattr(item, 'text'):
					parts.append(item.text)
				else:
					parts.append(str(item))
			return '\n'.join(parts)
		else:
			# Direct result or unknown format
			return str(result)

	def _json_schema_to_python_type(self, schema: dict, model_name: str = 'NestedModel') -> Any:
		"""Convert JSON Schema type to Python type.

		Args:
			schema: JSON Schema definition
			model_name: Name for nested models

		Returns:
			Python type corresponding to the schema
		"""
		json_type = schema.get('type', 'string')

		# Basic type mapping
		type_mapping = {
			'string': str,
			'number': float,
			'integer': int,
			'boolean': bool,
			'array': list,
			'null': type(None),
		}

		# Handle enums (they're still strings)
		if 'enum' in schema:
			return str

		# Handle objects with nested properties
		if json_type == 'object':
			properties = schema.get('properties', {})
			if properties:
				# Create nested pydantic model for objects with properties
				nested_fields = {}
				required_fields = set(schema.get('required', []))

				for prop_name, prop_schema in properties.items():
					# Recursively process nested properties
					prop_type = self._json_schema_to_python_type(prop_schema, f'{model_name}_{prop_name}')

					# Determine if field is required and handle defaults
					if prop_name in required_fields:
						default = ...  # Required field
					else:
						# Optional field - make type optional and handle default
						prop_type = prop_type | None
						if 'default' in prop_schema:
							default = prop_schema['default']
						else:
							default = None

					# Add field with description if available
					field_kwargs = {}
					if 'description' in prop_schema:
						field_kwargs['description'] = prop_schema['description']

					nested_fields[prop_name] = (prop_type, Field(default, **field_kwargs))

				# Create a BaseModel class with proper configuration
				class ConfiguredBaseModel(BaseModel):
					model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

				try:
					# Create and return nested pydantic model
					return create_model(model_name, __base__=ConfiguredBaseModel, **nested_fields)
				except Exception as e:
					logger.error(f'Failed to create nested model {model_name}: {e}')
					logger.debug(f'Fields: {nested_fields}')
					# Fallback to basic dict if model creation fails
					return dict
			else:
				# Object without properties - just return dict
				return dict

		# Handle arrays with specific item types
		if json_type == 'array':
			if 'items' in schema:
				# Get the item type recursively
				item_type = self._json_schema_to_python_type(schema['items'], f'{model_name}_item')
				# Return properly typed list
				return list[item_type]
			else:
				# Array without item type specification
				return list

		# Get base type for non-object types
		base_type = type_mapping.get(json_type, str)

		# Handle nullable/optional types
		if schema.get('nullable', False) or json_type == 'null':
			return base_type | None

		return base_type

	async def __aenter__(self):
		"""Async context manager entry."""
		await self.connect()
		return self

	async def __aexit__(self, exc_type, exc_val, exc_tb):
		"""Async context manager exit."""
		await self.disconnect()

# From mcp/client.py
class ConfiguredBaseModel(BaseModel):
				model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

from mcp.types import TextContent
from mcp.types import Tool

# From mcp/controller.py
class MCPToolWrapper:
	"""Wrapper to integrate MCP tools as browser-use actions."""

	def __init__(self, registry: Registry, mcp_command: str, mcp_args: list[str] | None = None):
		"""Initialize MCP tool wrapper.

		Args:
			registry: Browser-use action registry to register MCP tools
			mcp_command: Command to start MCP server (e.g., "npx")
			mcp_args: Arguments for MCP command (e.g., ["@playwright/mcp@latest"])
		"""
		if not MCP_AVAILABLE:
			raise ImportError('MCP SDK not installed. Install with: pip install mcp')

		self.registry = registry
		self.mcp_command = mcp_command
		self.mcp_args = mcp_args or []
		self.session: ClientSession | None = None
		self._tools: dict[str, Tool] = {}
		self._registered_actions: set[str] = set()
		self._shutdown_event = asyncio.Event()

	async def connect(self):
		"""Connect to MCP server and discover available tools."""
		if self.session:
			return  # Already connected

		logger.info(f'🔌 Connecting to MCP server: {self.mcp_command} {" ".join(self.mcp_args)}')

		# Create server parameters
		server_params = StdioServerParameters(command=self.mcp_command, args=self.mcp_args, env=None)

		# Connect to the MCP server
		async with stdio_client(server_params) as (read, write):
			async with ClientSession(read, write) as session:
				self.session = session

				# Initialize the connection
				await session.initialize()

				# Discover available tools
				tools_response = await session.list_tools()
				self._tools = {tool.name: tool for tool in tools_response.tools}

				logger.info(f'📦 Discovered {len(self._tools)} MCP tools: {list(self._tools.keys())}')

				# Register all discovered tools as actions
				for tool_name, tool in self._tools.items():
					self._register_tool_as_action(tool_name, tool)

				# Keep session alive while tools are being used
				await self._keep_session_alive()

	async def _keep_session_alive(self):
		"""Keep the MCP session alive."""
		# This will block until the session is closed
		# In practice, you'd want to manage this lifecycle better
		try:
			await self._shutdown_event.wait()
		except asyncio.CancelledError:
			pass

	def _register_tool_as_action(self, tool_name: str, tool: Tool):
		"""Register an MCP tool as a browser-use action.

		Args:
			tool_name: Name of the MCP tool
			tool: MCP Tool object with schema information
		"""
		if tool_name in self._registered_actions:
			return  # Already registered

		# Parse tool parameters to create Pydantic model
		param_fields = {}

		if tool.inputSchema:
			# MCP tools use JSON Schema for parameters
			properties = tool.inputSchema.get('properties', {})
			required = set(tool.inputSchema.get('required', []))

			for param_name, param_schema in properties.items():
				# Convert JSON Schema type to Python type
				param_type = self._json_schema_to_python_type(param_schema)

				# Determine if field is required
				if param_name in required:
					default = ...  # Required field
				else:
					default = param_schema.get('default', None)

				# Add field description if available
				field_kwargs = {}
				if 'description' in param_schema:
					field_kwargs['description'] = param_schema['description']

				param_fields[param_name] = (param_type, Field(default, **field_kwargs))

		# Create Pydantic model for the tool parameters
		param_model = create_model(f'{tool_name}_Params', **param_fields) if param_fields else None

		# Determine if this is a browser-specific tool
		is_browser_tool = tool_name.startswith('browser_')
		domains = None
		page_filter = None

		if is_browser_tool:
			# Browser tools should only be available when on a web page
			page_filter = lambda page: not is_new_tab_page(page.url)

		# Create wrapper function for the MCP tool
		async def mcp_action_wrapper(**kwargs):
			"""Wrapper function that calls the MCP tool."""
			if not self.session:
				raise RuntimeError(f'MCP session not connected for tool {tool_name}')

			# Extract parameters (excluding special injected params)
			special_params = {
				'page',
				'browser_session',
				'context',
				'page_extraction_llm',
				'file_system',
				'available_file_paths',
				'has_sensitive_data',
				'browser',
				'browser_context',
			}

			tool_params = {k: v for k, v in kwargs.items() if k not in special_params}

			logger.debug(f'🔧 Calling MCP tool {tool_name} with params: {tool_params}')

			try:
				# Call the MCP tool
				result = await self.session.call_tool(tool_name, tool_params)

				# Convert MCP result to ActionResult
				# MCP tools return results in various formats
				if hasattr(result, 'content'):
					# Handle structured content responses
					if isinstance(result.content, list):
						# Multiple content items
						content_parts = []
						for item in result.content:
							if isinstance(item, TextContent):
								content_parts.append(item.text)  # type: ignore[reportAttributeAccessIssue]
							else:
								content_parts.append(str(item))
						extracted_content = '\n'.join(content_parts)
					else:
						extracted_content = str(result.content)
				else:
					# Direct result
					extracted_content = str(result)

				return ActionResult(extracted_content=extracted_content)

			except Exception as e:
				logger.error(f'❌ MCP tool {tool_name} failed: {e}')
				return ActionResult(extracted_content=f'MCP tool {tool_name} failed: {str(e)}', error=str(e))

		# Set function name for better debugging
		mcp_action_wrapper.__name__ = tool_name
		mcp_action_wrapper.__qualname__ = f'mcp.{tool_name}'

		# Register the action with browser-use
		description = tool.description or f'MCP tool: {tool_name}'

		# Use the decorator to register the action
		decorated_wrapper = self.registry.action(
			description=description, param_model=param_model, domains=domains, page_filter=page_filter
		)(mcp_action_wrapper)

		self._registered_actions.add(tool_name)
		logger.info(f'✅ Registered MCP tool as action: {tool_name}')

	async def disconnect(self):
		"""Disconnect from the MCP server and clean up resources."""
		self._shutdown_event.set()
		if self.session:
			# Session cleanup will be handled by the context manager
			self.session = None

	def _json_schema_to_python_type(self, schema: dict) -> Any:
		"""Convert JSON Schema type to Python type.

		Args:
			schema: JSON Schema definition

		Returns:
			Python type corresponding to the schema
		"""
		json_type = schema.get('type', 'string')

		type_mapping = {
			'string': str,
			'number': float,
			'integer': int,
			'boolean': bool,
			'array': list,
			'object': dict,
		}

		base_type = type_mapping.get(json_type, str)

		# Handle nullable types
		if schema.get('nullable', False):
			return base_type | None

		return base_type


# From dom/utils.py
def cap_text_length(text: str, max_length: int) -> str:
	if len(text) > max_length:
		return text[:max_length] + '...'
	return text

from importlib import resources
from typing import TYPE_CHECKING
from browser_use.dom.views import DOMBaseNode
from browser_use.dom.views import DOMElementNode
from browser_use.dom.views import DOMState
from browser_use.dom.views import DOMTextNode
from browser_use.dom.views import SelectorMap
from browser_use.dom.views import ViewportInfo
from browser_use.observability import observe_debug
from browser_use.utils import time_execution_async
from browser_use.browser.types import Page

# From dom/service.py
class DomService:
	logger: logging.Logger

	def __init__(self, page: 'Page', logger: logging.Logger | None = None):
		self.page = page
		self.xpath_cache = {}
		self.logger = logger or logging.getLogger(__name__)

		self.js_code = resources.files('browser_use.dom.dom_tree').joinpath('index.js').read_text()

	# region - Clickable elements
	@observe_debug(ignore_input=True, ignore_output=True, name='get_clickable_elements')
	@time_execution_async('--get_clickable_elements')
	async def get_clickable_elements(
		self,
		highlight_elements: bool = True,
		focus_element: int = -1,
		viewport_expansion: int = 0,
	) -> DOMState:
		element_tree, selector_map = await self._build_dom_tree(highlight_elements, focus_element, viewport_expansion)
		return DOMState(element_tree=element_tree, selector_map=selector_map)

	@time_execution_async('--get_cross_origin_iframes')
	async def get_cross_origin_iframes(self) -> list[str]:
		# invisible cross-origin iframes are used for ads and tracking, dont open those
		hidden_frame_urls = await self.page.locator('iframe').filter(visible=False).evaluate_all('e => e.map(e => e.src)')

		is_ad_url = lambda url: any(
			domain in urlparse(url).netloc for domain in ('doubleclick.net', 'adroll.com', 'googletagmanager.com')
		)

		return [
			frame.url
			for frame in self.page.frames
			if urlparse(frame.url).netloc  # exclude data:urls and new tab pages
			and urlparse(frame.url).netloc != urlparse(self.page.url).netloc  # exclude same-origin iframes
			and frame.url not in hidden_frame_urls  # exclude hidden frames
			and not is_ad_url(frame.url)  # exclude most common ad network tracker frame URLs
		]

	@time_execution_async('--build_dom_tree')
	async def _build_dom_tree(
		self,
		highlight_elements: bool,
		focus_element: int,
		viewport_expansion: int,
	) -> tuple[DOMElementNode, SelectorMap]:
		if await self.page.evaluate('1+1') != 2:
			raise ValueError('The page cannot evaluate javascript code properly')

		if is_new_tab_page(self.page.url) or self.page.url.startswith('chrome://'):
			# short-circuit if the page is a new empty tab or chrome:// page for speed, no need to inject buildDomTree.js
			return (
				DOMElementNode(
					tag_name='body',
					xpath='',
					attributes={},
					children=[],
					is_visible=False,
					parent=None,
				),
				{},
			)

		# NOTE: We execute JS code in the browser to extract important DOM information.
		#       The returned hash map contains information about the DOM tree and the
		#       relationship between the DOM elements.
		debug_mode = self.logger.getEffectiveLevel() == logging.DEBUG
		args = {
			'doHighlightElements': highlight_elements,
			'focusHighlightIndex': focus_element,
			'viewportExpansion': viewport_expansion,
			'debugMode': debug_mode,
		}

		try:
			self.logger.debug(f'🔧 Starting JavaScript DOM analysis for {self.page.url[:50]}...')
			eval_page: dict = await self.page.evaluate(self.js_code, args)
			self.logger.debug('✅ JavaScript DOM analysis completed')
		except Exception as e:
			self.logger.error('Error evaluating JavaScript: %s', e)
			raise

		# Only log performance metrics in debug mode
		if debug_mode and 'perfMetrics' in eval_page:
			perf = eval_page['perfMetrics']

			# Get key metrics for summary
			total_nodes = perf.get('nodeMetrics', {}).get('totalNodes', 0)
			# processed_nodes = perf.get('nodeMetrics', {}).get('processedNodes', 0)

			# Count interactive elements from the DOM map
			interactive_count = 0
			if 'map' in eval_page:
				for node_data in eval_page['map'].values():
					if isinstance(node_data, dict) and node_data.get('isInteractive'):
						interactive_count += 1

			# Create concise summary
			url_short = self.page.url[:50] + '...' if len(self.page.url) > 50 else self.page.url
			self.logger.debug(
				'🔎 Ran buildDOMTree.js interactive element detection on: %s interactive=%d/%d\n',
				url_short,
				interactive_count,
				total_nodes,
				# processed_nodes,
			)

		self.logger.debug('🔄 Starting Python DOM tree construction...')
		result = await self._construct_dom_tree(eval_page)
		self.logger.debug('✅ Python DOM tree construction completed')
		return result

	@time_execution_async('--construct_dom_tree')
	async def _construct_dom_tree(
		self,
		eval_page: dict,
	) -> tuple[DOMElementNode, SelectorMap]:
		js_node_map = eval_page['map']
		js_root_id = eval_page['rootId']

		selector_map = {}
		node_map = {}

		for id, node_data in js_node_map.items():
			node, children_ids = self._parse_node(node_data)
			if node is None:
				continue

			node_map[id] = node

			if isinstance(node, DOMElementNode) and node.highlight_index is not None:
				selector_map[node.highlight_index] = node

			# NOTE: We know that we are building the tree bottom up
			#       and all children are already processed.
			if isinstance(node, DOMElementNode):
				for child_id in children_ids:
					if child_id not in node_map:
						continue

					child_node = node_map[child_id]

					child_node.parent = node
					node.children.append(child_node)

		html_to_dict = node_map[str(js_root_id)]

		del node_map
		del js_node_map
		del js_root_id

		if html_to_dict is None or not isinstance(html_to_dict, DOMElementNode):
			raise ValueError('Failed to parse HTML to dictionary')

		return html_to_dict, selector_map

	def _parse_node(
		self,
		node_data: dict,
	) -> tuple[DOMBaseNode | None, list[int]]:
		if not node_data:
			return None, []

		# Process text nodes immediately
		if node_data.get('type') == 'TEXT_NODE':
			text_node = DOMTextNode(
				text=node_data['text'],
				is_visible=node_data['isVisible'],
				parent=None,
			)
			return text_node, []

		# Process coordinates if they exist for element nodes

		viewport_info = None

		if 'viewport' in node_data:
			viewport_info = ViewportInfo(
				width=node_data['viewport']['width'],
				height=node_data['viewport']['height'],
			)

		element_node = DOMElementNode(
			tag_name=node_data['tagName'],
			xpath=node_data['xpath'],
			attributes=node_data.get('attributes', {}),
			children=[],
			is_visible=node_data.get('isVisible', False),
			is_interactive=node_data.get('isInteractive', False),
			is_top_element=node_data.get('isTopElement', False),
			is_in_viewport=node_data.get('isInViewport', False),
			highlight_index=node_data.get('highlightIndex'),
			shadow_root=node_data.get('shadowRoot', False),
			parent=None,
			viewport_info=viewport_info,
		)

		children_ids = node_data.get('children', [])

		return element_node, children_ids

from dataclasses import dataclass
from functools import cached_property
from typing import Optional
from browser_use.dom.history_tree_processor.view import CoordinateSet
from browser_use.dom.history_tree_processor.view import HashedDomElement
from browser_use.dom.history_tree_processor.view import ViewportInfo
from browser_use.dom.utils import cap_text_length
from browser_use.utils import time_execution_sync
from views import DOMElementNode
from browser_use.dom.history_tree_processor.service import HistoryTreeProcessor

# From dom/views.py
class DOMBaseNode:
	is_visible: bool
	# Use None as default and set parent later to avoid circular reference issues
	parent: Optional['DOMElementNode']

	def __json__(self) -> dict:
		raise NotImplementedError('DOMBaseNode is an abstract class')

# From dom/views.py
class DOMTextNode(DOMBaseNode):
	text: str
	type: str = 'TEXT_NODE'

	def has_parent_with_highlight_index(self) -> bool:
		current = self.parent
		while current is not None:
			# stop if the element has a highlight index (will be handled separately)
			if current.highlight_index is not None:
				return True

			current = current.parent
		return False

	def is_parent_in_viewport(self) -> bool:
		if self.parent is None:
			return False
		return self.parent.is_in_viewport

	def is_parent_top_element(self) -> bool:
		if self.parent is None:
			return False
		return self.parent.is_top_element

	def __json__(self) -> dict:
		return {
			'text': self.text,
			'type': self.type,
		}

# From dom/views.py
class DOMElementNode(DOMBaseNode):
	"""
	xpath: the xpath of the element from the last root node (shadow root or iframe OR document if no shadow root or iframe).
	To properly reference the element we need to recursively switch the root node until we find the element (work you way up the tree with `.parent`)
	"""

	tag_name: str
	xpath: str
	attributes: dict[str, str]
	children: list[DOMBaseNode]
	is_interactive: bool = False
	is_top_element: bool = False
	is_in_viewport: bool = False
	shadow_root: bool = False
	highlight_index: int | None = None
	viewport_coordinates: CoordinateSet | None = None
	page_coordinates: CoordinateSet | None = None
	viewport_info: ViewportInfo | None = None

	"""
	### State injected by the browser context.

	The idea is that the clickable elements are sometimes persistent from the previous page -> tells the model which objects are new/_how_ the state has changed
	"""
	is_new: bool | None = None

	def __json__(self) -> dict:
		return {
			'tag_name': self.tag_name,
			'xpath': self.xpath,
			'attributes': self.attributes,
			'is_visible': self.is_visible,
			'is_interactive': self.is_interactive,
			'is_top_element': self.is_top_element,
			'is_in_viewport': self.is_in_viewport,
			'shadow_root': self.shadow_root,
			'highlight_index': self.highlight_index,
			'viewport_coordinates': self.viewport_coordinates,
			'page_coordinates': self.page_coordinates,
			'children': [child.__json__() for child in self.children],
		}

	def __repr__(self) -> str:
		tag_str = f'<{self.tag_name}'

		# Add attributes
		for key, value in self.attributes.items():
			tag_str += f' {key}="{value}"'
		tag_str += '>'

		# Add extra info
		extras = []
		if self.is_interactive:
			extras.append('interactive')
		if self.is_top_element:
			extras.append('top')
		if self.shadow_root:
			extras.append('shadow-root')
		if self.highlight_index is not None:
			extras.append(f'highlight:{self.highlight_index}')
		if self.is_in_viewport:
			extras.append('in-viewport')

		if extras:
			tag_str += f' [{", ".join(extras)}]'

		return tag_str

	@cached_property
	def hash(self) -> HashedDomElement:
		from browser_use.dom.history_tree_processor.service import (
			HistoryTreeProcessor,
		)

		return HistoryTreeProcessor._hash_dom_element(self)

	def get_all_text_till_next_clickable_element(self, max_depth: int = -1) -> str:
		text_parts = []

		def collect_text(node: DOMBaseNode, current_depth: int) -> None:
			if max_depth != -1 and current_depth > max_depth:
				return

			# Skip this branch if we hit a highlighted element (except for the current node)
			if isinstance(node, DOMElementNode) and node != self and node.highlight_index is not None:
				return

			if isinstance(node, DOMTextNode):
				text_parts.append(node.text)
			elif isinstance(node, DOMElementNode):
				for child in node.children:
					collect_text(child, current_depth + 1)

		collect_text(self, 0)
		return '\n'.join(text_parts).strip()

	@time_execution_sync('--clickable_elements_to_string')
	def clickable_elements_to_string(self, include_attributes: list[str] | None = None) -> str:
		"""Convert the processed DOM content to HTML."""
		formatted_text = []

		if not include_attributes:
			include_attributes = DEFAULT_INCLUDE_ATTRIBUTES

		def process_node(node: DOMBaseNode, depth: int) -> None:
			next_depth = int(depth)
			depth_str = depth * '\t'

			if isinstance(node, DOMElementNode):
				# Add element with highlight_index
				if node.highlight_index is not None:
					next_depth += 1

					text = node.get_all_text_till_next_clickable_element()
					attributes_html_str = None
					if include_attributes:
						attributes_to_include = {
							key: str(value).strip()
							for key, value in node.attributes.items()
							if key in include_attributes and str(value).strip() != ''
						}

						# If value of any of the attributes is the same as ANY other value attribute only include the one that appears first in include_attributes
						# WARNING: heavy vibes, but it seems good enough for saving tokens (it kicks in hard when it's long text)

						# Pre-compute ordered keys that exist in both lists (faster than repeated lookups)
						ordered_keys = [key for key in include_attributes if key in attributes_to_include]

						if len(ordered_keys) > 1:  # Only process if we have multiple attributes
							keys_to_remove = set()  # Use set for O(1) lookups
							seen_values = {}  # value -> first_key_with_this_value

							for key in ordered_keys:
								value = attributes_to_include[key]
								if len(value) > 5:  # to not remove false, true, etc
									if value in seen_values:
										# This value was already seen with an earlier key, so remove this key
										keys_to_remove.add(key)
									else:
										# First time seeing this value, record it
										seen_values[value] = key

							# Remove duplicate keys (no need to check existence since we know they exist)
							for key in keys_to_remove:
								del attributes_to_include[key]

						# Easy LLM optimizations
						# if tag == role attribute, don't include it
						if node.tag_name == attributes_to_include.get('role'):
							del attributes_to_include['role']

						# Remove attributes that duplicate the node's text content
						attrs_to_remove_if_text_matches = ['aria-label', 'placeholder', 'title']
						for attr in attrs_to_remove_if_text_matches:
							if (
								attributes_to_include.get(attr)
								and attributes_to_include.get(attr, '').strip().lower() == text.strip().lower()
							):
								del attributes_to_include[attr]

						if attributes_to_include.items():
							# Format as key1='value1' key2='value2'
							attributes_html_str = ' '.join(
								f'{key}={cap_text_length(value, 15)}' for key, value in attributes_to_include.items()
							)

					# Build the line
					if node.is_new:
						highlight_indicator = f'*[{node.highlight_index}]'

					else:
						highlight_indicator = f'[{node.highlight_index}]'

					line = f'{depth_str}{highlight_indicator}<{node.tag_name}'

					if attributes_html_str:
						line += f' {attributes_html_str}'

					if text:
						# Add space before >text only if there were NO attributes added before
						text = text.strip()
						if not attributes_html_str:
							line += ' '
						line += f'>{text}'

					# Add space before /> only if neither attributes NOR text were added
					elif not attributes_html_str:
						line += ' '

					# makes sense to have if the website has lots of text -> so the LLM knows which things are part of the same clickable element and which are not
					line += ' />'  # 1 token
					formatted_text.append(line)

				# Process children regardless
				for child in node.children:
					process_node(child, next_depth)

			elif isinstance(node, DOMTextNode):
				# Add text only if it doesn't have a highlighted parent
				if node.has_parent_with_highlight_index():
					return

				if node.parent and node.parent.is_visible and node.parent.is_top_element:
					formatted_text.append(f'{depth_str}{node.text}')

		process_node(self, 0)
		return '\n'.join(formatted_text)

# From dom/views.py
class DOMState:
	element_tree: DOMElementNode
	selector_map: SelectorMap

# From dom/views.py
def has_parent_with_highlight_index(self) -> bool:
		current = self.parent
		while current is not None:
			# stop if the element has a highlight index (will be handled separately)
			if current.highlight_index is not None:
				return True

			current = current.parent
		return False

# From dom/views.py
def is_parent_in_viewport(self) -> bool:
		if self.parent is None:
			return False
		return self.parent.is_in_viewport

# From dom/views.py
def is_parent_top_element(self) -> bool:
		if self.parent is None:
			return False
		return self.parent.is_top_element

# From dom/views.py
def hash(self) -> HashedDomElement:
		from browser_use.dom.history_tree_processor.service import (
			HistoryTreeProcessor,
		)

		return HistoryTreeProcessor._hash_dom_element(self)

# From dom/views.py
def get_all_text_till_next_clickable_element(self, max_depth: int = -1) -> str:
		text_parts = []

		def collect_text(node: DOMBaseNode, current_depth: int) -> None:
			if max_depth != -1 and current_depth > max_depth:
				return

			# Skip this branch if we hit a highlighted element (except for the current node)
			if isinstance(node, DOMElementNode) and node != self and node.highlight_index is not None:
				return

			if isinstance(node, DOMTextNode):
				text_parts.append(node.text)
			elif isinstance(node, DOMElementNode):
				for child in node.children:
					collect_text(child, current_depth + 1)

		collect_text(self, 0)
		return '\n'.join(text_parts).strip()

# From dom/views.py
def clickable_elements_to_string(self, include_attributes: list[str] | None = None) -> str:
		"""Convert the processed DOM content to HTML."""
		formatted_text = []

		if not include_attributes:
			include_attributes = DEFAULT_INCLUDE_ATTRIBUTES

		def process_node(node: DOMBaseNode, depth: int) -> None:
			next_depth = int(depth)
			depth_str = depth * '\t'

			if isinstance(node, DOMElementNode):
				# Add element with highlight_index
				if node.highlight_index is not None:
					next_depth += 1

					text = node.get_all_text_till_next_clickable_element()
					attributes_html_str = None
					if include_attributes:
						attributes_to_include = {
							key: str(value).strip()
							for key, value in node.attributes.items()
							if key in include_attributes and str(value).strip() != ''
						}

						# If value of any of the attributes is the same as ANY other value attribute only include the one that appears first in include_attributes
						# WARNING: heavy vibes, but it seems good enough for saving tokens (it kicks in hard when it's long text)

						# Pre-compute ordered keys that exist in both lists (faster than repeated lookups)
						ordered_keys = [key for key in include_attributes if key in attributes_to_include]

						if len(ordered_keys) > 1:  # Only process if we have multiple attributes
							keys_to_remove = set()  # Use set for O(1) lookups
							seen_values = {}  # value -> first_key_with_this_value

							for key in ordered_keys:
								value = attributes_to_include[key]
								if len(value) > 5:  # to not remove false, true, etc
									if value in seen_values:
										# This value was already seen with an earlier key, so remove this key
										keys_to_remove.add(key)
									else:
										# First time seeing this value, record it
										seen_values[value] = key

							# Remove duplicate keys (no need to check existence since we know they exist)
							for key in keys_to_remove:
								del attributes_to_include[key]

						# Easy LLM optimizations
						# if tag == role attribute, don't include it
						if node.tag_name == attributes_to_include.get('role'):
							del attributes_to_include['role']

						# Remove attributes that duplicate the node's text content
						attrs_to_remove_if_text_matches = ['aria-label', 'placeholder', 'title']
						for attr in attrs_to_remove_if_text_matches:
							if (
								attributes_to_include.get(attr)
								and attributes_to_include.get(attr, '').strip().lower() == text.strip().lower()
							):
								del attributes_to_include[attr]

						if attributes_to_include.items():
							# Format as key1='value1' key2='value2'
							attributes_html_str = ' '.join(
								f'{key}={cap_text_length(value, 15)}' for key, value in attributes_to_include.items()
							)

					# Build the line
					if node.is_new:
						highlight_indicator = f'*[{node.highlight_index}]'

					else:
						highlight_indicator = f'[{node.highlight_index}]'

					line = f'{depth_str}{highlight_indicator}<{node.tag_name}'

					if attributes_html_str:
						line += f' {attributes_html_str}'

					if text:
						# Add space before >text only if there were NO attributes added before
						text = text.strip()
						if not attributes_html_str:
							line += ' '
						line += f'>{text}'

					# Add space before /> only if neither attributes NOR text were added
					elif not attributes_html_str:
						line += ' '

					# makes sense to have if the website has lots of text -> so the LLM knows which things are part of the same clickable element and which are not
					line += ' />'  # 1 token
					formatted_text.append(line)

				# Process children regardless
				for child in node.children:
					process_node(child, next_depth)

			elif isinstance(node, DOMTextNode):
				# Add text only if it doesn't have a highlighted parent
				if node.has_parent_with_highlight_index():
					return

				if node.parent and node.parent.is_visible and node.parent.is_top_element:
					formatted_text.append(f'{depth_str}{node.text}')

		process_node(self, 0)
		return '\n'.join(formatted_text)

# From dom/views.py
def collect_text(node: DOMBaseNode, current_depth: int) -> None:
			if max_depth != -1 and current_depth > max_depth:
				return

			# Skip this branch if we hit a highlighted element (except for the current node)
			if isinstance(node, DOMElementNode) and node != self and node.highlight_index is not None:
				return

			if isinstance(node, DOMTextNode):
				text_parts.append(node.text)
			elif isinstance(node, DOMElementNode):
				for child in node.children:
					collect_text(child, current_depth + 1)

# From dom/views.py
def process_node(node: DOMBaseNode, depth: int) -> None:
			next_depth = int(depth)
			depth_str = depth * '\t'

			if isinstance(node, DOMElementNode):
				# Add element with highlight_index
				if node.highlight_index is not None:
					next_depth += 1

					text = node.get_all_text_till_next_clickable_element()
					attributes_html_str = None
					if include_attributes:
						attributes_to_include = {
							key: str(value).strip()
							for key, value in node.attributes.items()
							if key in include_attributes and str(value).strip() != ''
						}

						# If value of any of the attributes is the same as ANY other value attribute only include the one that appears first in include_attributes
						# WARNING: heavy vibes, but it seems good enough for saving tokens (it kicks in hard when it's long text)

						# Pre-compute ordered keys that exist in both lists (faster than repeated lookups)
						ordered_keys = [key for key in include_attributes if key in attributes_to_include]

						if len(ordered_keys) > 1:  # Only process if we have multiple attributes
							keys_to_remove = set()  # Use set for O(1) lookups
							seen_values = {}  # value -> first_key_with_this_value

							for key in ordered_keys:
								value = attributes_to_include[key]
								if len(value) > 5:  # to not remove false, true, etc
									if value in seen_values:
										# This value was already seen with an earlier key, so remove this key
										keys_to_remove.add(key)
									else:
										# First time seeing this value, record it
										seen_values[value] = key

							# Remove duplicate keys (no need to check existence since we know they exist)
							for key in keys_to_remove:
								del attributes_to_include[key]

						# Easy LLM optimizations
						# if tag == role attribute, don't include it
						if node.tag_name == attributes_to_include.get('role'):
							del attributes_to_include['role']

						# Remove attributes that duplicate the node's text content
						attrs_to_remove_if_text_matches = ['aria-label', 'placeholder', 'title']
						for attr in attrs_to_remove_if_text_matches:
							if (
								attributes_to_include.get(attr)
								and attributes_to_include.get(attr, '').strip().lower() == text.strip().lower()
							):
								del attributes_to_include[attr]

						if attributes_to_include.items():
							# Format as key1='value1' key2='value2'
							attributes_html_str = ' '.join(
								f'{key}={cap_text_length(value, 15)}' for key, value in attributes_to_include.items()
							)

					# Build the line
					if node.is_new:
						highlight_indicator = f'*[{node.highlight_index}]'

					else:
						highlight_indicator = f'[{node.highlight_index}]'

					line = f'{depth_str}{highlight_indicator}<{node.tag_name}'

					if attributes_html_str:
						line += f' {attributes_html_str}'

					if text:
						# Add space before >text only if there were NO attributes added before
						text = text.strip()
						if not attributes_html_str:
							line += ' '
						line += f'>{text}'

					# Add space before /> only if neither attributes NOR text were added
					elif not attributes_html_str:
						line += ' '

					# makes sense to have if the website has lots of text -> so the LLM knows which things are part of the same clickable element and which are not
					line += ' />'  # 1 token
					formatted_text.append(line)

				# Process children regardless
				for child in node.children:
					process_node(child, next_depth)

			elif isinstance(node, DOMTextNode):
				# Add text only if it doesn't have a highlighted parent
				if node.has_parent_with_highlight_index():
					return

				if node.parent and node.parent.is_visible and node.parent.is_top_element:
					formatted_text.append(f'{depth_str}{node.text}')

import enum
from typing import Generic
from bubus.helpers import retry
from browser_use.agent.views import ActionModel
from browser_use.browser.views import BrowserError
from browser_use.controller.views import ClickElementAction
from browser_use.controller.views import CloseTabAction
from browser_use.controller.views import DoneAction
from browser_use.controller.views import GoToUrlAction
from browser_use.controller.views import InputTextAction
from browser_use.controller.views import NoParamsAction
from browser_use.controller.views import ScrollAction
from browser_use.controller.views import SearchGoogleAction
from browser_use.controller.views import SendKeysAction
from browser_use.controller.views import StructuredOutputAction
from browser_use.controller.views import SwitchTabAction
from browser_use.controller.views import UploadFileAction
from browser_use.llm.messages import UserMessage
from lmnr import Laminar
from functools import partial
import markdownify
from contextlib import nullcontext

# From controller/service.py
class Controller(Generic[Context]):
	def __init__(
		self,
		exclude_actions: list[str] = [],
		output_model: type[T] | None = None,
		display_files_in_done_text: bool = True,
	):
		self.registry = Registry[Context](exclude_actions)
		self.display_files_in_done_text = display_files_in_done_text

		"""Register all default browser actions"""

		self._register_done_action(output_model)

		# Basic Navigation Actions
		@self.registry.action(
			'Search the query in Google, the query should be a search query like humans search in Google, concrete and not vague or super long.',
			param_model=SearchGoogleAction,
		)
		async def search_google(params: SearchGoogleAction, browser_session: BrowserSession):
			search_url = f'https://www.google.com/search?q={params.query}&udm=14'

			page = await browser_session.get_current_page()
			if page.url.strip('/') == 'https://www.google.com':
				# SECURITY FIX: Use browser_session.navigate_to() instead of direct page.goto()
				# This ensures URL validation against allowed_domains is performed
				await browser_session.navigate_to(search_url)
			else:
				# create_new_tab already includes proper URL validation
				page = await browser_session.create_new_tab(search_url)

			msg = f'🔍  Searched for "{params.query}" in Google'
			logger.info(msg)
			return ActionResult(
				extracted_content=msg, include_in_memory=True, long_term_memory=f"Searched Google for '{params.query}'"
			)

		@self.registry.action(
			'Navigate to URL, set new_tab=True to open in new tab, False to navigate in current tab', param_model=GoToUrlAction
		)
		async def go_to_url(params: GoToUrlAction, browser_session: BrowserSession):
			try:
				if params.new_tab:
					# Open in new tab (logic from open_tab function)
					page = await browser_session.create_new_tab(params.url)
					tab_idx = browser_session.tabs.index(page)
					memory = f'Opened new tab with URL {params.url}'
					msg = f'🔗  Opened new tab #{tab_idx} with url {params.url}'
					logger.info(msg)
					return ActionResult(extracted_content=msg, include_in_memory=True, long_term_memory=memory)
				else:
					# Navigate in current tab (original logic)
					# SECURITY FIX: Use browser_session.navigate_to() instead of direct page.goto()
					# This ensures URL validation against allowed_domains is performed
					await browser_session.navigate_to(params.url)
					memory = f'Navigated to {params.url}'
					msg = f'🔗 {memory}'
					logger.info(msg)
					return ActionResult(extracted_content=msg, include_in_memory=True, long_term_memory=memory)
			except Exception as e:
				error_msg = str(e)
				# Check for network-related errors
				if any(
					err in error_msg
					for err in [
						'ERR_NAME_NOT_RESOLVED',
						'ERR_INTERNET_DISCONNECTED',
						'ERR_CONNECTION_REFUSED',
						'ERR_TIMED_OUT',
						'net::',
					]
				):
					site_unavailable_msg = f'Site unavailable: {params.url} - {error_msg}'
					logger.warning(site_unavailable_msg)
					raise BrowserError(site_unavailable_msg)
				else:
					# Re-raise non-network errors (including URLNotAllowedError for unauthorized domains)
					raise

		@self.registry.action('Go back', param_model=NoParamsAction)
		async def go_back(_: NoParamsAction, browser_session: BrowserSession):
			await browser_session.go_back()
			msg = '🔙  Navigated back'
			logger.info(msg)
			return ActionResult(extracted_content=msg)

		@self.registry.action(
			'Wait for x seconds default 3 (max 10 seconds). This can be used to wait until the page is fully loaded.'
		)
		async def wait(seconds: int = 3):
			# Cap wait time at maximum 10 seconds
			# Reduce the wait time by 3 seconds to account for the llm call which takes at least 3 seconds
			# So if the model decides to wait for 5 seconds, the llm call took at least 3 seconds, so we only need to wait for 2 seconds
			actual_seconds = min(max(seconds - 3, 0), 10)
			msg = f'🕒  Waiting for {actual_seconds + 3} seconds'
			logger.info(msg)
			await asyncio.sleep(actual_seconds)
			return ActionResult(extracted_content=msg)

		# Element Interaction Actions

		@self.registry.action('Click element by index', param_model=ClickElementAction)
		async def click_element_by_index(params: ClickElementAction, browser_session: BrowserSession):
			element_node = await browser_session.get_dom_element_by_index(params.index)
			if element_node is None:
				raise Exception(f'Element index {params.index} does not exist - retry or use alternative actions')

			initial_pages = len(browser_session.tabs)

			# if element has file uploader then dont click
			# Check if element is actually a file input (not just contains file-related keywords)
			if browser_session.is_file_input(element_node):
				msg = f'Index {params.index} - has an element which opens file upload dialog. To upload files please use a specific function to upload files '
				logger.info(msg)
				return ActionResult(extracted_content=msg, include_in_memory=True, success=False, long_term_memory=msg)

			msg = None

			try:
				download_path = await browser_session._click_element_node(element_node)
				if download_path:
					emoji = '💾'
					msg = f'Downloaded file to {download_path}'
				else:
					emoji = '🖱️'
					msg = f'Clicked button with index {params.index}: {element_node.get_all_text_till_next_clickable_element(max_depth=2)}'

				logger.info(f'{emoji} {msg}')
				logger.debug(f'Element xpath: {element_node.xpath}')
				if len(browser_session.tabs) > initial_pages:
					new_tab_msg = 'New tab opened - switching to it'
					msg += f' - {new_tab_msg}'
					emoji = '🔗'
					logger.info(f'{emoji} {new_tab_msg}')
					await browser_session.switch_to_tab(-1)
				return ActionResult(extracted_content=msg, include_in_memory=True, long_term_memory=msg)
			except Exception as e:
				error_msg = str(e)
				raise BrowserError(error_msg)

		@self.registry.action(
			'Click and input text into a input interactive element',
			param_model=InputTextAction,
		)
		async def input_text(params: InputTextAction, browser_session: BrowserSession, has_sensitive_data: bool = False):
			element_node = await browser_session.get_dom_element_by_index(params.index)
			if element_node is None:
				raise Exception(f'Element index {params.index} does not exist - retry or use alternative actions')

			try:
				await browser_session._input_text_element_node(element_node, params.text)
			except Exception:
				msg = f'Failed to input text into element {params.index}.'
				raise BrowserError(msg)

			if not has_sensitive_data:
				msg = f'⌨️  Input {params.text} into index {params.index}'
			else:
				msg = f'⌨️  Input sensitive data into index {params.index}'
			logger.info(msg)
			logger.debug(f'Element xpath: {element_node.xpath}')
			return ActionResult(
				extracted_content=msg,
				include_in_memory=True,
				long_term_memory=f"Input '{params.text}' into element {params.index}.",
			)

		@self.registry.action('Upload file to interactive element with file path', param_model=UploadFileAction)
		async def upload_file(params: UploadFileAction, browser_session: BrowserSession, available_file_paths: list[str]):
			if params.path not in available_file_paths:
				raise BrowserError(f'File path {params.path} is not available')

			if not os.path.exists(params.path):
				raise BrowserError(f'File {params.path} does not exist')

			file_upload_dom_el = await browser_session.find_file_upload_element_by_index(
				params.index, max_height=3, max_descendant_depth=3
			)

			if file_upload_dom_el is None:
				msg = f'No file upload element found at index {params.index}'
				logger.info(msg)
				raise BrowserError(msg)

			file_upload_el = await browser_session.get_locate_element(file_upload_dom_el)

			if file_upload_el is None:
				msg = f'No file upload element found at index {params.index}'
				logger.info(msg)
				raise BrowserError(msg)

			try:
				await file_upload_el.set_input_files(params.path)
				msg = f'📁 Successfully uploaded file to index {params.index}'
				logger.info(msg)
				return ActionResult(
					extracted_content=msg,
					include_in_memory=True,
					long_term_memory=f'Uploaded file {params.path} to element {params.index}',
				)
			except Exception as e:
				msg = f'Failed to upload file to index {params.index}: {str(e)}'
				logger.info(msg)
				raise BrowserError(msg)

		# Tab Management Actions

		@self.registry.action('Switch tab', param_model=SwitchTabAction)
		async def switch_tab(params: SwitchTabAction, browser_session: BrowserSession):
			await browser_session.switch_to_tab(params.page_id)
			page = await browser_session.get_current_page()
			try:
				await page.wait_for_load_state(state='domcontentloaded', timeout=5_000)
				# page was already loaded when we first navigated, this is additional to wait for onfocus/onblur animations/ajax to settle
			except Exception as e:
				pass
			msg = f'🔄  Switched to tab #{params.page_id} with url {page.url}'
			logger.info(msg)
			return ActionResult(
				extracted_content=msg, include_in_memory=True, long_term_memory=f'Switched to tab {params.page_id}'
			)

		@self.registry.action('Close an existing tab', param_model=CloseTabAction)
		async def close_tab(params: CloseTabAction, browser_session: BrowserSession):
			await browser_session.switch_to_tab(params.page_id)
			page = await browser_session.get_current_page()
			url = page.url
			await page.close()
			new_page = await browser_session.get_current_page()
			new_page_idx = browser_session.tabs.index(new_page)
			msg = f'❌  Closed tab #{params.page_id} with {url}, now focused on tab #{new_page_idx} with url {new_page.url}'
			logger.info(msg)
			return ActionResult(
				extracted_content=msg,
				include_in_memory=True,
				long_term_memory=f'Closed tab {params.page_id} with url {url}, now focused on tab {new_page_idx} with url {new_page.url}.',
			)

		# Content Actions

		@self.registry.action(
			"""Extract structured, semantic data (e.g. product description, price, all information about XYZ) from the current webpage based on a textual query.
This tool takes the entire markdown of the page and extracts the query from it. 
Set extract_links=True ONLY if your query requires extracting links/URLs from the page. 
Only use this for specific queries for information retrieval from the page. Don't use this to get interactive elements - the tool does not see HTML elements, only the markdown.
""",
		)
		async def extract_structured_data(
			query: str,
			extract_links: bool,
			page: Page,
			page_extraction_llm: BaseChatModel,
			file_system: FileSystem,
		):
			from functools import partial

			import markdownify

			strip = []

			if not extract_links:
				strip = ['a', 'img']

			# Run markdownify in a thread pool to avoid blocking the event loop
			loop = asyncio.get_event_loop()

			# Aggressive timeout for page content
			try:
				page_html_result = await asyncio.wait_for(page.content(), timeout=10.0)  # 5 second aggressive timeout
			except TimeoutError:
				raise RuntimeError('Page content extraction timed out after 5 seconds')
			except Exception as e:
				raise RuntimeError(f"Couldn't extract page content: {e}")

			page_html = page_html_result

			markdownify_func = partial(markdownify.markdownify, strip=strip)

			try:
				content = await asyncio.wait_for(
					loop.run_in_executor(None, markdownify_func, page_html), timeout=5.0
				)  # 5 second aggressive timeout
			except Exception as e:
				logger.warning(f'Markdownify failed: {type(e).__name__}')
				raise RuntimeError(f'Could not convert html to markdown: {type(e).__name__}')

			# manually append iframe text into the content so it's readable by the LLM (includes cross-origin iframes)
			for iframe in page.frames:
				try:
					await iframe.wait_for_load_state(timeout=1000)  # 1 second aggressive timeout for iframe load
				except Exception:
					pass

				if iframe.url != page.url and not iframe.url.startswith('data:') and not iframe.url.startswith('about:'):
					content += f'\n\nIFRAME {iframe.url}:\n'
					# Run markdownify in a thread pool for iframe content as well
					try:
						# Aggressive timeouts for iframe content
						iframe_html = await asyncio.wait_for(iframe.content(), timeout=2.0)  # 2 second aggressive timeout
						iframe_markdown = await asyncio.wait_for(
							loop.run_in_executor(None, markdownify_func, iframe_html),
							timeout=2.0,  # 2 second aggressive timeout for iframe markdownify
						)
					except Exception:
						iframe_markdown = ''  # Skip failed iframes
					content += iframe_markdown
			# replace multiple sequential \n with a single \n
			content = re.sub(r'\n+', '\n', content)

			# limit to 30000 characters - remove text in the middle (≈15000 tokens)
			max_chars = 30000
			if len(content) > max_chars:
				logger.info(f'Content is too long, removing middle {len(content) - max_chars} characters')
				content = (
					content[: max_chars // 2]
					+ '\n... left out the middle because it was too long ...\n'
					+ content[-max_chars // 2 :]
				)

			prompt = """You convert websites into structured information. Extract information from this webpage based on the query. Focus only on content relevant to the query. If 
1. The query is vague
2. Does not make sense for the page
3. Some/all of the information is not available

Explain the content of the page and that the requested information is not available in the page. Respond in JSON format.\nQuery: {query}\n Website:\n{page}"""
			try:
				formatted_prompt = prompt.format(query=query, page=content)
				# Aggressive timeout for LLM call
				response = await asyncio.wait_for(
					page_extraction_llm.ainvoke([UserMessage(content=formatted_prompt)]),
					timeout=120.0,  # 120 second aggressive timeout for LLM call
				)

				extracted_content = f'Page Link: {page.url}\nQuery: {query}\nExtracted Content:\n{response.completion}'

				# if content is small include it to memory
				MAX_MEMORY_SIZE = 600
				if len(extracted_content) < MAX_MEMORY_SIZE:
					memory = extracted_content
					include_extracted_content_only_once = False
				else:
					# find lines until MAX_MEMORY_SIZE
					lines = extracted_content.splitlines()
					display = ''
					display_lines_count = 0
					for line in lines:
						if len(display) + len(line) < MAX_MEMORY_SIZE:
							display += line + '\n'
							display_lines_count += 1
						else:
							break
					save_result = await file_system.save_extracted_content(extracted_content)
					memory = f'Extracted content from {page.url}\n<query>{query}\n</query>\n<extracted_content>\n{display}{len(lines) - display_lines_count} more lines...\n</extracted_content>\n<file_system>{save_result}</file_system>'
					include_extracted_content_only_once = True
				logger.info(f'📄 {memory}')
				return ActionResult(
					extracted_content=extracted_content,
					include_extracted_content_only_once=include_extracted_content_only_once,
					long_term_memory=memory,
				)
			except TimeoutError:
				error_msg = f'LLM call timed out for query: {query}'
				logger.warning(error_msg)
				raise RuntimeError(error_msg)
			except Exception as e:
				logger.debug(f'Error extracting content: {e}')
				msg = f'📄  Extracted from page\n: {content}\n'
				logger.info(msg)
				raise RuntimeError(str(e))

		@self.registry.action(
			'Scroll the page by specified number of pages (set down=True to scroll down, down=False to scroll up, num_pages=number of pages to scroll like 0.5 for half page, 1.0 for one page, etc.). Optional index parameter to scroll within a specific element or its scroll container (works well for dropdowns and custom UI components).',
			param_model=ScrollAction,
		)
		async def scroll(params: ScrollAction, browser_session: BrowserSession):
			"""
			(a) If index is provided, find scrollable containers in the element hierarchy and scroll directly.
			(b) If no index or no container found, use browser._scroll_container for container-aware scrolling.
			(c) If that JavaScript throws, fall back to window.scrollBy().
			"""
			page = await browser_session.get_current_page()

			# Helper function to get window height with retry decorator
			@retry(wait=1, retries=3, timeout=5)
			async def get_window_height():
				return await page.evaluate('() => window.innerHeight')

			# Get window height with retries
			try:
				window_height = await get_window_height()
			except Exception as e:
				raise RuntimeError(f'Scroll failed due to an error: {e}')
			window_height = window_height or 0

			# Determine scroll amount based on num_pages
			scroll_amount = int(window_height * params.num_pages)
			pages_scrolled = params.num_pages

			# Set direction based on down parameter
			dy = scroll_amount if params.down else -scroll_amount

			# Initialize result message components
			direction = 'down' if params.down else 'up'
			scroll_target = 'the page'

			# Element-specific scrolling if index is provided
			if params.index is not None:
				try:
					element_node = await browser_session.get_dom_element_by_index(params.index)
					if element_node is None:
						raise Exception(f'Element index {params.index} does not exist - retry or use alternative actions')

					# Try direct container scrolling (no events that might close dropdowns)
					container_scroll_js = """
					(params) => {
						const { dy, elementXPath } = params;
						
						// Get the target element by XPath
						const targetElement = document.evaluate(elementXPath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
						if (!targetElement) {
							return { success: false, reason: 'Element not found by XPath' };
						}

						console.log('[SCROLL DEBUG] Starting direct container scroll for element:', targetElement.tagName);
						
						// Try to find scrollable containers in the hierarchy (starting from element itself)
						let currentElement = targetElement;
						let scrollSuccess = false;
						let scrolledElement = null;
						let scrollDelta = 0;
						let attempts = 0;
						
						// Check up to 10 elements in hierarchy (including the target element itself)
						while (currentElement && attempts < 10) {
							const computedStyle = window.getComputedStyle(currentElement);
							const hasScrollableY = /(auto|scroll|overlay)/.test(computedStyle.overflowY);
							const canScrollVertically = currentElement.scrollHeight > currentElement.clientHeight;
							
							console.log('[SCROLL DEBUG] Checking element:', currentElement.tagName, 
								'hasScrollableY:', hasScrollableY, 
								'canScrollVertically:', canScrollVertically,
								'scrollHeight:', currentElement.scrollHeight,
								'clientHeight:', currentElement.clientHeight);
							
							if (hasScrollableY && canScrollVertically) {
								const beforeScroll = currentElement.scrollTop;
								const maxScroll = currentElement.scrollHeight - currentElement.clientHeight;
								
								// Calculate scroll amount (1/3 of provided dy for gentler scrolling)
								let scrollAmount = dy / 3;
								
								// Ensure we don't scroll beyond bounds
								if (scrollAmount > 0) {
									scrollAmount = Math.min(scrollAmount, maxScroll - beforeScroll);
								} else {
									scrollAmount = Math.max(scrollAmount, -beforeScroll);
								}
								
								// Try direct scrollTop manipulation (most reliable)
								currentElement.scrollTop = beforeScroll + scrollAmount;
								
								const afterScroll = currentElement.scrollTop;
								const actualScrollDelta = afterScroll - beforeScroll;
								
								console.log('[SCROLL DEBUG] Scroll attempt:', currentElement.tagName, 
									'before:', beforeScroll, 'after:', afterScroll, 'delta:', actualScrollDelta);
								
								if (Math.abs(actualScrollDelta) > 0.5) {
									scrollSuccess = true;
									scrolledElement = currentElement;
									scrollDelta = actualScrollDelta;
									console.log('[SCROLL DEBUG] Successfully scrolled container:', currentElement.tagName, 'delta:', actualScrollDelta);
									break;
								}
							}
							
							// Move to parent (but don't go beyond body for dropdown case)
							if (currentElement === document.body || currentElement === document.documentElement) {
								break;
							}
							currentElement = currentElement.parentElement;
							attempts++;
						}
						
						if (scrollSuccess) {
							// Successfully scrolled a container
							return { 
								success: true, 
								method: 'direct_container_scroll',
								containerType: 'element', 
								containerTag: scrolledElement.tagName.toLowerCase(),
								containerClass: scrolledElement.className || '',
								containerId: scrolledElement.id || '',
								scrollDelta: scrollDelta
							};
						} else {
							// No container found or could scroll
							console.log('[SCROLL DEBUG] No scrollable container found for element');
							return { 
								success: false, 
								reason: 'No scrollable container found',
								needsPageScroll: true
							};
						}
					}
					"""

					# Pass parameters as a single object
					scroll_params = {'dy': dy, 'elementXPath': element_node.xpath}
					result = await page.evaluate(container_scroll_js, scroll_params)

					if result['success']:
						if result['containerType'] == 'element':
							container_info = f'{result["containerTag"]}'
							if result['containerId']:
								container_info += f'#{result["containerId"]}'
							elif result['containerClass']:
								container_info += f'.{result["containerClass"].split()[0]}'
							scroll_target = f"element {params.index}'s scroll container ({container_info})"
							# Don't do additional page scrolling since we successfully scrolled the container
						else:
							scroll_target = f'the page (fallback from element {params.index})'
					else:
						# Container scroll failed, need page-level scrolling
						logger.debug(f'Container scroll failed for element {params.index}: {result.get("reason", "Unknown")}')
						scroll_target = f'the page (no container found for element {params.index})'
						# This will trigger page-level scrolling below

				except Exception as e:
					logger.debug(f'Element-specific scrolling failed for index {params.index}: {e}')
					scroll_target = f'the page (fallback from element {params.index})'
					# Fall through to page-level scrolling

			# Page-level scrolling (default or fallback)
			if (
				scroll_target == 'the page'
				or 'fallback' in scroll_target
				or 'no container found' in scroll_target
				or 'mouse wheel failed' in scroll_target
			):
				logger.debug(f'🔄 Performing page-level scrolling. Reason: {scroll_target}')
				try:
					await browser_session._scroll_container(cast(int, dy))
				except Exception as e:
					# Hard fallback: always works on root scroller
					await page.evaluate('(y) => window.scrollBy(0, y)', dy)
					logger.debug('Smart scroll failed; used window.scrollBy fallback', exc_info=e)

			# Create descriptive message
			if pages_scrolled == 1.0:
				long_term_memory = f'Scrolled {direction} {scroll_target} by one page'
			else:
				long_term_memory = f'Scrolled {direction} {scroll_target} by {pages_scrolled} pages'

			msg = f'🔍 {long_term_memory}'

			logger.info(msg)
			return ActionResult(extracted_content=msg, include_in_memory=True, long_term_memory=long_term_memory)

		@self.registry.action(
			'Send strings of special keys to use Playwright page.keyboard.press - examples include Escape, Backspace, Insert, PageDown, Delete, Enter, or Shortcuts such as `Control+o`, `Control+Shift+T`',
			param_model=SendKeysAction,
		)
		async def send_keys(params: SendKeysAction, page: Page):
			try:
				await page.keyboard.press(params.keys)
			except Exception as e:
				if 'Unknown key' in str(e):
					# loop over the keys and try to send each one
					for key in params.keys:
						try:
							await page.keyboard.press(key)
						except Exception as e:
							logger.debug(f'Error sending key {key}: {str(e)}')
							raise e
				else:
					raise e
			msg = f'⌨️  Sent keys: {params.keys}'
			logger.info(msg)
			return ActionResult(extracted_content=msg, include_in_memory=True, long_term_memory=f'Sent keys: {params.keys}')

		@self.registry.action(
			description='Scroll to a text in the current page',
		)
		async def scroll_to_text(text: str, page: Page):  # type: ignore
			try:
				# Try different locator strategies
				locators = [
					page.get_by_text(text, exact=False),
					page.locator(f'text={text}'),
					page.locator(f"//*[contains(text(), '{text}')]"),
				]

				for locator in locators:
					try:
						if await locator.count() == 0:
							continue

						element = locator.first
						is_visible = await element.is_visible()
						bbox = await element.bounding_box()

						if is_visible and bbox is not None and bbox['width'] > 0 and bbox['height'] > 0:
							await element.scroll_into_view_if_needed()
							await asyncio.sleep(0.5)  # Wait for scroll to complete
							msg = f'🔍  Scrolled to text: {text}'
							logger.info(msg)
							return ActionResult(
								extracted_content=msg, include_in_memory=True, long_term_memory=f'Scrolled to text: {text}'
							)

					except Exception as e:
						logger.debug(f'Locator attempt failed: {str(e)}')
						continue

				msg = f"Text '{text}' not found or not visible on page"
				logger.info(msg)
				return ActionResult(
					extracted_content=msg,
					include_in_memory=True,
					long_term_memory=f"Tried scrolling to text '{text}' but it was not found",
				)

			except Exception as e:
				msg = f"Failed to scroll to text '{text}': {str(e)}"
				logger.error(msg)
				raise BrowserError(msg)

		# File System Actions
		@self.registry.action(
			'Write or append content to file_name in file system. Allowed extensions are .md, .txt, .json, .csv, .pdf. For .pdf files, write the content in markdown format and it will automatically be converted to a properly formatted PDF document.'
		)
		async def write_file(
			file_name: str,
			content: str,
			file_system: FileSystem,
			append: bool = False,
			trailing_newline: bool = True,
			leading_newline: bool = False,
		):
			if trailing_newline:
				content += '\n'
			if leading_newline:
				content = '\n' + content
			if append:
				result = await file_system.append_file(file_name, content)
			else:
				result = await file_system.write_file(file_name, content)
			logger.info(f'💾 {result}')
			return ActionResult(extracted_content=result, include_in_memory=True, long_term_memory=result)

		@self.registry.action(
			'Replace old_str with new_str in file_name. old_str must exactly match the string to replace in original text. Recommended tool to mark completed items in todo.md or change specific contents in a file.'
		)
		async def replace_file_str(file_name: str, old_str: str, new_str: str, file_system: FileSystem):
			result = await file_system.replace_file_str(file_name, old_str, new_str)
			logger.info(f'💾 {result}')
			return ActionResult(extracted_content=result, include_in_memory=True, long_term_memory=result)

		@self.registry.action('Read file_name from file system')
		async def read_file(file_name: str, available_file_paths: list[str], file_system: FileSystem):
			if available_file_paths and file_name in available_file_paths:
				result = await file_system.read_file(file_name, external_file=True)
			else:
				result = await file_system.read_file(file_name)

			MAX_MEMORY_SIZE = 1000
			if len(result) > MAX_MEMORY_SIZE:
				lines = result.splitlines()
				display = ''
				lines_count = 0
				for line in lines:
					if len(display) + len(line) < MAX_MEMORY_SIZE:
						display += line + '\n'
						lines_count += 1
					else:
						break
				remaining_lines = len(lines) - lines_count
				memory = f'{display}{remaining_lines} more lines...' if remaining_lines > 0 else display
			else:
				memory = result
			logger.info(f'💾 {memory}')
			return ActionResult(
				extracted_content=result,
				include_in_memory=True,
				long_term_memory=memory,
				include_extracted_content_only_once=True,
			)

		@self.registry.action(
			description='Get all options from a native dropdown or ARIA menu',
		)
		async def get_dropdown_options(index: int, browser_session: BrowserSession) -> ActionResult:
			"""Get all options from a native dropdown or ARIA menu"""
			page = await browser_session.get_current_page()
			dom_element = await browser_session.get_dom_element_by_index(index)
			if dom_element is None:
				raise Exception(f'Element index {index} does not exist - retry or use alternative actions')

			try:
				# Frame-aware approach since we know it works
				all_options = []
				frame_index = 0

				for frame in page.frames:
					try:
						# First check if it's a native select element
						options = await frame.evaluate(
							"""
							(xpath) => {
								const element = document.evaluate(xpath, document, null,
									XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
								if (!element) return null;

								// Check if it's a native select element
								if (element.tagName.toLowerCase() === 'select') {
									return {
										type: 'select',
										options: Array.from(element.options).map(opt => ({
											text: opt.text, //do not trim, because we are doing exact match in select_dropdown_option
											value: opt.value,
											index: opt.index
										})),
										id: element.id,
										name: element.name
									};
								}
								
								// Check if it's an ARIA menu
								if (element.getAttribute('role') === 'menu' || 
									element.getAttribute('role') === 'listbox' ||
									element.getAttribute('role') === 'combobox') {
									// Find all menu items
									const menuItems = element.querySelectorAll('[role="menuitem"], [role="option"]');
									const options = [];
									
									menuItems.forEach((item, idx) => {
										// Get the text content of the menu item
										const text = item.textContent.trim();
										if (text) {
											options.push({
												text: text,
												value: text, // For ARIA menus, use text as value
												index: idx
											});
										}
									});
									
									return {
										type: 'aria',
										options: options,
										id: element.id || '',
										name: element.getAttribute('aria-label') || ''
									};
								}
								
								return null;
							}
						""",
							dom_element.xpath,
						)

						if options:
							logger.debug(f'Found {options["type"]} dropdown in frame {frame_index}')
							logger.debug(f'Element ID: {options["id"]}, Name: {options["name"]}')

							formatted_options = []
							for opt in options['options']:
								# encoding ensures AI uses the exact string in select_dropdown_option
								encoded_text = json.dumps(opt['text'])
								formatted_options.append(f'{opt["index"]}: text={encoded_text}')

							all_options.extend(formatted_options)

					except Exception as frame_e:
						logger.debug(f'Frame {frame_index} evaluation failed: {str(frame_e)}')

					frame_index += 1

				if all_options:
					msg = '\n'.join(all_options)
					msg += '\nUse the exact text string in select_dropdown_option'
					logger.info(msg)
					return ActionResult(
						extracted_content=msg,
						include_in_memory=True,
						long_term_memory=f'Found dropdown options for index {index}.',
						include_extracted_content_only_once=True,
					)
				else:
					msg = 'No options found in any frame for dropdown'
					logger.info(msg)
					return ActionResult(
						extracted_content=msg, include_in_memory=True, long_term_memory='No dropdown options found'
					)

			except Exception as e:
				logger.error(f'Failed to get dropdown options: {str(e)}')
				msg = f'Error getting options: {str(e)}'
				logger.info(msg)
				return ActionResult(extracted_content=msg, include_in_memory=True)

		@self.registry.action(
			description='Select dropdown option or ARIA menu item for interactive element index by the text of the option you want to select',
		)
		async def select_dropdown_option(
			index: int,
			text: str,
			browser_session: BrowserSession,
		) -> ActionResult:
			"""Select dropdown option or ARIA menu item by the text of the option you want to select"""
			page = await browser_session.get_current_page()
			dom_element = await browser_session.get_dom_element_by_index(index)
			if dom_element is None:
				raise Exception(f'Element index {index} does not exist - retry or use alternative actions')

			logger.debug(f"Attempting to select '{text}' using xpath: {dom_element.xpath}")
			logger.debug(f'Element attributes: {dom_element.attributes}')
			logger.debug(f'Element tag: {dom_element.tag_name}')

			xpath = '//' + dom_element.xpath

			try:
				frame_index = 0
				for frame in page.frames:
					try:
						logger.debug(f'Trying frame {frame_index} URL: {frame.url}')

						# First check what type of element we're dealing with
						element_info_js = """
							(xpath) => {
								try {
									const element = document.evaluate(xpath, document, null,
										XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
									if (!element) return null;
									
									const tagName = element.tagName.toLowerCase();
									const role = element.getAttribute('role');
									
									// Check if it's a native select
									if (tagName === 'select') {
										return {
											type: 'select',
											found: true,
											id: element.id,
											name: element.name,
											tagName: element.tagName,
											optionCount: element.options.length,
											currentValue: element.value,
											availableOptions: Array.from(element.options).map(o => o.text.trim())
										};
									}
									
									// Check if it's an ARIA menu or similar
									if (role === 'menu' || role === 'listbox' || role === 'combobox') {
										const menuItems = element.querySelectorAll('[role="menuitem"], [role="option"]');
										return {
											type: 'aria',
											found: true,
											id: element.id || '',
											role: role,
											tagName: element.tagName,
											itemCount: menuItems.length,
											availableOptions: Array.from(menuItems).map(item => item.textContent.trim())
										};
									}
									
									return {
										error: `Element is neither a select nor an ARIA menu (tag: ${tagName}, role: ${role})`,
										found: false
									};
								} catch (e) {
									return {error: e.toString(), found: false};
								}
							}
						"""

						element_info = await frame.evaluate(element_info_js, dom_element.xpath)

						if element_info and element_info.get('found'):
							logger.debug(f'Found {element_info.get("type")} element in frame {frame_index}: {element_info}')

							if element_info.get('type') == 'select':
								# Handle native select element
								# "label" because we are selecting by text
								# nth(0) to disable error thrown by strict mode
								# timeout=1000 because we are already waiting for all network events
								selected_option_values = (
									await frame.locator('//' + dom_element.xpath).nth(0).select_option(label=text, timeout=1000)
								)

								msg = f'selected option {text} with value {selected_option_values}'
								logger.info(msg + f' in frame {frame_index}')

								return ActionResult(
									extracted_content=msg, include_in_memory=True, long_term_memory=f"Selected option '{text}'"
								)

							elif element_info.get('type') == 'aria':
								# Handle ARIA menu
								click_aria_item_js = """
									(params) => {
										const { xpath, targetText } = params;
										try {
											const element = document.evaluate(xpath, document, null,
												XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
											if (!element) return {success: false, error: 'Element not found'};
											
											// Find all menu items
											const menuItems = element.querySelectorAll('[role="menuitem"], [role="option"]');
											
											for (const item of menuItems) {
												const itemText = item.textContent.trim();
												if (itemText === targetText) {
													// Simulate click on the menu item
													item.click();
													
													// Also try dispatching a click event in case the click handler needs it
													const clickEvent = new MouseEvent('click', {
														view: window,
														bubbles: true,
														cancelable: true
													});
													item.dispatchEvent(clickEvent);
													
													return {
														success: true,
														message: `Clicked menu item: ${targetText}`
													};
												}
											}
											
											return {
												success: false,
												error: `Menu item with text '${targetText}' not found`
											};
										} catch (e) {
											return {success: false, error: e.toString()};
										}
									}
								"""

								result = await frame.evaluate(
									click_aria_item_js, {'xpath': dom_element.xpath, 'targetText': text}
								)

								if result.get('success'):
									msg = result.get('message', f'Selected ARIA menu item: {text}')
									logger.info(msg + f' in frame {frame_index}')
									return ActionResult(
										extracted_content=msg,
										include_in_memory=True,
										long_term_memory=f"Selected menu item '{text}'",
									)
								else:
									logger.error(f'Failed to select ARIA menu item: {result.get("error")}')
									continue

						elif element_info:
							logger.error(f'Frame {frame_index} error: {element_info.get("error")}')
							continue

					except Exception as frame_e:
						logger.error(f'Frame {frame_index} attempt failed: {str(frame_e)}')
						logger.error(f'Frame type: {type(frame)}')
						logger.error(f'Frame URL: {frame.url}')

					frame_index += 1

				msg = f"Could not select option '{text}' in any frame"
				logger.info(msg)
				return ActionResult(extracted_content=msg, include_in_memory=True, long_term_memory=msg)

			except Exception as e:
				msg = f'Selection failed: {str(e)}'
				logger.error(msg)
				raise BrowserError(msg)

		@self.registry.action('Google Sheets: Get the contents of the entire sheet', domains=['https://docs.google.com'])
		async def read_sheet_contents(page: Page):
			# select all cells
			await page.keyboard.press('Enter')
			await page.keyboard.press('Escape')
			await page.keyboard.press('ControlOrMeta+A')
			await page.keyboard.press('ControlOrMeta+C')

			extracted_tsv = await page.evaluate('() => navigator.clipboard.readText()')
			return ActionResult(
				extracted_content=extracted_tsv,
				include_in_memory=True,
				long_term_memory='Retrieved sheet contents',
				include_extracted_content_only_once=True,
			)

		@self.registry.action('Google Sheets: Get the contents of a cell or range of cells', domains=['https://docs.google.com'])
		async def read_cell_contents(cell_or_range: str, browser_session: BrowserSession):
			page = await browser_session.get_current_page()

			await select_cell_or_range(cell_or_range=cell_or_range, page=page)

			await page.keyboard.press('ControlOrMeta+C')
			await asyncio.sleep(0.1)
			extracted_tsv = await page.evaluate('() => navigator.clipboard.readText()')
			return ActionResult(
				extracted_content=extracted_tsv,
				include_in_memory=True,
				long_term_memory=f'Retrieved contents from {cell_or_range}',
				include_extracted_content_only_once=True,
			)

		@self.registry.action(
			'Google Sheets: Update the content of a cell or range of cells', domains=['https://docs.google.com']
		)
		async def update_cell_contents(cell_or_range: str, new_contents_tsv: str, browser_session: BrowserSession):
			page = await browser_session.get_current_page()

			await select_cell_or_range(cell_or_range=cell_or_range, page=page)

			# simulate paste event from clipboard with TSV content
			await page.evaluate(f"""
				const clipboardData = new DataTransfer();
				clipboardData.setData('text/plain', `{new_contents_tsv}`);
				document.activeElement.dispatchEvent(new ClipboardEvent('paste', {{clipboardData}}));
			""")

			return ActionResult(
				extracted_content=f'Updated cells: {cell_or_range} = {new_contents_tsv}',
				include_in_memory=False,
				long_term_memory=f'Updated cells {cell_or_range} with {new_contents_tsv}',
			)

		@self.registry.action('Google Sheets: Clear whatever cells are currently selected', domains=['https://docs.google.com'])
		async def clear_cell_contents(cell_or_range: str, browser_session: BrowserSession):
			page = await browser_session.get_current_page()

			await select_cell_or_range(cell_or_range=cell_or_range, page=page)

			await page.keyboard.press('Backspace')
			return ActionResult(
				extracted_content=f'Cleared cells: {cell_or_range}',
				include_in_memory=False,
				long_term_memory=f'Cleared cells {cell_or_range}',
			)

		@self.registry.action('Google Sheets: Select a specific cell or range of cells', domains=['https://docs.google.com'])
		async def select_cell_or_range(cell_or_range: str, page: Page):
			await page.keyboard.press('Enter')  # make sure we dont delete current cell contents if we were last editing
			await page.keyboard.press('Escape')  # to clear current focus (otherwise select range popup is additive)
			await asyncio.sleep(0.1)
			await page.keyboard.press('Home')  # move cursor to the top left of the sheet first
			await page.keyboard.press('ArrowUp')
			await asyncio.sleep(0.1)
			await page.keyboard.press('Control+G')  # open the goto range popup
			await asyncio.sleep(0.2)
			await page.keyboard.type(cell_or_range, delay=0.05)
			await asyncio.sleep(0.2)
			await page.keyboard.press('Enter')
			await asyncio.sleep(0.2)
			await page.keyboard.press('Escape')  # to make sure the popup still closes in the case where the jump failed
			return ActionResult(
				extracted_content=f'Selected cells: {cell_or_range}',
				include_in_memory=False,
				long_term_memory=f'Selected cells {cell_or_range}',
			)

		@self.registry.action(
			'Google Sheets: Fallback method to type text into (only one) currently selected cell',
			domains=['https://docs.google.com'],
		)
		async def fallback_input_into_single_selected_cell(text: str, page: Page):
			await page.keyboard.type(text, delay=0.1)
			await page.keyboard.press('Enter')  # make sure to commit the input so it doesn't get overwritten by the next action
			await page.keyboard.press('ArrowUp')
			return ActionResult(
				extracted_content=f'Inputted text {text}',
				include_in_memory=False,
				long_term_memory=f"Inputted text '{text}' into cell",
			)

	# Custom done action for structured output
	def _register_done_action(self, output_model: type[T] | None, display_files_in_done_text: bool = True):
		if output_model is not None:
			self.display_files_in_done_text = display_files_in_done_text

			@self.registry.action(
				'Complete task - with return text and if the task is finished (success=True) or not yet completely finished (success=False), because last step is reached',
				param_model=StructuredOutputAction[output_model],
			)
			async def done(params: StructuredOutputAction):
				# Exclude success from the output JSON since it's an internal parameter
				output_dict = params.data.model_dump()

				# Enums are not serializable, convert to string
				for key, value in output_dict.items():
					if isinstance(value, enum.Enum):
						output_dict[key] = value.value

				return ActionResult(
					is_done=True,
					success=params.success,
					extracted_content=json.dumps(output_dict),
					long_term_memory=f'Task completed. Success Status: {params.success}',
				)

		else:

			@self.registry.action(
				'Complete task - provide a summary of results for the user. Set success=True if task completed successfully, false otherwise. Text should be your response to the user summarizing results. Include files you would like to display to the user in files_to_display.',
				param_model=DoneAction,
			)
			async def done(params: DoneAction, file_system: FileSystem):
				user_message = params.text

				len_text = len(params.text)
				len_max_memory = 100
				memory = f'Task completed: {params.success} - {params.text[:len_max_memory]}'
				if len_text > len_max_memory:
					memory += f' - {len_text - len_max_memory} more characters'

				attachments = []
				if params.files_to_display:
					if self.display_files_in_done_text:
						file_msg = ''
						for file_name in params.files_to_display:
							if file_name == 'todo.md':
								continue
							file_content = file_system.display_file(file_name)
							if file_content:
								file_msg += f'\n\n{file_name}:\n{file_content}'
								attachments.append(file_name)
						if file_msg:
							user_message += '\n\nAttachments:'
							user_message += file_msg
						else:
							logger.warning('Agent wanted to display files but none were found')
					else:
						for file_name in params.files_to_display:
							if file_name == 'todo.md':
								continue
							file_content = file_system.display_file(file_name)
							if file_content:
								attachments.append(file_name)

				attachments = [str(file_system.get_dir() / file_name) for file_name in attachments]

				return ActionResult(
					is_done=True,
					success=params.success,
					extracted_content=user_message,
					long_term_memory=memory,
					attachments=attachments,
				)

	def use_structured_output_action(self, output_model: type[T]):
		self._register_done_action(output_model)

	# Register ---------------------------------------------------------------

	def action(self, description: str, **kwargs):
		"""Decorator for registering custom actions

		@param description: Describe the LLM what the function does (better description == better function calling)
		"""
		return self.registry.action(description, **kwargs)

	# Act --------------------------------------------------------------------
	@observe_debug(ignore_input=True, ignore_output=True, name='act')
	@time_execution_sync('--act')
	async def act(
		self,
		action: ActionModel,
		browser_session: BrowserSession,
		#
		page_extraction_llm: BaseChatModel | None = None,
		sensitive_data: dict[str, str | dict[str, str]] | None = None,
		available_file_paths: list[str] | None = None,
		file_system: FileSystem | None = None,
		#
		context: Context | None = None,
	) -> ActionResult:
		"""Execute an action"""

		for action_name, params in action.model_dump(exclude_unset=True).items():
			if params is not None:
				# Use Laminar span if available, otherwise use no-op context manager
				if Laminar is not None:
					span_context = Laminar.start_as_current_span(
						name=action_name,
						input={
							'action': action_name,
							'params': params,
						},
						span_type='TOOL',
					)
				else:
					# No-op context manager when lmnr is not available
					from contextlib import nullcontext

					span_context = nullcontext()

				with span_context:
					try:
						result = await self.registry.execute_action(
							action_name=action_name,
							params=params,
							browser_session=browser_session,
							page_extraction_llm=page_extraction_llm,
							file_system=file_system,
							sensitive_data=sensitive_data,
							available_file_paths=available_file_paths,
							context=context,
						)
					except Exception as e:
						result = ActionResult(error=str(e))

					if Laminar is not None:
						Laminar.set_span_output(result)

				if isinstance(result, str):
					return ActionResult(extracted_content=result)
				elif isinstance(result, ActionResult):
					return result
				elif result is None:
					return ActionResult()
				else:
					raise ValueError(f'Invalid action result type: {type(result)} of {result}')
		return ActionResult()

# From controller/service.py
def use_structured_output_action(self, output_model: type[T]):
		self._register_done_action(output_model)

# From controller/service.py
def action(self, description: str, **kwargs):
		"""Decorator for registering custom actions

		@param description: Describe the LLM what the function does (better description == better function calling)
		"""
		return self.registry.action(description, **kwargs)


# From controller/views.py
class SearchGoogleAction(BaseModel):
	query: str

# From controller/views.py
class GoToUrlAction(BaseModel):
	url: str
	new_tab: bool = False

# From controller/views.py
class ClickElementAction(BaseModel):
	index: int

# From controller/views.py
class InputTextAction(BaseModel):
	index: int
	text: str

# From controller/views.py
class DoneAction(BaseModel):
	text: str
	success: bool
	files_to_display: list[str] | None = []

# From controller/views.py
class StructuredOutputAction(BaseModel, Generic[T]):
	success: bool = True
	data: T

# From controller/views.py
class SwitchTabAction(BaseModel):
	page_id: int

# From controller/views.py
class CloseTabAction(BaseModel):
	page_id: int

# From controller/views.py
class ScrollAction(BaseModel):
	down: bool  # True to scroll down, False to scroll up
	num_pages: float  # Number of pages to scroll (0.5 = half page, 1.0 = one page, etc.)
	index: int | None = None

# From controller/views.py
class SendKeysAction(BaseModel):
	keys: str

# From controller/views.py
class UploadFileAction(BaseModel):
	index: int
	path: str

# From controller/views.py
class ExtractPageContentAction(BaseModel):
	value: str

# From controller/views.py
class NoParamsAction(BaseModel):
	"""
	Accepts absolutely anything in the incoming data
	and discards it, so the final parsed model is empty.
	"""

	model_config = ConfigDict(extra='ignore')

from bubus import BaseEvent
from browser_use.sync.auth import TEMP_USER_ID
from browser_use.sync.auth import DeviceAuthClient

# From sync/service.py
class CloudSync:
	"""Service for syncing events to the Browser Use cloud"""

	def __init__(self, base_url: str | None = None, enable_auth: bool = True):
		# Backend API URL for all API requests - can be passed directly or defaults to env var
		self.base_url = base_url or CONFIG.BROWSER_USE_CLOUD_API_URL
		self.enable_auth = enable_auth
		self.auth_client = DeviceAuthClient(base_url=self.base_url) if enable_auth else None
		self.pending_events: list[BaseEvent] = []
		self.auth_task = None
		self.session_id: str | None = None

	async def handle_event(self, event: BaseEvent) -> None:
		"""Handle an event by sending it to the cloud"""
		try:
			# Extract session ID from CreateAgentSessionEvent
			if event.event_type == 'CreateAgentSessionEvent' and hasattr(event, 'id'):
				self.session_id = str(event.id)  # type: ignore

			# Start authentication flow on first step (after first LLM response)
			if event.event_type == 'CreateAgentStepEvent' and 'step' in dict(event):
				step = dict(event)['step']
				# logger.debug(f'Got CreateAgentStepEvent with step={step}')
				# Trigger on the first step (step=2 because n_steps is incremented before actions)
				if step == 2 and self.enable_auth and self.auth_client:
					if not hasattr(self, 'auth_task') or self.auth_task is None:
						# Start auth in background
						if self.session_id:
							# logger.info('Triggering auth on first step event')
							# Always run auth to show the cloud URL, even if already authenticated
							self.auth_task = asyncio.create_task(self._background_auth(agent_session_id=self.session_id))
						else:
							logger.warning('Cannot start auth - session_id not set yet')

			# Send event to cloud
			await self._send_event(event)

		except Exception as e:
			logger.error(f'Failed to handle {event.event_type} event: {type(e).__name__}: {e}', exc_info=True)

	async def _send_event(self, event: BaseEvent) -> None:
		"""Send event to cloud API"""
		try:
			headers = {}

			# override user_id on event with auth client user_id if available
			if self.auth_client:
				event.user_id = str(self.auth_client.user_id)  # type: ignore
			else:
				event.user_id = TEMP_USER_ID  # type: ignore

			# Add auth headers if available
			if self.auth_client:
				headers.update(self.auth_client.get_headers())

			# Send event (batch format with direct BaseEvent serialization)
			async with httpx.AsyncClient() as client:
				# Serialize event and add device_id to all events
				event_data = event.model_dump(mode='json')
				if self.auth_client and self.auth_client.device_id:
					event_data['device_id'] = self.auth_client.device_id

				response = await client.post(
					f'{self.base_url.rstrip("/")}/api/v1/events',
					json={'events': [event_data]},
					headers=headers,
					timeout=10.0,
				)

				if response.status_code == 401 and self.auth_client and not self.auth_client.is_authenticated:
					# Store event for retry after auth
					self.pending_events.append(event)
				elif response.status_code >= 400:
					# Log error but don't raise - we want to fail silently
					logger.debug(
						f'Failed to send sync event: POST {response.request.url} {response.status_code} - {response.text}'
					)
		except httpx.TimeoutException:
			logger.warning(f'⚠️ Event send timed out after 10 seconds: {event}')
		except httpx.ConnectError as e:
			# logger.warning(f'⚠️ Failed to connect to cloud service at {self.base_url}: {e}')
			pass
		except httpx.HTTPError as e:
			logger.warning(f'⚠️ HTTP error sending event {event}: {type(e).__name__}: {e}')
		except Exception as e:
			logger.warning(f'⚠️ Unexpected error sending event {event}: {type(e).__name__}: {e}')

	async def _background_auth(self, agent_session_id: str) -> None:
		"""Run authentication in background or show cloud URL if already authenticated"""
		assert self.auth_client, 'enable_auth=True must be set before calling CloudSync_background_auth()'
		assert self.session_id, 'session_id must be set before calling CloudSync._background_auth() can fire'
		try:
			# If already authenticated, just show the cloud URL
			if self.auth_client.is_authenticated:
				# Use frontend URL for user-facing links
				frontend_url = CONFIG.BROWSER_USE_CLOUD_UI_URL or self.base_url.replace('//api.', '//cloud.')
				session_url = f'{frontend_url.rstrip("/")}/agent/{agent_session_id}'
				terminal_width, _terminal_height = shutil.get_terminal_size((80, 20))
				logger.info('─' * max(terminal_width - 40, 20) + '\n')
				logger.info('🌐  View the details of this run in Browser Use Cloud:')
				logger.info(f'    👉  {session_url}')
				logger.info('─' * max(terminal_width - 40, 20) + '\n\n')
				return

			# Otherwise run full authentication
			success = await self.auth_client.authenticate(
				agent_session_id=agent_session_id,
				show_instructions=True,
			)

			if success:
				# Resend any pending events
				await self._resend_pending_events()

				# Update WAL events with real user_id
				# await self._update_wal_user_ids(agent_session_id)

		except Exception as e:
			logger.debug(f'Cloud sync authentication failed: {e}')

	async def _resend_pending_events(self) -> None:
		"""Resend events that were queued during auth"""
		if not self.pending_events:
			return

		# Send all pending events
		for event in self.pending_events:
			try:
				await self._send_event(event)
			except Exception as e:
				logger.warning(f'Failed to resend pending event: {e}')

		self.pending_events.clear()

	# async def _update_wal_user_ids(self, session_id: str) -> None:
	# 	"""Update user IDs in WAL file after authentication"""
	# 	try:
	# 		assert self.auth_client, 'Cloud sync must be authenticated to update WAL user ID'

	# 		wal_path = CONFIG.BROWSER_USE_CONFIG_DIR / 'events' / f'{session_id}.jsonl'
	# 		if not await anyio.Path(wal_path).exists():
	# 			raise FileNotFoundError(
	# 				f'CloudSync failed to update saved event user_ids after auth: Agent EventBus WAL file not found: {wal_path}'
	# 			)

	# 		# Read all events
	# 		events = []
	# 		content = await anyio.Path(wal_path).read_text()
	# 		for line in content.splitlines():
	# 			if line.strip():
	# 				events.append(json.loads(line))

	# 		# Update user_id and device_id
	# 		user_id = self.auth_client.user_id
	# 		device_id = self.auth_client.device_id
	# 		for event in events:
	# 			if 'user_id' in event:
	# 				event['user_id'] = user_id
	# 			# Add device_id to all events
	# 			event['device_id'] = device_id

	# 		# Write back
	# 		updated_content = '\n'.join(json.dumps(event) for event in events) + '\n'
	# 		await anyio.Path(wal_path).write_text(updated_content)

	# 	except Exception as e:
	# 		logger.warning(f'Failed to update WAL user IDs: {e}')

	async def wait_for_auth(self) -> None:
		"""Wait for authentication to complete if in progress"""
		if self.auth_task and not self.auth_task.done():
			await self.auth_task

	async def authenticate(self, show_instructions: bool = True) -> bool:
		"""Authenticate with the cloud service"""
		if not self.auth_client:
			return False

		return await self.auth_client.authenticate(agent_session_id=self.session_id, show_instructions=show_instructions)

from uuid_extensions import uuid7str

# From sync/auth.py
class CloudAuthConfig(BaseModel):
	"""Configuration for cloud authentication"""

	api_token: str | None = None
	user_id: str | None = None
	authorized_at: datetime | None = None

	@classmethod
	def load_from_file(cls) -> 'CloudAuthConfig':
		"""Load auth config from local file"""

		config_path = CONFIG.BROWSER_USE_CONFIG_DIR / 'cloud_auth.json'
		if config_path.exists():
			try:
				with open(config_path) as f:
					data = json.load(f)
				return cls.model_validate(data)
			except Exception:
				# Return empty config if file is corrupted
				pass
		return cls()

	def save_to_file(self) -> None:
		"""Save auth config to local file"""

		CONFIG.BROWSER_USE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

		config_path = CONFIG.BROWSER_USE_CONFIG_DIR / 'cloud_auth.json'
		with open(config_path, 'w') as f:
			json.dump(self.model_dump(mode='json'), f, indent=2, default=str)

		# Set restrictive permissions (owner read/write only) for security
		try:
			os.chmod(config_path, 0o600)
		except Exception:
			# Some systems may not support chmod, continue anyway
			pass

# From sync/auth.py
class DeviceAuthClient:
	"""Client for OAuth2 device authorization flow"""

	def __init__(self, base_url: str | None = None, http_client: httpx.AsyncClient | None = None):
		# Backend API URL for OAuth requests - can be passed directly or defaults to env var
		self.base_url = base_url or CONFIG.BROWSER_USE_CLOUD_API_URL
		self.client_id = 'library'
		self.scope = 'read write'

		# If no client provided, we'll create one per request
		self.http_client = http_client

		# Temporary user ID for pre-auth events
		self.temp_user_id = TEMP_USER_ID

		# Get or create persistent device ID
		self.device_id = get_or_create_device_id()

		# Load existing auth if available
		self.auth_config = CloudAuthConfig.load_from_file()

	@property
	def is_authenticated(self) -> bool:
		"""Check if we have valid authentication"""
		return bool(self.auth_config.api_token and self.auth_config.user_id)

	@property
	def api_token(self) -> str | None:
		"""Get the current API token"""
		return self.auth_config.api_token

	@property
	def user_id(self) -> str:
		"""Get the current user ID (temporary or real)"""
		return self.auth_config.user_id or self.temp_user_id

	async def start_device_authorization(
		self,
		agent_session_id: str | None = None,
	) -> dict:
		"""
		Start the device authorization flow.
		Returns device authorization details including user code and verification URL.
		"""
		if self.http_client:
			response = await self.http_client.post(
				f'{self.base_url.rstrip("/")}/api/v1/oauth/device/authorize',
				data={
					'client_id': self.client_id,
					'scope': self.scope,
					'agent_session_id': agent_session_id,
					'device_id': self.device_id,
				},
			)
			response.raise_for_status()
			return response.json()
		else:
			async with httpx.AsyncClient() as client:
				response = await client.post(
					f'{self.base_url.rstrip("/")}/api/v1/oauth/device/authorize',
					data={
						'client_id': self.client_id,
						'scope': self.scope,
						'agent_session_id': agent_session_id,
						'device_id': self.device_id,
					},
				)
				response.raise_for_status()
				return response.json()

	async def poll_for_token(
		self,
		device_code: str,
		interval: float = 3.0,
		timeout: float = 1800.0,
	) -> dict | None:
		"""
		Poll for the access token.
		Returns token info when authorized, None if timeout.
		"""
		start_time = time.time()

		if self.http_client:
			# Use injected client for all requests
			while time.time() - start_time < timeout:
				try:
					response = await self.http_client.post(
						f'{self.base_url.rstrip("/")}/api/v1/oauth/device/token',
						data={
							'grant_type': 'urn:ietf:params:oauth:grant-type:device_code',
							'device_code': device_code,
							'client_id': self.client_id,
						},
					)

					if response.status_code == 200:
						data = response.json()

						# Check for pending authorization
						if data.get('error') == 'authorization_pending':
							await asyncio.sleep(interval)
							continue

						# Check for slow down
						if data.get('error') == 'slow_down':
							interval = data.get('interval', interval * 2)
							await asyncio.sleep(interval)
							continue

						# Check for other errors
						if 'error' in data:
							print(f'Error: {data.get("error_description", data["error"])}')
							return None

						# Success! We have a token
						if 'access_token' in data:
							return data

					elif response.status_code == 400:
						# Error response
						data = response.json()
						if data.get('error') not in ['authorization_pending', 'slow_down']:
							print(f'Error: {data.get("error_description", "Unknown error")}')
							return None

					else:
						print(f'Unexpected status code: {response.status_code}')
						return None

				except Exception as e:
					print(f'Error polling for token: {e}')

				await asyncio.sleep(interval)
		else:
			# Create a new client for polling
			async with httpx.AsyncClient() as client:
				while time.time() - start_time < timeout:
					try:
						response = await client.post(
							f'{self.base_url.rstrip("/")}/api/v1/oauth/device/token',
							data={
								'grant_type': 'urn:ietf:params:oauth:grant-type:device_code',
								'device_code': device_code,
								'client_id': self.client_id,
							},
						)

						if response.status_code == 200:
							data = response.json()

							# Check for pending authorization
							if data.get('error') == 'authorization_pending':
								await asyncio.sleep(interval)
								continue

							# Check for slow down
							if data.get('error') == 'slow_down':
								interval = data.get('interval', interval * 2)
								await asyncio.sleep(interval)
								continue

							# Check for other errors
							if 'error' in data:
								print(f'Error: {data.get("error_description", data["error"])}')
								return None

							# Success! We have a token
							if 'access_token' in data:
								return data

						elif response.status_code == 400:
							# Error response
							data = response.json()
							if data.get('error') not in ['authorization_pending', 'slow_down']:
								print(f'Error: {data.get("error_description", "Unknown error")}')
								return None

						else:
							print(f'Unexpected status code: {response.status_code}')
							return None

					except Exception as e:
						print(f'Error polling for token: {e}')

					await asyncio.sleep(interval)

		return None

	async def authenticate(
		self,
		agent_session_id: str | None = None,
		show_instructions: bool = True,
	) -> bool:
		"""
		Run the full authentication flow.
		Returns True if authentication successful.
		"""
		import logging

		logger = logging.getLogger(__name__)

		try:
			# Start device authorization
			device_auth = await self.start_device_authorization(agent_session_id)

			# Use frontend URL for user-facing links
			frontend_url = CONFIG.BROWSER_USE_CLOUD_UI_URL or self.base_url.replace('//api.', '//cloud.')

			# Replace backend URL with frontend URL in verification URIs
			verification_uri = device_auth['verification_uri'].replace(self.base_url, frontend_url)
			verification_uri_complete = device_auth['verification_uri_complete'].replace(self.base_url, frontend_url)

			terminal_width, _terminal_height = shutil.get_terminal_size((80, 20))
			if show_instructions:
				logger.info('─' * max(terminal_width - 40, 20))
				logger.info('🌐  View the details of this run in Browser Use Cloud:')
				logger.info(f'    👉  {verification_uri_complete}')
				logger.info('─' * max(terminal_width - 40, 20) + '\n')

			# Poll for token
			token_data = await self.poll_for_token(
				device_code=device_auth['device_code'],
				interval=device_auth.get('interval', 5),
			)

			if token_data and token_data.get('access_token'):
				# Save authentication
				self.auth_config.api_token = token_data['access_token']
				self.auth_config.user_id = token_data.get('user_id', self.temp_user_id)
				self.auth_config.authorized_at = datetime.now()
				self.auth_config.save_to_file()

				if show_instructions:
					logger.debug('✅  Authentication successful! Cloud sync is now enabled with your browser-use account.')

				return True

		except httpx.HTTPStatusError as e:
			# HTTP error with response
			if e.response.status_code == 404:
				logger.warning(
					'Cloud sync authentication endpoint not found (404). Check your BROWSER_USE_CLOUD_API_URL setting.'
				)
			else:
				logger.warning(f'Failed to authenticate with cloud service: HTTP {e.response.status_code} - {e.response.text}')
		except httpx.RequestError as e:
			# Connection/network errors
			# logger.warning(f'Failed to connect to cloud service: {type(e).__name__}: {e}')
			pass
		except Exception as e:
			# Other unexpected errors
			logger.warning(f'❌ Unexpected error during cloud sync authentication: {type(e).__name__}: {e}')

		if show_instructions:
			logger.debug(f'❌ Sync authentication failed or timed out with {CONFIG.BROWSER_USE_CLOUD_API_URL}')

		return False

	def get_headers(self) -> dict:
		"""Get headers for API requests"""
		if self.api_token:
			return {'Authorization': f'Bearer {self.api_token}'}
		return {}

	def clear_auth(self) -> None:
		"""Clear stored authentication"""
		self.auth_config = CloudAuthConfig()
		self.auth_config.save_to_file()

# From sync/auth.py
def get_or_create_device_id() -> str:
	"""Get or create a persistent device ID for this installation."""
	device_id_path = CONFIG.BROWSER_USE_CONFIG_DIR / 'device_id'

	# Try to read existing device ID
	if device_id_path.exists():
		try:
			device_id = device_id_path.read_text().strip()
			if device_id:  # Make sure it's not empty
				return device_id
		except Exception:
			# If we can't read it, we'll create a new one
			pass

	# Create new device ID
	device_id = uuid7str()

	# Ensure config directory exists
	CONFIG.BROWSER_USE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

	# Write device ID to file
	device_id_path.write_text(device_id)

	return device_id

# From sync/auth.py
def load_from_file(cls) -> 'CloudAuthConfig':
		"""Load auth config from local file"""

		config_path = CONFIG.BROWSER_USE_CONFIG_DIR / 'cloud_auth.json'
		if config_path.exists():
			try:
				with open(config_path) as f:
					data = json.load(f)
				return cls.model_validate(data)
			except Exception:
				# Return empty config if file is corrupted
				pass
		return cls()

# From sync/auth.py
def save_to_file(self) -> None:
		"""Save auth config to local file"""

		CONFIG.BROWSER_USE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

		config_path = CONFIG.BROWSER_USE_CONFIG_DIR / 'cloud_auth.json'
		with open(config_path, 'w') as f:
			json.dump(self.model_dump(mode='json'), f, indent=2, default=str)

		# Set restrictive permissions (owner read/write only) for security
		try:
			os.chmod(config_path, 0o600)
		except Exception:
			# Some systems may not support chmod, continue anyway
			pass

# From sync/auth.py
def is_authenticated(self) -> bool:
		"""Check if we have valid authentication"""
		return bool(self.auth_config.api_token and self.auth_config.user_id)

# From sync/auth.py
def api_token(self) -> str | None:
		"""Get the current API token"""
		return self.auth_config.api_token

# From sync/auth.py
def user_id(self) -> str:
		"""Get the current user ID (temporary or real)"""
		return self.auth_config.user_id or self.temp_user_id

# From sync/auth.py
def get_headers(self) -> dict:
		"""Get headers for API requests"""
		if self.api_token:
			return {'Authorization': f'Bearer {self.api_token}'}
		return {}

# From sync/auth.py
def clear_auth(self) -> None:
		"""Clear stored authentication"""
		self.auth_config = CloudAuthConfig()
		self.auth_config.save_to_file()

import importlib.resources
from browser_use.llm.messages import ContentPartImageParam
from browser_use.llm.messages import ContentPartTextParam
from browser_use.llm.messages import ImageURL
from browser_use.llm.messages import SystemMessage
from browser_use.agent.views import AgentStepInfo
from browser_use.browser.views import BrowserStateSummary

# From agent/prompts.py
class SystemPrompt:
	def __init__(
		self,
		action_description: str,
		max_actions_per_step: int = 10,
		override_system_message: str | None = None,
		extend_system_message: str | None = None,
		use_thinking: bool = True,
		flash_mode: bool = False,
	):
		self.default_action_description = action_description
		self.max_actions_per_step = max_actions_per_step
		self.use_thinking = use_thinking
		self.flash_mode = flash_mode
		prompt = ''
		if override_system_message:
			prompt = override_system_message
		else:
			self._load_prompt_template()
			prompt = self.prompt_template.format(max_actions=self.max_actions_per_step)

		if extend_system_message:
			prompt += f'\n{extend_system_message}'

		self.system_message = SystemMessage(content=prompt, cache=True)

	def _load_prompt_template(self) -> None:
		"""Load the prompt template from the markdown file."""
		try:
			# Choose the appropriate template based on flash_mode and use_thinking settings
			if self.flash_mode:
				template_filename = 'system_prompt_flash.md'
			elif self.use_thinking:
				template_filename = 'system_prompt.md'
			else:
				template_filename = 'system_prompt_no_thinking.md'

			# This works both in development and when installed as a package
			with importlib.resources.files('browser_use.agent').joinpath(template_filename).open('r', encoding='utf-8') as f:
				self.prompt_template = f.read()
		except Exception as e:
			raise RuntimeError(f'Failed to load system prompt template: {e}')

	def get_system_message(self) -> SystemMessage:
		"""
		Get the system prompt for the agent.

		Returns:
		    SystemMessage: Formatted system prompt
		"""
		return self.system_message

# From agent/prompts.py
class AgentMessagePrompt:
	vision_detail_level: Literal['auto', 'low', 'high']

	def __init__(
		self,
		browser_state_summary: 'BrowserStateSummary',
		file_system: 'FileSystem',
		agent_history_description: str | None = None,
		read_state_description: str | None = None,
		task: str | None = None,
		include_attributes: list[str] | None = None,
		step_info: Optional['AgentStepInfo'] = None,
		page_filtered_actions: str | None = None,
		max_clickable_elements_length: int = 40000,
		sensitive_data: str | None = None,
		available_file_paths: list[str] | None = None,
		screenshots: list[str] | None = None,
		vision_detail_level: Literal['auto', 'low', 'high'] = 'auto',
	):
		self.browser_state: 'BrowserStateSummary' = browser_state_summary
		self.file_system: 'FileSystem | None' = file_system
		self.agent_history_description: str | None = agent_history_description
		self.read_state_description: str | None = read_state_description
		self.task: str | None = task
		self.include_attributes = include_attributes
		self.step_info = step_info
		self.page_filtered_actions: str | None = page_filtered_actions
		self.max_clickable_elements_length: int = max_clickable_elements_length
		self.sensitive_data: str | None = sensitive_data
		self.available_file_paths: list[str] | None = available_file_paths
		self.screenshots = screenshots or []
		self.vision_detail_level = vision_detail_level
		assert self.browser_state

	@observe_debug(ignore_input=True, ignore_output=True, name='_get_browser_state_description')
	def _get_browser_state_description(self) -> str:
		elements_text = self.browser_state.element_tree.clickable_elements_to_string(include_attributes=self.include_attributes)

		if len(elements_text) > self.max_clickable_elements_length:
			elements_text = elements_text[: self.max_clickable_elements_length]
			truncated_text = f' (truncated to {self.max_clickable_elements_length} characters)'
		else:
			truncated_text = ''

		has_content_above = (self.browser_state.pixels_above or 0) > 0
		has_content_below = (self.browser_state.pixels_below or 0) > 0

		# Enhanced page information for the model
		page_info_text = ''
		if self.browser_state.page_info:
			pi = self.browser_state.page_info
			# Compute page statistics dynamically
			pages_above = pi.pixels_above / pi.viewport_height if pi.viewport_height > 0 else 0
			pages_below = pi.pixels_below / pi.viewport_height if pi.viewport_height > 0 else 0
			total_pages = pi.page_height / pi.viewport_height if pi.viewport_height > 0 else 0
			current_page_position = pi.scroll_y / max(pi.page_height - pi.viewport_height, 1)
			page_info_text = f'Page info: {pi.viewport_width}x{pi.viewport_height}px viewport, {pi.page_width}x{pi.page_height}px total page size, {pages_above:.1f} pages above, {pages_below:.1f} pages below, {total_pages:.1f} total pages, at {current_page_position:.0%} of page'

		if elements_text != '':
			if has_content_above:
				if self.browser_state.page_info:
					pi = self.browser_state.page_info
					pages_above = pi.pixels_above / pi.viewport_height if pi.viewport_height > 0 else 0
					elements_text = f'... {self.browser_state.pixels_above} pixels above ({pages_above:.1f} pages) - scroll to see more or extract structured data if you are looking for specific information ...\n{elements_text}'
				else:
					elements_text = f'... {self.browser_state.pixels_above} pixels above - scroll to see more or extract structured data if you are looking for specific information ...\n{elements_text}'
			else:
				elements_text = f'[Start of page]\n{elements_text}'
			if has_content_below:
				if self.browser_state.page_info:
					pi = self.browser_state.page_info
					pages_below = pi.pixels_below / pi.viewport_height if pi.viewport_height > 0 else 0
					elements_text = f'{elements_text}\n... {self.browser_state.pixels_below} pixels below ({pages_below:.1f} pages) - scroll to see more or extract structured data if you are looking for specific information ...'
				else:
					elements_text = f'{elements_text}\n... {self.browser_state.pixels_below} pixels below - scroll to see more or extract structured data if you are looking for specific information ...'
			else:
				elements_text = f'{elements_text}\n[End of page]'
		else:
			elements_text = 'empty page'

		tabs_text = ''
		current_tab_candidates = []

		# Find tabs that match both URL and title to identify current tab more reliably
		for tab in self.browser_state.tabs:
			if tab.url == self.browser_state.url and tab.title == self.browser_state.title:
				current_tab_candidates.append(tab.page_id)

		# If we have exactly one match, mark it as current
		# Otherwise, don't mark any tab as current to avoid confusion
		current_tab_id = current_tab_candidates[0] if len(current_tab_candidates) == 1 else None

		for tab in self.browser_state.tabs:
			tabs_text += f'Tab {tab.page_id}: {tab.url} - {tab.title[:30]}\n'

		current_tab_text = f'Current tab: {current_tab_id}' if current_tab_id is not None else ''

		# Check if current page is a PDF viewer and add appropriate message
		pdf_message = ''
		if self.browser_state.is_pdf_viewer:
			pdf_message = 'PDF viewer cannot be rendered. In this page, DO NOT use the extract_structured_data action as PDF content cannot be rendered. Use the read_file action on the downloaded PDF in available_file_paths to read the full content.\n\n'

		browser_state = f"""{current_tab_text}
Available tabs:
{tabs_text}
{page_info_text}
{pdf_message}Interactive elements from top layer of the current page inside the viewport{truncated_text}:
{elements_text}
"""
		return browser_state

	def _get_agent_state_description(self) -> str:
		if self.step_info:
			step_info_description = f'Step {self.step_info.step_number + 1} of {self.step_info.max_steps} max possible steps\n'
		else:
			step_info_description = ''
		time_str = datetime.now().strftime('%Y-%m-%d %H:%M')
		step_info_description += f'Current date and time: {time_str}'

		_todo_contents = self.file_system.get_todo_contents() if self.file_system else ''
		if not len(_todo_contents):
			_todo_contents = '[Current todo.md is empty, fill it with your plan when applicable]'

		agent_state = f"""
<user_request>
{self.task}
</user_request>
<file_system>
{self.file_system.describe() if self.file_system else 'No file system available'}
</file_system>
<todo_contents>
{_todo_contents}
</todo_contents>
"""
		if self.sensitive_data:
			agent_state += f'<sensitive_data>\n{self.sensitive_data}\n</sensitive_data>\n'

		agent_state += f'<step_info>\n{step_info_description}\n</step_info>\n'
		if self.available_file_paths:
			agent_state += '<available_file_paths>\n' + '\n'.join(self.available_file_paths) + '\n</available_file_paths>\n'
		return agent_state

	@observe_debug(ignore_input=True, ignore_output=True, name='get_user_message')
	def get_user_message(self, use_vision: bool = True) -> UserMessage:
		"""Get complete state as a single cached message"""
		# Don't pass screenshot to model if page is a new tab page, step is 0, and there's only one tab
		if (
			is_new_tab_page(self.browser_state.url)
			and self.step_info is not None
			and self.step_info.step_number == 0
			and len(self.browser_state.tabs) == 1
		):
			use_vision = False

		# Build complete state description
		state_description = (
			'<agent_history>\n'
			+ (self.agent_history_description.strip('\n') if self.agent_history_description else '')
			+ '\n</agent_history>\n'
		)
		state_description += '<agent_state>\n' + self._get_agent_state_description().strip('\n') + '\n</agent_state>\n'
		state_description += '<browser_state>\n' + self._get_browser_state_description().strip('\n') + '\n</browser_state>\n'
		# Only add read_state if it has content
		read_state_description = self.read_state_description.strip('\n').strip() if self.read_state_description else ''
		if read_state_description:
			state_description += '<read_state>\n' + read_state_description + '\n</read_state>\n'

		if self.page_filtered_actions:
			state_description += '<page_specific_actions>\n'
			state_description += self.page_filtered_actions + '\n'
			state_description += '</page_specific_actions>\n'

		if use_vision is True and self.screenshots:
			# Start with text description
			content_parts: list[ContentPartTextParam | ContentPartImageParam] = [ContentPartTextParam(text=state_description)]

			# Add screenshots with labels
			for i, screenshot in enumerate(self.screenshots):
				if i == len(self.screenshots) - 1:
					label = 'Current screenshot:'
				else:
					# Use simple, accurate labeling since we don't have actual step timing info
					label = 'Previous screenshot:'

				# Add label as text content
				content_parts.append(ContentPartTextParam(text=label))

				# Add the screenshot
				content_parts.append(
					ContentPartImageParam(
						image_url=ImageURL(
							url=f'data:image/png;base64,{screenshot}',
							media_type='image/png',
							detail=self.vision_detail_level,
						),
					)
				)

			return UserMessage(content=content_parts, cache=True)

		return UserMessage(content=state_description, cache=True)

# From agent/prompts.py
def get_system_message(self) -> SystemMessage:
		"""
		Get the system prompt for the agent.

		Returns:
		    SystemMessage: Formatted system prompt
		"""
		return self.system_message

# From agent/prompts.py
def get_user_message(self, use_vision: bool = True) -> UserMessage:
		"""Get complete state as a single cached message"""
		# Don't pass screenshot to model if page is a new tab page, step is 0, and there's only one tab
		if (
			is_new_tab_page(self.browser_state.url)
			and self.step_info is not None
			and self.step_info.step_number == 0
			and len(self.browser_state.tabs) == 1
		):
			use_vision = False

		# Build complete state description
		state_description = (
			'<agent_history>\n'
			+ (self.agent_history_description.strip('\n') if self.agent_history_description else '')
			+ '\n</agent_history>\n'
		)
		state_description += '<agent_state>\n' + self._get_agent_state_description().strip('\n') + '\n</agent_state>\n'
		state_description += '<browser_state>\n' + self._get_browser_state_description().strip('\n') + '\n</browser_state>\n'
		# Only add read_state if it has content
		read_state_description = self.read_state_description.strip('\n').strip() if self.read_state_description else ''
		if read_state_description:
			state_description += '<read_state>\n' + read_state_description + '\n</read_state>\n'

		if self.page_filtered_actions:
			state_description += '<page_specific_actions>\n'
			state_description += self.page_filtered_actions + '\n'
			state_description += '</page_specific_actions>\n'

		if use_vision is True and self.screenshots:
			# Start with text description
			content_parts: list[ContentPartTextParam | ContentPartImageParam] = [ContentPartTextParam(text=state_description)]

			# Add screenshots with labels
			for i, screenshot in enumerate(self.screenshots):
				if i == len(self.screenshots) - 1:
					label = 'Current screenshot:'
				else:
					# Use simple, accurate labeling since we don't have actual step timing info
					label = 'Previous screenshot:'

				# Add label as text content
				content_parts.append(ContentPartTextParam(text=label))

				# Add the screenshot
				content_parts.append(
					ContentPartImageParam(
						image_url=ImageURL(
							url=f'data:image/png;base64,{screenshot}',
							media_type='image/png',
							detail=self.vision_detail_level,
						),
					)
				)

			return UserMessage(content=content_parts, cache=True)

		return UserMessage(content=state_description, cache=True)

from datetime import timezone
from pydantic import field_validator

# From agent/cloud_events.py
class UpdateAgentTaskEvent(BaseEvent):
	# Required fields for identification
	id: str  # The task ID to update
	user_id: str = Field(max_length=255)  # For authorization
	device_id: str | None = Field(None, max_length=255)  # Device ID for auth lookup

	# Optional fields that can be updated
	stopped: bool | None = None
	paused: bool | None = None
	done_output: str | None = Field(None, max_length=MAX_STRING_LENGTH)
	finished_at: datetime | None = None
	agent_state: dict | None = None
	user_feedback_type: str | None = Field(None, max_length=10)  # UserFeedbackType enum value as string
	user_comment: str | None = Field(None, max_length=MAX_COMMENT_LENGTH)
	gif_url: str | None = Field(None, max_length=MAX_URL_LENGTH)

	@classmethod
	def from_agent(cls, agent) -> 'UpdateAgentTaskEvent':
		"""Create an UpdateAgentTaskEvent from an Agent instance"""
		if not hasattr(agent, '_task_start_time'):
			raise ValueError('Agent must have _task_start_time attribute')

		done_output = agent.history.final_result() if agent.history else None
		return cls(
			id=str(agent.task_id),
			user_id='',  # To be filled by cloud handler
			device_id=agent.cloud_sync.auth_client.device_id
			if hasattr(agent, 'cloud_sync') and agent.cloud_sync and agent.cloud_sync.auth_client
			else None,
			stopped=agent.state.stopped if hasattr(agent.state, 'stopped') else False,
			paused=agent.state.paused if hasattr(agent.state, 'paused') else False,
			done_output=done_output,
			finished_at=datetime.now(timezone.utc) if agent.history and agent.history.is_done() else None,
			agent_state=agent.state.model_dump() if hasattr(agent.state, 'model_dump') else {},
			user_feedback_type=None,
			user_comment=None,
			gif_url=None,
			# user_feedback_type and user_comment would be set by the API/frontend
			# gif_url would be set after GIF generation if needed
		)

# From agent/cloud_events.py
class CreateAgentOutputFileEvent(BaseEvent):
	# Model fields
	id: str = Field(default_factory=uuid7str)
	user_id: str = Field(max_length=255)
	device_id: str | None = Field(None, max_length=255)  # Device ID for auth lookup
	task_id: str
	file_name: str = Field(max_length=255)
	file_content: str | None = None  # Base64 encoded file content
	content_type: str | None = Field(None, max_length=100)  # MIME type for file uploads
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	@field_validator('file_content')
	@classmethod
	def validate_file_size(cls, v: str | None) -> str | None:
		"""Validate base64 file content size."""
		if v is None:
			return v
		# Remove data URL prefix if present
		if ',' in v:
			v = v.split(',')[1]
		# Estimate decoded size (base64 is ~33% larger)
		estimated_size = len(v) * 3 / 4
		if estimated_size > MAX_FILE_CONTENT_SIZE:
			raise ValueError(f'File content exceeds maximum size of {MAX_FILE_CONTENT_SIZE / 1024 / 1024}MB')
		return v

	@classmethod
	async def from_agent_and_file(cls, agent, output_path: str) -> 'CreateAgentOutputFileEvent':
		"""Create a CreateAgentOutputFileEvent from a file path"""

		gif_path = Path(output_path)
		if not gif_path.exists():
			raise FileNotFoundError(f'File not found: {output_path}')

		gif_size = os.path.getsize(gif_path)

		# Read GIF content for base64 encoding if needed
		gif_content = None
		if gif_size < 50 * 1024 * 1024:  # Only read if < 50MB
			async with await anyio.open_file(gif_path, 'rb') as f:
				gif_bytes = await f.read()
				gif_content = base64.b64encode(gif_bytes).decode('utf-8')

		return cls(
			user_id='',  # To be filled by cloud handler
			device_id=agent.cloud_sync.auth_client.device_id
			if hasattr(agent, 'cloud_sync') and agent.cloud_sync and agent.cloud_sync.auth_client
			else None,
			task_id=str(agent.task_id),
			file_name=gif_path.name,
			file_content=gif_content,  # Base64 encoded
			content_type='image/gif',
		)

# From agent/cloud_events.py
class CreateAgentStepEvent(BaseEvent):
	# Model fields
	id: str = Field(default_factory=uuid7str)
	user_id: str = Field(max_length=255)  # Added for authorization checks
	device_id: str | None = Field(None, max_length=255)  # Device ID for auth lookup
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	agent_task_id: str
	step: int
	evaluation_previous_goal: str = Field(max_length=MAX_STRING_LENGTH)
	memory: str = Field(max_length=MAX_STRING_LENGTH)
	next_goal: str = Field(max_length=MAX_STRING_LENGTH)
	actions: list[dict]
	screenshot_url: str | None = Field(None, max_length=MAX_FILE_CONTENT_SIZE)  # ~50MB for base64 images
	url: str = Field(default='', max_length=MAX_URL_LENGTH)

	@field_validator('screenshot_url')
	@classmethod
	def validate_screenshot_size(cls, v: str | None) -> str | None:
		"""Validate screenshot URL or base64 content size."""
		if v is None or not v.startswith('data:'):
			return v
		# It's base64 data, check size
		if ',' in v:
			base64_part = v.split(',')[1]
			estimated_size = len(base64_part) * 3 / 4
			if estimated_size > MAX_FILE_CONTENT_SIZE:
				raise ValueError(f'Screenshot content exceeds maximum size of {MAX_FILE_CONTENT_SIZE / 1024 / 1024}MB')
		return v

	@classmethod
	def from_agent_step(
		cls, agent, model_output, result: list, actions_data: list[dict], browser_state_summary
	) -> 'CreateAgentStepEvent':
		"""Create a CreateAgentStepEvent from agent step data"""
		# Get first action details if available
		first_action = model_output.action[0] if model_output.action else None

		# Extract current state from model output
		current_state = model_output.current_state if hasattr(model_output, 'current_state') else None

		# Capture screenshot as base64 data URL if available
		screenshot_url = None
		if browser_state_summary.screenshot:
			screenshot_url = f'data:image/png;base64,{browser_state_summary.screenshot}'

		return cls(
			user_id='',  # To be filled by cloud handler
			device_id=agent.cloud_sync.auth_client.device_id
			if hasattr(agent, 'cloud_sync') and agent.cloud_sync and agent.cloud_sync.auth_client
			else None,
			agent_task_id=str(agent.task_id),
			step=agent.state.n_steps,
			evaluation_previous_goal=current_state.evaluation_previous_goal if current_state else '',
			memory=current_state.memory if current_state else '',
			next_goal=current_state.next_goal if current_state else '',
			actions=actions_data,  # List of action dicts
			url=browser_state_summary.url,
			screenshot_url=screenshot_url,
		)

# From agent/cloud_events.py
class CreateAgentTaskEvent(BaseEvent):
	# Model fields
	id: str = Field(default_factory=uuid7str)
	user_id: str = Field(max_length=255)  # Added for authorization checks
	device_id: str | None = Field(None, max_length=255)  # Device ID for auth lookup
	agent_session_id: str
	llm_model: str = Field(max_length=100)  # LLMModel enum value as string
	stopped: bool = False
	paused: bool = False
	task: str = Field(max_length=MAX_TASK_LENGTH)
	done_output: str | None = Field(None, max_length=MAX_STRING_LENGTH)
	scheduled_task_id: str | None = None
	started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	finished_at: datetime | None = None
	agent_state: dict = Field(default_factory=dict)
	user_feedback_type: str | None = Field(None, max_length=10)  # UserFeedbackType enum value as string
	user_comment: str | None = Field(None, max_length=MAX_COMMENT_LENGTH)
	gif_url: str | None = Field(None, max_length=MAX_URL_LENGTH)

	@classmethod
	def from_agent(cls, agent) -> 'CreateAgentTaskEvent':
		"""Create a CreateAgentTaskEvent from an Agent instance"""
		return cls(
			id=str(agent.task_id),
			user_id='',  # To be filled by cloud handler
			device_id=agent.cloud_sync.auth_client.device_id
			if hasattr(agent, 'cloud_sync') and agent.cloud_sync and agent.cloud_sync.auth_client
			else None,
			agent_session_id=str(agent.session_id),
			task=agent.task,
			llm_model=agent.llm.model_name,
			agent_state=agent.state.model_dump() if hasattr(agent.state, 'model_dump') else {},
			stopped=False,
			paused=False,
			done_output=None,
			started_at=datetime.fromtimestamp(agent._task_start_time, tz=timezone.utc),
			finished_at=None,
			user_feedback_type=None,
			user_comment=None,
			gif_url=None,
		)

# From agent/cloud_events.py
class CreateAgentSessionEvent(BaseEvent):
	# Model fields
	id: str = Field(default_factory=uuid7str)
	user_id: str = Field(max_length=255)
	device_id: str | None = Field(None, max_length=255)  # Device ID for auth lookup
	browser_session_id: str = Field(max_length=255)
	browser_session_live_url: str = Field(max_length=MAX_URL_LENGTH)
	browser_session_cdp_url: str = Field(max_length=MAX_URL_LENGTH)
	browser_session_stopped: bool = False
	browser_session_stopped_at: datetime | None = None
	is_source_api: bool | None = None
	browser_state: dict = Field(default_factory=dict)
	browser_session_data: dict | None = None

	@classmethod
	def from_agent(cls, agent) -> 'CreateAgentSessionEvent':
		"""Create a CreateAgentSessionEvent from an Agent instance"""
		return cls(
			id=str(agent.session_id),
			user_id='',  # To be filled by cloud handler
			device_id=agent.cloud_sync.auth_client.device_id
			if hasattr(agent, 'cloud_sync') and agent.cloud_sync and agent.cloud_sync.auth_client
			else None,
			browser_session_id=agent.browser_session.id,
			browser_session_live_url='',  # To be filled by cloud handler
			browser_session_cdp_url='',  # To be filled by cloud handler
			browser_state={
				'viewport': agent.browser_profile.viewport if agent.browser_profile else {'width': 1280, 'height': 720},
				'user_agent': agent.browser_profile.user_agent if agent.browser_profile else None,
				'headless': agent.browser_profile.headless if agent.browser_profile else True,
				'initial_url': None,  # Will be updated during execution
				'final_url': None,  # Will be updated during execution
				'total_pages_visited': 0,  # Will be updated during execution
				'session_duration_seconds': 0,  # Will be updated during execution
			},
			browser_session_data={
				'cookies': [],
				'secrets': {},
				# TODO: send secrets safely so tasks can be replayed on cloud seamlessly
				# 'secrets': dict(agent.sensitive_data) if agent.sensitive_data else {},
				'allowed_domains': agent.browser_profile.allowed_domains if agent.browser_profile else [],
			},
		)

# From agent/cloud_events.py
def from_agent(cls, agent) -> 'UpdateAgentTaskEvent':
		"""Create an UpdateAgentTaskEvent from an Agent instance"""
		if not hasattr(agent, '_task_start_time'):
			raise ValueError('Agent must have _task_start_time attribute')

		done_output = agent.history.final_result() if agent.history else None
		return cls(
			id=str(agent.task_id),
			user_id='',  # To be filled by cloud handler
			device_id=agent.cloud_sync.auth_client.device_id
			if hasattr(agent, 'cloud_sync') and agent.cloud_sync and agent.cloud_sync.auth_client
			else None,
			stopped=agent.state.stopped if hasattr(agent.state, 'stopped') else False,
			paused=agent.state.paused if hasattr(agent.state, 'paused') else False,
			done_output=done_output,
			finished_at=datetime.now(timezone.utc) if agent.history and agent.history.is_done() else None,
			agent_state=agent.state.model_dump() if hasattr(agent.state, 'model_dump') else {},
			user_feedback_type=None,
			user_comment=None,
			gif_url=None,
			# user_feedback_type and user_comment would be set by the API/frontend
			# gif_url would be set after GIF generation if needed
		)

# From agent/cloud_events.py
def validate_file_size(cls, v: str | None) -> str | None:
		"""Validate base64 file content size."""
		if v is None:
			return v
		# Remove data URL prefix if present
		if ',' in v:
			v = v.split(',')[1]
		# Estimate decoded size (base64 is ~33% larger)
		estimated_size = len(v) * 3 / 4
		if estimated_size > MAX_FILE_CONTENT_SIZE:
			raise ValueError(f'File content exceeds maximum size of {MAX_FILE_CONTENT_SIZE / 1024 / 1024}MB')
		return v

# From agent/cloud_events.py
def validate_screenshot_size(cls, v: str | None) -> str | None:
		"""Validate screenshot URL or base64 content size."""
		if v is None or not v.startswith('data:'):
			return v
		# It's base64 data, check size
		if ',' in v:
			base64_part = v.split(',')[1]
			estimated_size = len(base64_part) * 3 / 4
			if estimated_size > MAX_FILE_CONTENT_SIZE:
				raise ValueError(f'Screenshot content exceeds maximum size of {MAX_FILE_CONTENT_SIZE / 1024 / 1024}MB')
		return v

# From agent/cloud_events.py
def from_agent_step(
		cls, agent, model_output, result: list, actions_data: list[dict], browser_state_summary
	) -> 'CreateAgentStepEvent':
		"""Create a CreateAgentStepEvent from agent step data"""
		# Get first action details if available
		first_action = model_output.action[0] if model_output.action else None

		# Extract current state from model output
		current_state = model_output.current_state if hasattr(model_output, 'current_state') else None

		# Capture screenshot as base64 data URL if available
		screenshot_url = None
		if browser_state_summary.screenshot:
			screenshot_url = f'data:image/png;base64,{browser_state_summary.screenshot}'

		return cls(
			user_id='',  # To be filled by cloud handler
			device_id=agent.cloud_sync.auth_client.device_id
			if hasattr(agent, 'cloud_sync') and agent.cloud_sync and agent.cloud_sync.auth_client
			else None,
			agent_task_id=str(agent.task_id),
			step=agent.state.n_steps,
			evaluation_previous_goal=current_state.evaluation_previous_goal if current_state else '',
			memory=current_state.memory if current_state else '',
			next_goal=current_state.next_goal if current_state else '',
			actions=actions_data,  # List of action dicts
			url=browser_state_summary.url,
			screenshot_url=screenshot_url,
		)

import gc
import inspect
import tempfile
from collections.abc import Awaitable
from browser_use.agent.cloud_events import CreateAgentOutputFileEvent
from browser_use.agent.cloud_events import CreateAgentSessionEvent
from browser_use.agent.cloud_events import CreateAgentStepEvent
from browser_use.agent.cloud_events import CreateAgentTaskEvent
from browser_use.agent.cloud_events import UpdateAgentTaskEvent
from browser_use.agent.message_manager.utils import save_conversation
from browser_use.dom.views import DEFAULT_INCLUDE_ATTRIBUTES
from browser_use.llm.messages import BaseMessage
from browser_use.tokens.service import TokenCost
from bubus import EventBus
from pydantic import ValidationError
from browser_use.agent.message_manager.service import MessageManager
from browser_use.agent.prompts import SystemPrompt
from browser_use.agent.views import AgentError
from browser_use.agent.views import AgentHistory
from browser_use.agent.views import AgentHistoryList
from browser_use.agent.views import AgentOutput
from browser_use.agent.views import AgentState
from browser_use.agent.views import AgentStructuredOutput
from browser_use.agent.views import BrowserStateHistory
from browser_use.agent.views import StepMetadata
from browser_use.browser.session import DEFAULT_BROWSER_PROFILE
from browser_use.browser.types import Browser
from browser_use.browser.types import BrowserContext
from browser_use.controller.registry.views import ActionModel
from browser_use.dom.history_tree_processor.service import DOMHistoryElement
from browser_use.observability import observe
from browser_use.sync import CloudSync
from browser_use.telemetry.service import ProductTelemetry
from browser_use.telemetry.views import AgentTelemetryEvent
from browser_use.utils import _log_pretty_path
from browser_use.utils import get_git_info
from browser_use.utils import SignalHandler
from browser_use.screenshots.service import ScreenshotService
from browser_use.agent.gif import create_history_gif
from anthropic import RateLimitError
from google.api_core.exceptions import ResourceExhausted
from openai import RateLimitError

# From agent/service.py
class Agent(Generic[Context, AgentStructuredOutput]):
	browser_session: BrowserSession | None = None
	_logger: logging.Logger | None = None

	@time_execution_sync('--init')
	def __init__(
		self,
		task: str,
		llm: BaseChatModel,
		# Optional parameters
		page: Page | None = None,
		browser: Browser | BrowserSession | None = None,
		browser_context: BrowserContext | None = None,
		browser_profile: BrowserProfile | None = None,
		browser_session: BrowserSession | None = None,
		controller: Controller[Context] | None = None,
		# Initial agent run parameters
		sensitive_data: dict[str, str | dict[str, str]] | None = None,
		initial_actions: list[dict[str, dict[str, Any]]] | None = None,
		# Cloud Callbacks
		register_new_step_callback: (
			Callable[['BrowserStateSummary', 'AgentOutput', int], None]  # Sync callback
			| Callable[['BrowserStateSummary', 'AgentOutput', int], Awaitable[None]]  # Async callback
			| None
		) = None,
		register_done_callback: (
			Callable[['AgentHistoryList'], Awaitable[None]]  # Async Callback
			| Callable[['AgentHistoryList'], None]  # Sync Callback
			| None
		) = None,
		register_external_agent_status_raise_error_callback: Callable[[], Awaitable[bool]] | None = None,
		# Agent settings
		output_model_schema: type[AgentStructuredOutput] | None = None,
		use_vision: bool = True,
		use_vision_for_planner: bool = False,  # Deprecated
		save_conversation_path: str | Path | None = None,
		save_conversation_path_encoding: str | None = 'utf-8',
		max_failures: int = 3,
		retry_delay: int = 10,
		override_system_message: str | None = None,
		extend_system_message: str | None = None,
		validate_output: bool = False,
		generate_gif: bool | str = False,
		available_file_paths: list[str] | None = None,
		include_attributes: list[str] = DEFAULT_INCLUDE_ATTRIBUTES,
		max_actions_per_step: int = 10,
		use_thinking: bool = True,
		flash_mode: bool = False,
		max_history_items: int | None = None,
		page_extraction_llm: BaseChatModel | None = None,
		planner_llm: BaseChatModel | None = None,  # Deprecated
		planner_interval: int = 1,  # Deprecated
		is_planner_reasoning: bool = False,  # Deprecated
		extend_planner_system_message: str | None = None,  # Deprecated
		injected_agent_state: AgentState | None = None,
		context: Context | None = None,
		source: str | None = None,
		file_system_path: str | None = None,
		task_id: str | None = None,
		cloud_sync: CloudSync | None = None,
		calculate_cost: bool = False,
		display_files_in_done_text: bool = True,
		include_tool_call_examples: bool = False,
		vision_detail_level: Literal['auto', 'low', 'high'] = 'auto',
		llm_timeout: int = 60,
		step_timeout: int = 180,
		**kwargs,
	):
		if not isinstance(llm, BaseChatModel):
			raise ValueError('invalid llm, must be from browser_use.llm')
		# Check for deprecated planner parameters
		planner_params = [planner_llm, use_vision_for_planner, is_planner_reasoning, extend_planner_system_message]
		if any(param is not None and param is not False for param in planner_params) or planner_interval != 1:
			logger.warning(
				'⚠️ Planner functionality has been removed in browser-use v0.3.3+. '
				'The planner_llm, use_vision_for_planner, planner_interval, is_planner_reasoning, '
				'and extend_planner_system_message parameters are deprecated and will be ignored. '
				'Please remove these parameters from your Agent() initialization.'
			)

		# Check for deprecated memory parameters
		if kwargs.get('enable_memory', False) or kwargs.get('memory_config') is not None:
			logger.warning(
				'Memory support has been removed as of version 0.3.2. '
				'The agent context for memory is significantly improved and no longer requires the old memory system. '
				"Please remove the 'enable_memory' and 'memory_config' parameters."
			)
			kwargs['enable_memory'] = False
			kwargs['memory_config'] = None

		if page_extraction_llm is None:
			page_extraction_llm = llm
		if available_file_paths is None:
			available_file_paths = []

		self.id = task_id or uuid7str()
		self.task_id: str = self.id
		self.session_id: str = uuid7str()

		# Initialize available file paths as direct attribute
		self.available_file_paths = available_file_paths

		# Create instance-specific logger
		self._logger = logging.getLogger(f'browser_use.Agent[{self.task_id[-3:]}]')

		# Core components
		self.task = task
		self.llm = llm
		self.controller = (
			controller if controller is not None else Controller(display_files_in_done_text=display_files_in_done_text)
		)

		# Structured output
		self.output_model_schema = output_model_schema
		if self.output_model_schema is not None:
			self.controller.use_structured_output_action(self.output_model_schema)

		self.sensitive_data = sensitive_data

		self.settings = AgentSettings(
			use_vision=use_vision,
			vision_detail_level=vision_detail_level,
			use_vision_for_planner=False,  # Always False now (deprecated)
			save_conversation_path=save_conversation_path,
			save_conversation_path_encoding=save_conversation_path_encoding,
			max_failures=max_failures,
			retry_delay=retry_delay,
			override_system_message=override_system_message,
			extend_system_message=extend_system_message,
			validate_output=validate_output,
			generate_gif=generate_gif,
			include_attributes=include_attributes,
			max_actions_per_step=max_actions_per_step,
			use_thinking=use_thinking,
			flash_mode=flash_mode,
			max_history_items=max_history_items,
			page_extraction_llm=page_extraction_llm,
			planner_llm=None,  # Always None now (deprecated)
			planner_interval=1,  # Always 1 now (deprecated)
			is_planner_reasoning=False,  # Always False now (deprecated)
			extend_planner_system_message=None,  # Always None now (deprecated)
			calculate_cost=calculate_cost,
			include_tool_call_examples=include_tool_call_examples,
			llm_timeout=llm_timeout,
			step_timeout=step_timeout,
		)

		# Token cost service
		self.token_cost_service = TokenCost(include_cost=calculate_cost)
		self.token_cost_service.register_llm(llm)
		self.token_cost_service.register_llm(page_extraction_llm)
		# Note: No longer registering planner_llm (deprecated)

		# Initialize state
		self.state = injected_agent_state or AgentState()

		# Initialize history
		self.history = AgentHistoryList(history=[], usage=None)

		# Initialize agent directory
		import time

		timestamp = int(time.time())
		base_tmp = Path(tempfile.gettempdir())
		self.agent_directory = base_tmp / f'browser_use_agent_{self.id}_{timestamp}'

		# Initialize file system and screenshot service
		self._set_file_system(file_system_path)
		self._set_screenshot_service()

		# Action setup
		self._setup_action_models()
		self._set_browser_use_version_and_source(source)
		self.initial_actions = self._convert_initial_actions(initial_actions) if initial_actions else None

		# Verify we can connect to the model
		self._verify_and_setup_llm()

		# TODO: move this logic to the LLMs
		# Handle users trying to use use_vision=True with DeepSeek models
		if 'deepseek' in self.llm.model.lower():
			self.logger.warning('⚠️ DeepSeek models do not support use_vision=True yet. Setting use_vision=False for now...')
			self.settings.use_vision = False
		# Note: No longer checking planner_llm for DeepSeek (deprecated)

		# Handle users trying to use use_vision=True with XAI models
		if 'grok' in self.llm.model.lower():
			self.logger.warning('⚠️ XAI models do not support use_vision=True yet. Setting use_vision=False for now...')
			self.settings.use_vision = False
		# Note: No longer checking planner_llm for XAI models (deprecated)

		self.logger.info(
			f'🧠 Starting a browser-use agent {self.version} with base_model={self.llm.model}'
			f'{" +vision" if self.settings.use_vision else ""}'
			f' extraction_model={self.settings.page_extraction_llm.model if self.settings.page_extraction_llm else "Unknown"}'
			# Note: No longer logging planner_model (deprecated)
			f'{" +file_system" if self.file_system else ""}'
		)

		# Initialize available actions for system prompt (only non-filtered actions)
		# These will be used for the system prompt to maintain caching
		self.unfiltered_actions = self.controller.registry.get_prompt_description()

		# Initialize message manager with state
		# Initial system prompt with all actions - will be updated during each step
		self._message_manager = MessageManager(
			task=task,
			system_message=SystemPrompt(
				action_description=self.unfiltered_actions,
				max_actions_per_step=self.settings.max_actions_per_step,
				override_system_message=override_system_message,
				extend_system_message=extend_system_message,
				use_thinking=self.settings.use_thinking,
				flash_mode=self.settings.flash_mode,
			).get_system_message(),
			file_system=self.file_system,
			state=self.state.message_manager_state,
			use_thinking=self.settings.use_thinking,
			# Settings that were previously in MessageManagerSettings
			include_attributes=self.settings.include_attributes,
			sensitive_data=sensitive_data,
			max_history_items=self.settings.max_history_items,
			vision_detail_level=self.settings.vision_detail_level,
			include_tool_call_examples=self.settings.include_tool_call_examples,
		)

		if isinstance(browser, BrowserSession):
			browser_session = browser_session or browser

		browser_context = page.context if page else browser_context
		# assert not (browser_session and browser_profile), 'Cannot provide both browser_session and browser_profile'
		# assert not (browser_session and browser), 'Cannot provide both browser_session and browser'
		# assert not (browser_profile and browser), 'Cannot provide both browser_profile and browser'
		# assert not (browser_profile and browser_context), 'Cannot provide both browser_profile and browser_context'
		# assert not (browser and browser_context), 'Cannot provide both browser and browser_context'
		# assert not (browser_session and browser_context), 'Cannot provide both browser_session and browser_context'
		browser_profile = browser_profile or DEFAULT_BROWSER_PROFILE

		if browser_session:
			# Always copy sessions that are passed in to avoid agents overwriting each other's agent_current_page and human_current_page by accident
			# The model_copy() method now handles copying all necessary fields and setting up ownership
			if browser_session._owns_browser_resources:
				self.browser_session = browser_session
			else:
				self.logger.warning(
					'⚠️ Attempting to use multiple Agents with the same BrowserSession! This is not supported yet and will likely lead to strange behavior, use separate BrowserSessions for each Agent.'
				)
				self.browser_session = browser_session.model_copy()
		else:
			if browser is not None:
				assert isinstance(browser, Browser), 'Browser is not set up'
			self.browser_session = BrowserSession(
				browser_profile=browser_profile,
				browser=browser,
				browser_context=browser_context,
				agent_current_page=page,
				id=uuid7str()[:-4] + self.id[-4:],  # re-use the same 4-char suffix so they show up together in logs
			)

		if self.sensitive_data:
			# Check if sensitive_data has domain-specific credentials
			has_domain_specific_credentials = any(isinstance(v, dict) for v in self.sensitive_data.values())

			# If no allowed_domains are configured, show a security warning
			if not self.browser_profile.allowed_domains:
				self.logger.error(
					'⚠️⚠️⚠️ Agent(sensitive_data=••••••••) was provided but BrowserSession(allowed_domains=[...]) is not locked down! ⚠️⚠️⚠️\n'
					'          ☠️ If the agent visits a malicious website and encounters a prompt-injection attack, your sensitive_data may be exposed!\n\n'
					'             https://docs.browser-use.com/customize/browser-settings#restrict-urls\n'
					'Waiting 10 seconds before continuing... Press [Ctrl+C] to abort.'
				)
				if sys.stdin.isatty():
					try:
						time.sleep(10)
					except KeyboardInterrupt:
						print(
							'\n\n 🛑 Exiting now... set BrowserSession(allowed_domains=["example.com", "example.org"]) to only domains you trust to see your sensitive_data.'
						)
						sys.exit(0)
				else:
					pass  # no point waiting if we're not in an interactive shell
				self.logger.warning(
					'‼️ Continuing with insecure settings for now... but this will become a hard error in the future!'
				)

			# If we're using domain-specific credentials, validate domain patterns
			elif has_domain_specific_credentials:
				# For domain-specific format, ensure all domain patterns are included in allowed_domains
				domain_patterns = [k for k, v in self.sensitive_data.items() if isinstance(v, dict)]

				# Validate each domain pattern against allowed_domains
				for domain_pattern in domain_patterns:
					is_allowed = False
					for allowed_domain in self.browser_profile.allowed_domains:
						# Special cases that don't require URL matching
						if domain_pattern == allowed_domain or allowed_domain == '*':
							is_allowed = True
							break

						# Need to create example URLs to compare the patterns
						# Extract the domain parts, ignoring scheme
						pattern_domain = domain_pattern.split('://')[-1] if '://' in domain_pattern else domain_pattern
						allowed_domain_part = allowed_domain.split('://')[-1] if '://' in allowed_domain else allowed_domain

						# Check if pattern is covered by an allowed domain
						# Example: "google.com" is covered by "*.google.com"
						if pattern_domain == allowed_domain_part or (
							allowed_domain_part.startswith('*.')
							and (
								pattern_domain == allowed_domain_part[2:]
								or pattern_domain.endswith('.' + allowed_domain_part[2:])
							)
						):
							is_allowed = True
							break

					if not is_allowed:
						self.logger.warning(
							f'⚠️ Domain pattern "{domain_pattern}" in sensitive_data is not covered by any pattern in allowed_domains={self.browser_profile.allowed_domains}\n'
							f'   This may be a security risk as credentials could be used on unintended domains.'
						)

		# Callbacks
		self.register_new_step_callback = register_new_step_callback
		self.register_done_callback = register_done_callback
		self.register_external_agent_status_raise_error_callback = register_external_agent_status_raise_error_callback

		# Context
		self.context: Context | None = context

		# Telemetry
		self.telemetry = ProductTelemetry()

		# Event bus with WAL persistence
		# Default to ~/.config/browseruse/events/{agent_session_id}.jsonl
		# wal_path = CONFIG.BROWSER_USE_CONFIG_DIR / 'events' / f'{self.session_id}.jsonl'
		self.eventbus = EventBus(name=f'Agent_{str(self.id)[-4:]}')

		# Cloud sync service
		self.enable_cloud_sync = CONFIG.BROWSER_USE_CLOUD_SYNC
		if self.enable_cloud_sync or cloud_sync is not None:
			self.cloud_sync = cloud_sync or CloudSync()
			# Register cloud sync handler
			self.eventbus.on('*', self.cloud_sync.handle_event)

		if self.settings.save_conversation_path:
			self.settings.save_conversation_path = Path(self.settings.save_conversation_path).expanduser().resolve()
			self.logger.info(f'💬 Saving conversation to {_log_pretty_path(self.settings.save_conversation_path)}')

		# Initialize download tracking
		assert self.browser_session is not None, 'BrowserSession is not set up'
		self.has_downloads_path = self.browser_session.browser_profile.downloads_path is not None
		if self.has_downloads_path:
			self._last_known_downloads: list[str] = []
			self.logger.info('📁 Initialized download tracking for agent')

		self._external_pause_event = asyncio.Event()
		self._external_pause_event.set()

	@property
	def logger(self) -> logging.Logger:
		"""Get instance-specific logger with task ID in the name"""

		_browser_session_id = self.browser_session.id if self.browser_session else self.id
		_current_page_id = str(id(self.browser_session and self.browser_session.agent_current_page))[-2:]
		return logging.getLogger(f'browser_use.Agent🅰 {self.task_id[-4:]} on 🆂 {_browser_session_id[-4:]} 🅟 {_current_page_id}')

	@property
	def browser(self) -> Browser:
		assert self.browser_session is not None, 'BrowserSession is not set up'
		assert self.browser_session.browser is not None, 'Browser is not set up'
		return self.browser_session.browser

	@property
	def browser_context(self) -> BrowserContext:
		assert self.browser_session is not None, 'BrowserSession is not set up'
		assert self.browser_session.browser_context is not None, 'BrowserContext is not set up'
		return self.browser_session.browser_context

	@property
	def browser_profile(self) -> BrowserProfile:
		assert self.browser_session is not None, 'BrowserSession is not set up'
		return self.browser_session.browser_profile

	async def _check_and_update_downloads(self, context: str = '') -> None:
		"""Check for new downloads and update available file paths."""
		if not self.has_downloads_path:
			return

		assert self.browser_session is not None, 'BrowserSession is not set up'

		try:
			current_downloads = self.browser_session.downloaded_files
			if current_downloads != self._last_known_downloads:
				self._update_available_file_paths(current_downloads)
				self._last_known_downloads = current_downloads
				if context:
					self.logger.debug(f'📁 {context}: Updated available files')
		except Exception as e:
			error_context = f' {context}' if context else ''
			self.logger.debug(f'📁 Failed to check for downloads{error_context}: {type(e).__name__}: {e}')

	def _update_available_file_paths(self, downloads: list[str]) -> None:
		"""Update available_file_paths with downloaded files."""
		if not self.has_downloads_path:
			return

		current_files = set(self.available_file_paths or [])
		new_files = set(downloads) - current_files

		if new_files:
			self.available_file_paths = list(current_files | new_files)

			self.logger.info(
				f'📁 Added {len(new_files)} downloaded files to available_file_paths (total: {len(self.available_file_paths)} files)'
			)
			for file_path in new_files:
				self.logger.info(f'📄 New file available: {file_path}')
		else:
			self.logger.info(f'📁 No new downloads detected (tracking {len(current_files)} files)')

	def _set_file_system(self, file_system_path: str | None = None) -> None:
		# Check for conflicting parameters
		if self.state.file_system_state and file_system_path:
			raise ValueError(
				'Cannot provide both file_system_state (from agent state) and file_system_path. '
				'Either restore from existing state or create new file system at specified path, not both.'
			)

		# Check if we should restore from existing state first
		if self.state.file_system_state:
			try:
				# Restore file system from state at the exact same location
				self.file_system = FileSystem.from_state(self.state.file_system_state)
				# The parent directory of base_dir is the original file_system_path
				self.file_system_path = str(self.file_system.base_dir)
				logger.info(f'💾 File system restored from state to: {self.file_system_path}')
				return
			except Exception as e:
				logger.error(f'💾 Failed to restore file system from state: {e}')
				raise e

		# Initialize new file system
		try:
			if file_system_path:
				self.file_system = FileSystem(file_system_path)
				self.file_system_path = file_system_path
			else:
				# Use the agent directory for file system
				self.file_system = FileSystem(self.agent_directory)
				self.file_system_path = str(self.agent_directory)
		except Exception as e:
			logger.error(f'💾 Failed to initialize file system: {e}.')
			raise e

		# Save file system state to agent state
		self.state.file_system_state = self.file_system.get_state()

		logger.info(f'💾 File system path: {self.file_system_path}')

	def _set_screenshot_service(self) -> None:
		"""Initialize screenshot service using agent directory"""
		try:
			from browser_use.screenshots.service import ScreenshotService

			self.screenshot_service = ScreenshotService(self.agent_directory)
			logger.info(f'📸 Screenshot service initialized in: {self.agent_directory}/screenshots')
		except Exception as e:
			logger.error(f'📸 Failed to initialize screenshot service: {e}.')
			raise e

	def save_file_system_state(self) -> None:
		"""Save current file system state to agent state"""
		if self.file_system:
			self.state.file_system_state = self.file_system.get_state()
		else:
			logger.error('💾 File system is not set up. Cannot save state.')
			raise ValueError('File system is not set up. Cannot save state.')

	def _set_browser_use_version_and_source(self, source_override: str | None = None) -> None:
		"""Get the version from pyproject.toml and determine the source of the browser-use package"""
		# Use the helper function for version detection
		version = get_browser_use_version()

		# Determine source
		try:
			package_root = Path(__file__).parent.parent.parent
			repo_files = ['.git', 'README.md', 'docs', 'examples']
			if all(Path(package_root / file).exists() for file in repo_files):
				source = 'git'
			else:
				source = 'pip'
		except Exception as e:
			self.logger.debug(f'Error determining source: {e}')
			source = 'unknown'

		if source_override is not None:
			source = source_override
		# self.logger.debug(f'Version: {version}, Source: {source}')  # moved later to _log_agent_run so that people are more likely to include it in copy-pasted support ticket logs
		self.version = version
		self.source = source

	# def _set_model_names(self) -> None:
	# 	self.chat_model_library = self.llm.provider
	# 	self.model_name = self.llm.model

	# 	if self.settings.planner_llm:
	# 		if hasattr(self.settings.planner_llm, 'model_name'):
	# 			self.planner_model_name = self.settings.planner_llm.model_name  # type: ignore
	# 		elif hasattr(self.settings.planner_llm, 'model'):
	# 			self.planner_model_name = self.settings.planner_llm.model  # type: ignore
	# 		else:
	# 			self.planner_model_name = 'Unknown'
	# 	else:
	# 		self.planner_model_name = None

	def _setup_action_models(self) -> None:
		"""Setup dynamic action models from controller's registry"""
		# Initially only include actions with no filters
		self.ActionModel = self.controller.registry.create_action_model()
		# Create output model with the dynamic actions
		if self.settings.flash_mode:
			self.AgentOutput = AgentOutput.type_with_custom_actions_flash_mode(self.ActionModel)
		elif self.settings.use_thinking:
			self.AgentOutput = AgentOutput.type_with_custom_actions(self.ActionModel)
		else:
			self.AgentOutput = AgentOutput.type_with_custom_actions_no_thinking(self.ActionModel)

		# used to force the done action when max_steps is reached
		self.DoneActionModel = self.controller.registry.create_action_model(include_actions=['done'])
		if self.settings.flash_mode:
			self.DoneAgentOutput = AgentOutput.type_with_custom_actions_flash_mode(self.DoneActionModel)
		elif self.settings.use_thinking:
			self.DoneAgentOutput = AgentOutput.type_with_custom_actions(self.DoneActionModel)
		else:
			self.DoneAgentOutput = AgentOutput.type_with_custom_actions_no_thinking(self.DoneActionModel)

	def add_new_task(self, new_task: str) -> None:
		"""Add a new task to the agent, keeping the same task_id as tasks are continuous"""
		# Simply delegate to message manager - no need for new task_id or events
		# The task continues with new instructions, it doesn't end and start a new one
		self.task = new_task
		self._message_manager.add_new_task(new_task)

	@observe_debug(ignore_input=True, ignore_output=True, name='_raise_if_stopped_or_paused')
	async def _raise_if_stopped_or_paused(self) -> None:
		"""Utility function that raises an InterruptedError if the agent is stopped or paused."""

		if self.register_external_agent_status_raise_error_callback:
			if await self.register_external_agent_status_raise_error_callback():
				raise InterruptedError

		if self.state.stopped or self.state.paused:
			# self.logger.debug('Agent paused after getting state')
			raise InterruptedError

	@observe(name='agent.step', ignore_output=True, ignore_input=True)
	@time_execution_async('--step')
	async def step(self, step_info: AgentStepInfo | None = None) -> None:
		"""Execute one step of the task"""
		# Initialize timing first, before any exceptions can occur
		self.step_start_time = time.time()

		browser_state_summary = None

		try:
			# Phase 1: Prepare context and timing
			browser_state_summary = await self._prepare_context(step_info)

			# Phase 2: Get model output and execute actions
			await self._get_next_action(browser_state_summary)
			await self._execute_actions()

			# Phase 3: Post-processing
			await self._post_process()

		except Exception as e:
			# Handle ALL exceptions in one place
			await self._handle_step_error(e)

		finally:
			await self._finalize(browser_state_summary)

	async def _prepare_context(self, step_info: AgentStepInfo | None = None) -> BrowserStateSummary:
		"""Prepare the context for the step: browser state, action models, page actions"""
		# step_start_time is now set in step() method

		assert self.browser_session is not None, 'BrowserSession is not set up'

		self.logger.debug(f'🌐 Step {self.state.n_steps}: Getting browser state...')
		# Capture screenshots if needed for either vision (LLM input) or GIF generation
		should_capture_screenshot = self.settings.use_vision or bool(self.settings.generate_gif)
		browser_state_summary = await self.browser_session.get_browser_state_with_recovery(
			cache_clickable_elements_hashes=True, include_screenshot=should_capture_screenshot
		)
		current_page = await self.browser_session.get_current_page()

		# Check for new downloads after getting browser state (catches PDF auto-downloads and previous step downloads)
		await self._check_and_update_downloads(f'Step {self.state.n_steps}: after getting browser state')

		self._log_step_context(current_page, browser_state_summary)
		await self._raise_if_stopped_or_paused()

		# Update action models with page-specific actions
		self.logger.debug(f'📝 Step {self.state.n_steps}: Updating action models...')
		await self._update_action_models_for_page(current_page)

		# Get page-specific filtered actions
		page_filtered_actions = self.controller.registry.get_prompt_description(current_page)

		# Page-specific actions will be included directly in the browser_state message
		self.logger.debug(f'💬 Step {self.state.n_steps}: Creating state messages for context...')
		self._message_manager.create_state_messages(
			browser_state_summary=browser_state_summary,
			model_output=self.state.last_model_output,
			result=self.state.last_result,
			step_info=step_info,
			use_vision=self.settings.use_vision,
			page_filtered_actions=page_filtered_actions if page_filtered_actions else None,
			sensitive_data=self.sensitive_data,
			available_file_paths=self.available_file_paths,  # Always pass current available_file_paths
		)

		await self._handle_final_step(step_info)
		return browser_state_summary

	@observe_debug(ignore_input=True, name='get_next_action')
	async def _get_next_action(self, browser_state_summary: BrowserStateSummary) -> None:
		"""Execute LLM interaction with retry logic and handle callbacks"""
		input_messages = self._message_manager.get_messages()
		self.logger.debug(
			f'🤖 Step {self.state.n_steps}: Calling LLM with {len(input_messages)} messages (model: {self.llm.model})...'
		)

		try:
			model_output = await asyncio.wait_for(
				self._get_model_output_with_retry(input_messages), timeout=self.settings.llm_timeout
			)
		except TimeoutError:
			raise TimeoutError(
				f'LLM call timed out after {self.settings.llm_timeout} seconds. Keep your thinking and output short.'
			)

		self.state.last_model_output = model_output

		# Check again for paused/stopped state after getting model output
		await self._raise_if_stopped_or_paused()

		# Handle callbacks and conversation saving
		await self._handle_post_llm_processing(browser_state_summary, input_messages)

		# check again if Ctrl+C was pressed before we commit the output to history
		await self._raise_if_stopped_or_paused()

	async def _execute_actions(self) -> None:
		"""Execute the actions from model output"""
		if self.state.last_model_output is None:
			raise ValueError('No model output to execute actions from')

		self.logger.debug(f'⚡ Step {self.state.n_steps}: Executing {len(self.state.last_model_output.action)} actions...')
		result = await self.multi_act(self.state.last_model_output.action)
		self.logger.debug(f'✅ Step {self.state.n_steps}: Actions completed')

		self.state.last_result = result

	async def _post_process(self) -> None:
		"""Handle post-action processing like download tracking and result logging"""
		assert self.browser_session is not None, 'BrowserSession is not set up'

		# Check for new downloads after executing actions
		await self._check_and_update_downloads('after executing actions')

		self.state.consecutive_failures = 0
		self.logger.debug(f'🔄 Step {self.state.n_steps}: Consecutive failures reset to: {self.state.consecutive_failures}')

		# Log completion results
		if self.state.last_result and len(self.state.last_result) > 0 and self.state.last_result[-1].is_done:
			self.logger.info(f'📄 Result: {self.state.last_result[-1].extracted_content}')
			if self.state.last_result[-1].attachments:
				self.logger.info('📎 Click links below to access the attachments:')
				for file_path in self.state.last_result[-1].attachments:
					self.logger.info(f'👉 {file_path}')

	async def _handle_step_error(self, error: Exception) -> None:
		"""Handle all types of errors that can occur during a step"""

		# Handle all other exceptions
		include_trace = self.logger.isEnabledFor(logging.DEBUG)
		error_msg = AgentError.format_error(error, include_trace=include_trace)
		prefix = f'❌ Result failed {self.state.consecutive_failures + 1}/{self.settings.max_failures} times:\n '
		self.state.consecutive_failures += 1

		# TODO: figure out what to do here
		if isinstance(error, (ValidationError, ValueError)):
			self.logger.error(f'{prefix}{error_msg}')
			# Add context message to help model fix validation errors
			validation_hint = 'Your output format was invalid. Please follow the exact schema structure required for actions.'
			# self._message_manager._add_context_message(UserMessage(content=validation_hint))

			if 'Max token limit reached' in error_msg:
				token_hint = 'Your response was too long. Keep your thinking and output concise.'
				# self._message_manager._add_context_message(UserMessage(content=token_hint))
		# Handle InterruptedError specially
		elif isinstance(error, InterruptedError):
			error_msg = 'The agent was interrupted mid-step' + (f' - {error}' if error else '')
			self.logger.error(f'{prefix}{error_msg}')
		elif 'Could not parse response' in error_msg or 'tool_use_failed' in error_msg:
			# give model a hint how output should look like
			logger.debug(f'Model: {self.llm.model} failed')
			error_msg += '\n\nReturn a valid JSON object with the required fields.'
			logger.error(f'{prefix}{error_msg}')
			# Add context message to help model fix parsing errors
			parse_hint = 'Your response could not be parsed. Return a valid JSON object with the required fields.'
			# self._message_manager._add_context_message(UserMessage(content=parse_hint))
		else:
			from anthropic import RateLimitError as AnthropicRateLimitError
			from google.api_core.exceptions import ResourceExhausted
			from openai import RateLimitError

			# Define a tuple of rate limit error types for easier maintenance
			RATE_LIMIT_ERRORS = (
				RateLimitError,  # OpenAI
				ResourceExhausted,  # Google
				AnthropicRateLimitError,  # Anthropic
			)

			if isinstance(error, RATE_LIMIT_ERRORS) or 'on tokens per minute (TPM): Limit' in error_msg:
				logger.warning(f'{prefix}{error_msg}')
				await asyncio.sleep(self.settings.retry_delay)
			else:
				self.logger.error(f'{prefix}{error_msg}')

		self.state.last_result = [ActionResult(error=error_msg)]
		return None

	async def _finalize(self, browser_state_summary: BrowserStateSummary | None) -> None:
		"""Finalize the step with history, logging, and events"""
		step_end_time = time.time()
		if not self.state.last_result:
			return

		if browser_state_summary:
			metadata = StepMetadata(
				step_number=self.state.n_steps,
				step_start_time=self.step_start_time,
				step_end_time=step_end_time,
			)

			# Use _make_history_item like main branch
			await self._make_history_item(self.state.last_model_output, browser_state_summary, self.state.last_result, metadata)

		# Log step completion summary
		self._log_step_completion_summary(self.step_start_time, self.state.last_result)

		# Save file system state after step completion
		self.save_file_system_state()

		# Emit both step created and executed events
		if browser_state_summary and self.state.last_model_output:
			# Extract key step data for the event
			actions_data = []
			if self.state.last_model_output.action:
				for action in self.state.last_model_output.action:
					action_dict = action.model_dump() if hasattr(action, 'model_dump') else {}
					actions_data.append(action_dict)

			# Emit CreateAgentStepEvent
			step_event = CreateAgentStepEvent.from_agent_step(
				self, self.state.last_model_output, self.state.last_result, actions_data, browser_state_summary
			)
			self.eventbus.dispatch(step_event)

		# Increment step counter after step is fully completed
		self.state.n_steps += 1

	async def _handle_final_step(self, step_info: AgentStepInfo | None = None) -> None:
		"""Handle special processing for the last step"""
		if step_info and step_info.is_last_step():
			# Add last step warning if needed
			msg = 'Now comes your last step. Use only the "done" action now. No other actions - so here your action sequence must have length 1.'
			msg += '\nIf the task is not yet fully finished as requested by the user, set success in "done" to false! E.g. if not all steps are fully completed.'
			msg += '\nIf the task is fully finished, set success in "done" to true.'
			msg += '\nInclude everything you found out for the ultimate task in the done text.'
			self.logger.info('Last step finishing up')
			self._message_manager._add_context_message(UserMessage(content=msg))
			self.AgentOutput = self.DoneAgentOutput

	async def _get_model_output_with_retry(self, input_messages: list[BaseMessage]) -> AgentOutput:
		"""Get model output with retry logic for empty actions"""
		model_output = await self.get_model_output(input_messages)
		self.logger.debug(
			f'✅ Step {self.state.n_steps}: Got LLM response with {len(model_output.action) if model_output.action else 0} actions'
		)

		if (
			not model_output.action
			or not isinstance(model_output.action, list)
			or all(action.model_dump() == {} for action in model_output.action)
		):
			self.logger.warning('Model returned empty action. Retrying...')

			clarification_message = UserMessage(
				content='You forgot to return an action. Please respond only with a valid JSON action according to the expected format.'
			)

			retry_messages = input_messages + [clarification_message]
			model_output = await self.get_model_output(retry_messages)

			if not model_output.action or all(action.model_dump() == {} for action in model_output.action):
				self.logger.warning('Model still returned empty after retry. Inserting safe noop action.')
				action_instance = self.ActionModel()
				setattr(
					action_instance,
					'done',
					{
						'success': False,
						'text': 'No next action returned by LLM!',
					},
				)
				model_output.action = [action_instance]

		return model_output

	async def _handle_post_llm_processing(
		self, browser_state_summary: BrowserStateSummary, input_messages: list[BaseMessage]
	) -> None:
		"""Handle callbacks and conversation saving after LLM interaction"""
		if self.register_new_step_callback and self.state.last_model_output:
			if inspect.iscoroutinefunction(self.register_new_step_callback):
				await self.register_new_step_callback(browser_state_summary, self.state.last_model_output, self.state.n_steps)
			else:
				self.register_new_step_callback(browser_state_summary, self.state.last_model_output, self.state.n_steps)

		if self.settings.save_conversation_path and self.state.last_model_output:
			# Treat save_conversation_path as a directory (consistent with other recording paths)
			conversation_dir = Path(self.settings.save_conversation_path)
			conversation_filename = f'conversation_{self.id}_{self.state.n_steps}.txt'
			target = conversation_dir / conversation_filename
			await save_conversation(
				input_messages,
				self.state.last_model_output,
				target,
				self.settings.save_conversation_path_encoding,
			)

	async def _make_history_item(
		self,
		model_output: AgentOutput | None,
		browser_state_summary: BrowserStateSummary,
		result: list[ActionResult],
		metadata: StepMetadata | None = None,
	) -> None:
		"""Create and store history item"""

		if model_output:
			interacted_elements = AgentHistory.get_interacted_element(model_output, browser_state_summary.selector_map)
		else:
			interacted_elements = [None]

		# Store screenshot and get path
		screenshot_path = None
		if browser_state_summary.screenshot:
			screenshot_path = await self.screenshot_service.store_screenshot(browser_state_summary.screenshot, self.state.n_steps)

		state_history = BrowserStateHistory(
			url=browser_state_summary.url,
			title=browser_state_summary.title,
			tabs=browser_state_summary.tabs,
			interacted_element=interacted_elements,
			screenshot_path=screenshot_path,
		)

		history_item = AgentHistory(
			model_output=model_output,
			result=result,
			state=state_history,
			metadata=metadata,
		)

		self.history.add_item(history_item)

	def _remove_think_tags(self, text: str) -> str:
		THINK_TAGS = re.compile(r'<think>.*?</think>', re.DOTALL)
		STRAY_CLOSE_TAG = re.compile(r'.*?</think>', re.DOTALL)
		# Step 1: Remove well-formed <think>...</think>
		text = re.sub(THINK_TAGS, '', text)
		# Step 2: If there's an unmatched closing tag </think>,
		#         remove everything up to and including that.
		text = re.sub(STRAY_CLOSE_TAG, '', text)
		return text.strip()

	@time_execution_async('--get_next_action')
	@observe_debug(ignore_input=True, ignore_output=True, name='get_model_output')
	async def get_model_output(self, input_messages: list[BaseMessage]) -> AgentOutput:
		"""Get next action from LLM based on current state"""

		try:
			response = await self.llm.ainvoke(input_messages, output_format=self.AgentOutput)
			parsed = response.completion

			# cut the number of actions to max_actions_per_step if needed
			if len(parsed.action) > self.settings.max_actions_per_step:
				parsed.action = parsed.action[: self.settings.max_actions_per_step]

			if not (hasattr(self.state, 'paused') and (self.state.paused or self.state.stopped)):
				log_response(parsed, self.controller.registry.registry, self.logger)

			self._log_next_action_summary(parsed)
			return parsed
		except ValidationError as e:
			# Just re-raise - Pydantic's validation errors are already descriptive
			raise

	def _log_agent_run(self) -> None:
		"""Log the agent run"""
		self.logger.info(f'🚀 Starting task: {self.task}')

		self.logger.debug(f'🤖 Browser-Use Library Version {self.version} ({self.source})')

	def _log_step_context(self, current_page, browser_state_summary) -> None:
		"""Log step context information"""
		url_short = current_page.url[:50] + '...' if len(current_page.url) > 50 else current_page.url
		interactive_count = len(browser_state_summary.selector_map) if browser_state_summary else 0
		self.logger.info(
			f'📍 Step {self.state.n_steps}: Evaluating page with {interactive_count} interactive elements on: {url_short}'
		)

	def _log_next_action_summary(self, parsed: 'AgentOutput') -> None:
		"""Log a comprehensive summary of the next action(s)"""
		if not (self.logger.isEnabledFor(logging.DEBUG) and parsed.action):
			return

		action_count = len(parsed.action)

		# Collect action details
		action_details = []
		for i, action in enumerate(parsed.action):
			action_data = action.model_dump(exclude_unset=True)
			action_name = next(iter(action_data.keys())) if action_data else 'unknown'
			action_params = action_data.get(action_name, {}) if action_data else {}

			# Format key parameters concisely
			param_summary = []
			if isinstance(action_params, dict):
				for key, value in action_params.items():
					if key == 'index':
						param_summary.append(f'#{value}')
					elif key == 'text' and isinstance(value, str):
						text_preview = value[:30] + '...' if len(value) > 30 else value
						param_summary.append(f'text="{text_preview}"')
					elif key == 'url':
						param_summary.append(f'url="{value}"')
					elif key == 'success':
						param_summary.append(f'success={value}')
					elif isinstance(value, (str, int, bool)):
						val_str = str(value)[:30] + '...' if len(str(value)) > 30 else str(value)
						param_summary.append(f'{key}={val_str}')

			param_str = f'({", ".join(param_summary)})' if param_summary else ''
			action_details.append(f'{action_name}{param_str}')

		# Create summary based on single vs multi-action
		if action_count == 1:
			self.logger.info(f'☝️ Decided next action: {action_name}{param_str}')
		else:
			summary_lines = [f'✌️ Decided next {action_count} multi-actions:']
			for i, detail in enumerate(action_details):
				summary_lines.append(f'          {i + 1}. {detail}')
			self.logger.info('\n'.join(summary_lines))

	def _log_step_completion_summary(self, step_start_time: float, result: list[ActionResult]) -> None:
		"""Log step completion summary with action count, timing, and success/failure stats"""
		if not result:
			return

		step_duration = time.time() - step_start_time
		action_count = len(result)

		# Count success and failures
		success_count = sum(1 for r in result if not r.error)
		failure_count = action_count - success_count

		# Format success/failure indicators
		success_indicator = f'✅ {success_count}' if success_count > 0 else ''
		failure_indicator = f'❌ {failure_count}' if failure_count > 0 else ''
		status_parts = [part for part in [success_indicator, failure_indicator] if part]
		status_str = ' | '.join(status_parts) if status_parts else '✅ 0'

		self.logger.info(f'📍 Step {self.state.n_steps}: Ran {action_count} actions in {step_duration:.2f}s: {status_str}')

	def _log_agent_event(self, max_steps: int, agent_run_error: str | None = None) -> None:
		"""Sent the agent event for this run to telemetry"""

		token_summary = self.token_cost_service.get_usage_tokens_for_model(self.llm.model)

		# Prepare action_history data correctly
		action_history_data = []
		for item in self.history.history:
			if item.model_output and item.model_output.action:
				# Convert each ActionModel in the step to its dictionary representation
				step_actions = [
					action.model_dump(exclude_unset=True)
					for action in item.model_output.action
					if action  # Ensure action is not None if list allows it
				]
				action_history_data.append(step_actions)
			else:
				# Append None or [] if a step had no actions or no model output
				action_history_data.append(None)

		final_res = self.history.final_result()
		final_result_str = json.dumps(final_res) if final_res is not None else None

		self.telemetry.capture(
			AgentTelemetryEvent(
				task=self.task,
				model=self.llm.model,
				model_provider=self.llm.provider,
				planner_llm=self.settings.planner_llm.model if self.settings.planner_llm else None,
				max_steps=max_steps,
				max_actions_per_step=self.settings.max_actions_per_step,
				use_vision=self.settings.use_vision,
				use_validation=self.settings.validate_output,
				version=self.version,
				source=self.source,
				cdp_url=urlparse(self.browser_session.cdp_url).hostname
				if self.browser_session and self.browser_session.cdp_url
				else None,
				action_errors=self.history.errors(),
				action_history=action_history_data,
				urls_visited=self.history.urls(),
				steps=self.state.n_steps,
				total_input_tokens=token_summary.prompt_tokens,
				total_duration_seconds=self.history.total_duration_seconds(),
				success=self.history.is_successful(),
				final_result_response=final_result_str,
				error_message=agent_run_error,
			)
		)

	async def take_step(self, step_info: AgentStepInfo | None = None) -> tuple[bool, bool]:
		"""Take a step

		Returns:
		        Tuple[bool, bool]: (is_done, is_valid)
		"""
		await self.step(step_info)

		if self.history.is_done():
			await self.log_completion()
			if self.register_done_callback:
				if inspect.iscoroutinefunction(self.register_done_callback):
					await self.register_done_callback(self.history)
				else:
					self.register_done_callback(self.history)
			return True, True

		return False, False

	@observe(name='agent.run', metadata={'task': '{{task}}', 'debug': '{{debug}}'})
	@time_execution_async('--run')
	async def run(
		self,
		max_steps: int = 100,
		on_step_start: AgentHookFunc | None = None,
		on_step_end: AgentHookFunc | None = None,
	) -> AgentHistoryList[AgentStructuredOutput]:
		"""Execute the task with maximum number of steps"""

		loop = asyncio.get_event_loop()
		agent_run_error: str | None = None  # Initialize error tracking variable
		self._force_exit_telemetry_logged = False  # ADDED: Flag for custom telemetry on force exit

		# Set up the  signal handler with callbacks specific to this agent
		from browser_use.utils import SignalHandler

		# Define the custom exit callback function for second CTRL+C
		def on_force_exit_log_telemetry():
			self._log_agent_event(max_steps=max_steps, agent_run_error='SIGINT: Cancelled by user')
			# NEW: Call the flush method on the telemetry instance
			if hasattr(self, 'telemetry') and self.telemetry:
				self.telemetry.flush()
			self._force_exit_telemetry_logged = True  # Set the flag

		signal_handler = SignalHandler(
			loop=loop,
			pause_callback=self.pause,
			resume_callback=self.resume,
			custom_exit_callback=on_force_exit_log_telemetry,  # Pass the new telemetrycallback
			exit_on_second_int=True,
		)
		signal_handler.register()

		try:
			self._log_agent_run()

			self.logger.debug(
				f'🔧 Agent setup: Task ID {self.task_id[-4:]}, Session ID {self.session_id[-4:]}, Browser Session ID {self.browser_session.id[-4:] if self.browser_session else "None"}'
			)

			# Initialize timing for session and task
			self._session_start_time = time.time()
			self._task_start_time = self._session_start_time  # Initialize task start time

			self.logger.debug('📡 Dispatching CreateAgentSessionEvent...')
			# Emit CreateAgentSessionEvent at the START of run()
			self.eventbus.dispatch(CreateAgentSessionEvent.from_agent(self))

			self.logger.debug('📡 Dispatching CreateAgentTaskEvent...')
			# Emit CreateAgentTaskEvent at the START of run()
			self.eventbus.dispatch(CreateAgentTaskEvent.from_agent(self))

			# Execute initial actions if provided
			if self.initial_actions:
				self.logger.debug(f'⚡ Executing {len(self.initial_actions)} initial actions...')
				result = await self.multi_act(self.initial_actions, check_for_new_elements=False)
				self.state.last_result = result
				self.logger.debug('✅ Initial actions completed')

			self.logger.debug(f'🔄 Starting main execution loop with max {max_steps} steps...')
			for step in range(max_steps):
				# Replace the polling with clean pause-wait
				if self.state.paused:
					self.logger.debug(f'⏸️ Step {step}: Agent paused, waiting to resume...')
					await self.wait_until_resumed()
					signal_handler.reset()

				# Check if we should stop due to too many failures
				if self.state.consecutive_failures >= self.settings.max_failures:
					self.logger.error(f'❌ Stopping due to {self.settings.max_failures} consecutive failures')
					agent_run_error = f'Stopped due to {self.settings.max_failures} consecutive failures'
					break

				# Check control flags before each step
				if self.state.stopped:
					self.logger.info('🛑 Agent stopped')
					agent_run_error = 'Agent stopped programmatically'
					break

				while self.state.paused:
					await asyncio.sleep(0.2)  # Small delay to prevent CPU spinning
					if self.state.stopped:  # Allow stopping while paused
						agent_run_error = 'Agent stopped programmatically while paused'
						break

				if on_step_start is not None:
					await on_step_start(self)

				self.logger.debug(f'🚶 Starting step {step + 1}/{max_steps}...')
				step_info = AgentStepInfo(step_number=step, max_steps=max_steps)

				try:
					await asyncio.wait_for(
						self.step(step_info),
						timeout=self.settings.step_timeout,
					)
					self.logger.debug(f'✅ Completed step {step + 1}/{max_steps}')
				except TimeoutError:
					# Handle step timeout gracefully
					error_msg = f'Step {step + 1} timed out after {self.settings.step_timeout} seconds'
					self.logger.error(f'⏰ {error_msg}')
					self.state.consecutive_failures += 1
					self.state.last_result = [ActionResult(error=error_msg)]

				if on_step_end is not None:
					await on_step_end(self)

				if self.history.is_done():
					self.logger.debug(f'🎯 Task completed after {step + 1} steps!')
					await self.log_completion()

					if self.register_done_callback:
						if inspect.iscoroutinefunction(self.register_done_callback):
							await self.register_done_callback(self.history)
						else:
							self.register_done_callback(self.history)

					# Task completed
					break
			else:
				agent_run_error = 'Failed to complete task in maximum steps'

				self.history.add_item(
					AgentHistory(
						model_output=None,
						result=[ActionResult(error=agent_run_error, include_in_memory=True)],
						state=BrowserStateHistory(
							url='',
							title='',
							tabs=[],
							interacted_element=[],
							screenshot_path=None,
						),
						metadata=None,
					)
				)

				self.logger.info(f'❌ {agent_run_error}')

			self.logger.debug('📊 Collecting usage summary...')
			self.history.usage = await self.token_cost_service.get_usage_summary()

			# set the model output schema and call it on the fly
			if self.history._output_model_schema is None and self.output_model_schema is not None:
				self.history._output_model_schema = self.output_model_schema

			self.logger.debug('🏁 Agent.run() completed successfully')
			return self.history

		except KeyboardInterrupt:
			# Already handled by our signal handler, but catch any direct KeyboardInterrupt as well
			self.logger.info('Got KeyboardInterrupt during execution, returning current history')
			agent_run_error = 'KeyboardInterrupt'

			self.history.usage = await self.token_cost_service.get_usage_summary()

			return self.history

		except Exception as e:
			self.logger.error(f'Agent run failed with exception: {e}', exc_info=True)
			agent_run_error = str(e)
			raise e

		finally:
			# Log token usage summary
			await self.token_cost_service.log_usage_summary()

			# Unregister signal handlers before cleanup
			signal_handler.unregister()

			if not self._force_exit_telemetry_logged:  # MODIFIED: Check the flag
				try:
					self._log_agent_event(max_steps=max_steps, agent_run_error=agent_run_error)
				except Exception as log_e:  # Catch potential errors during logging itself
					self.logger.error(f'Failed to log telemetry event: {log_e}', exc_info=True)
			else:
				# ADDED: Info message when custom telemetry for SIGINT was already logged
				self.logger.info('Telemetry for force exit (SIGINT) was logged by custom exit callback.')

			# NOTE: CreateAgentSessionEvent and CreateAgentTaskEvent are now emitted at the START of run()
			# to match backend requirements for CREATE events to be fired when entities are created,
			# not when they are completed

			# Emit UpdateAgentTaskEvent at the END of run() with final task state
			self.eventbus.dispatch(UpdateAgentTaskEvent.from_agent(self))

			# Generate GIF if needed before stopping event bus
			if self.settings.generate_gif:
				output_path: str = 'agent_history.gif'
				if isinstance(self.settings.generate_gif, str):
					output_path = self.settings.generate_gif

				# Lazy import gif module to avoid heavy startup cost
				from browser_use.agent.gif import create_history_gif

				create_history_gif(task=self.task, history=self.history, output_path=output_path)

				# Only emit output file event if GIF was actually created
				if Path(output_path).exists():
					output_event = await CreateAgentOutputFileEvent.from_agent_and_file(self, output_path)
					self.eventbus.dispatch(output_event)

			# Wait briefly for cloud auth to start and print the URL, but don't block for completion
			if self.enable_cloud_sync and hasattr(self, 'cloud_sync'):
				if self.cloud_sync.auth_task and not self.cloud_sync.auth_task.done():
					try:
						# Wait up to 1 second for auth to start and print URL
						await asyncio.wait_for(self.cloud_sync.auth_task, timeout=1.0)
					except TimeoutError:
						logger.info('Cloud authentication started - continuing in background')
					except Exception as e:
						logger.debug(f'Cloud authentication error: {e}')

			# Stop the event bus gracefully, waiting for all events to be processed
			# Use longer timeout to avoid deadlocks in tests with multiple agents
			await self.eventbus.stop(timeout=10.0)

			await self.close()

	@observe_debug(ignore_input=True, ignore_output=True)
	@time_execution_async('--multi_act')
	async def multi_act(
		self,
		actions: list[ActionModel],
		check_for_new_elements: bool = True,
	) -> list[ActionResult]:
		"""Execute multiple actions"""
		results: list[ActionResult] = []

		assert self.browser_session is not None, 'BrowserSession is not set up'
		cached_selector_map = {}
		cached_path_hashes = set()
		# check all actions if any has index, if so, get the selector map
		for action in actions:
			if action.get_index() is not None:
				cached_selector_map = await self.browser_session.get_selector_map()
				cached_path_hashes = {e.hash.branch_path_hash for e in cached_selector_map.values()}
				break

		# loop over actions and execute them
		for i, action in enumerate(actions):
			if i > 0:
				# ONLY ALLOW TO CALL `done` IF IT IS A SINGLE ACTION
				if action.model_dump(exclude_unset=True).get('done') is not None:
					msg = f'Done action is allowed only as a single action - stopped after action {i} / {len(actions)}.'
					logger.info(msg)
					break

				if action.get_index() is not None:
					new_browser_state_summary = await self.browser_session.get_browser_state_with_recovery(
						cache_clickable_elements_hashes=False, include_screenshot=False
					)
					new_selector_map = new_browser_state_summary.selector_map

					# Detect index change after previous action
					orig_target = cached_selector_map.get(action.get_index())  # type: ignore
					orig_target_hash = orig_target.hash.branch_path_hash if orig_target else None
					new_target = new_selector_map.get(action.get_index())  # type: ignore
					new_target_hash = new_target.hash.branch_path_hash if new_target else None
					if orig_target_hash != new_target_hash:
						msg = f'Element index changed after action {i} / {len(actions)}, because page changed.'
						logger.info(msg)
						results.append(
							ActionResult(
								extracted_content=msg,
								include_in_memory=True,
								long_term_memory=msg,
							)
						)
						break

					new_path_hashes = {e.hash.branch_path_hash for e in new_selector_map.values()}
					if check_for_new_elements and not new_path_hashes.issubset(cached_path_hashes):
						# next action requires index but there are new elements on the page
						msg = f'Something new appeared after action {i} / {len(actions)}, following actions are NOT executed and should be retried.'
						logger.info(msg)
						results.append(
							ActionResult(
								extracted_content=msg,
								include_in_memory=True,
								long_term_memory=msg,
							)
						)
						break

				# wait between actions
				await asyncio.sleep(self.browser_profile.wait_between_actions)

			try:
				await self._raise_if_stopped_or_paused()

				result = await self.controller.act(
					action=action,
					browser_session=self.browser_session,
					file_system=self.file_system,
					page_extraction_llm=self.settings.page_extraction_llm,
					sensitive_data=self.sensitive_data,
					available_file_paths=self.available_file_paths,
					context=self.context,
				)

				results.append(result)

				# Get action name from the action model
				action_data = action.model_dump(exclude_unset=True)
				action_name = next(iter(action_data.keys())) if action_data else 'unknown'
				action_params = getattr(action, action_name, '')
				self.logger.info(f'☑️ Executed action {i + 1}/{len(actions)}: {action_name}({action_params})')
				if results[-1].is_done or results[-1].error or i == len(actions) - 1:
					break

			except Exception as e:
				# Handle any exceptions during action execution
				self.logger.error(f'Action {i + 1} failed: {type(e).__name__}: {e}')
				raise e

		return results

	async def log_completion(self) -> None:
		"""Log the completion of the task"""
		if self.history.is_successful():
			self.logger.info('✅ Task completed successfully')
		else:
			self.logger.info('❌ Task completed without success')

	async def rerun_history(
		self,
		history: AgentHistoryList,
		max_retries: int = 3,
		skip_failures: bool = True,
		delay_between_actions: float = 2.0,
	) -> list[ActionResult]:
		"""
		Rerun a saved history of actions with error handling and retry logic.

		Args:
		                history: The history to replay
		                max_retries: Maximum number of retries per action
		                skip_failures: Whether to skip failed actions or stop execution
		                delay_between_actions: Delay between actions in seconds

		Returns:
		                List of action results
		"""
		# Execute initial actions if provided
		if self.initial_actions:
			result = await self.multi_act(self.initial_actions)
			self.state.last_result = result

		results = []

		for i, history_item in enumerate(history.history):
			goal = history_item.model_output.current_state.next_goal if history_item.model_output else ''
			self.logger.info(f'Replaying step {i + 1}/{len(history.history)}: goal: {goal}')

			if (
				not history_item.model_output
				or not history_item.model_output.action
				or history_item.model_output.action == [None]
			):
				self.logger.warning(f'Step {i + 1}: No action to replay, skipping')
				results.append(ActionResult(error='No action to replay'))
				continue

			retry_count = 0
			while retry_count < max_retries:
				try:
					result = await self._execute_history_step(history_item, delay_between_actions)
					results.extend(result)
					break

				except Exception as e:
					retry_count += 1
					if retry_count == max_retries:
						error_msg = f'Step {i + 1} failed after {max_retries} attempts: {str(e)}'
						self.logger.error(error_msg)
						if not skip_failures:
							results.append(ActionResult(error=error_msg))
							raise RuntimeError(error_msg)
					else:
						self.logger.warning(f'Step {i + 1} failed (attempt {retry_count}/{max_retries}), retrying...')
						await asyncio.sleep(delay_between_actions)

		return results

	async def _execute_history_step(self, history_item: AgentHistory, delay: float) -> list[ActionResult]:
		"""Execute a single step from history with element validation"""
		assert self.browser_session is not None, 'BrowserSession is not set up'
		state = await self.browser_session.get_browser_state_with_recovery(
			cache_clickable_elements_hashes=False, include_screenshot=False
		)
		if not state or not history_item.model_output:
			raise ValueError('Invalid state or model output')
		updated_actions = []
		for i, action in enumerate(history_item.model_output.action):
			updated_action = await self._update_action_indices(
				history_item.state.interacted_element[i],
				action,
				state,
			)
			updated_actions.append(updated_action)

			if updated_action is None:
				raise ValueError(f'Could not find matching element {i} in current page')

		result = await self.multi_act(updated_actions)

		await asyncio.sleep(delay)
		return result

	async def _update_action_indices(
		self,
		historical_element: DOMHistoryElement | None,
		action: ActionModel,  # Type this properly based on your action model
		browser_state_summary: BrowserStateSummary,
	) -> ActionModel | None:
		"""
		Update action indices based on current page state.
		Returns updated action or None if element cannot be found.
		"""
		if not historical_element or not browser_state_summary.element_tree:
			return action

		current_element = HistoryTreeProcessor.find_history_element_in_tree(
			historical_element, browser_state_summary.element_tree
		)

		if not current_element or current_element.highlight_index is None:
			return None

		old_index = action.get_index()
		if old_index != current_element.highlight_index:
			action.set_index(current_element.highlight_index)
			self.logger.info(f'Element moved in DOM, updated index from {old_index} to {current_element.highlight_index}')

		return action

	async def load_and_rerun(self, history_file: str | Path | None = None, **kwargs) -> list[ActionResult]:
		"""
		Load history from file and rerun it.

		Args:
		                history_file: Path to the history file
		                **kwargs: Additional arguments passed to rerun_history
		"""
		if not history_file:
			history_file = 'AgentHistory.json'
		history = AgentHistoryList.load_from_file(history_file, self.AgentOutput)
		return await self.rerun_history(history, **kwargs)

	def save_history(self, file_path: str | Path | None = None) -> None:
		"""Save the history to a file"""
		if not file_path:
			file_path = 'AgentHistory.json'
		self.history.save_to_file(file_path)

	async def wait_until_resumed(self):
		await self._external_pause_event.wait()

	def pause(self) -> None:
		"""Pause the agent before the next step"""
		print(
			'\n\n⏸️  Got [Ctrl+C], paused the agent and left the browser open.\n\tPress [Enter] to resume or [Ctrl+C] again to quit.'
		)
		self.state.paused = True
		self._external_pause_event.clear()

		# Task paused

		# The signal handler will handle the asyncio pause logic for us
		# No need to duplicate the code here

	def resume(self) -> None:
		"""Resume the agent"""
		print('----------------------------------------------------------------------')
		print('▶️  Got Enter, resuming agent execution where it left off...\n')
		self.state.paused = False
		self._external_pause_event.set()

		# Task resumed

		# The signal handler should have already reset the flags
		# through its reset() method when called from run()

		# playwright browser is always immediately killed by the first Ctrl+C (no way to stop that)
		# so we need to restart the browser if user wants to continue
		# the _init() method exists, even through its shows a linter error
		if self.browser:
			self.logger.info('🌎 Restarting/reconnecting to browser...')
			loop = asyncio.get_event_loop()
			loop.create_task(self.browser._init())  # type: ignore
			loop.create_task(asyncio.sleep(5))

	def stop(self) -> None:
		"""Stop the agent"""
		self.logger.info('⏹️ Agent stopping')
		self.state.stopped = True

		# Task stopped

	def _convert_initial_actions(self, actions: list[dict[str, dict[str, Any]]]) -> list[ActionModel]:
		"""Convert dictionary-based actions to ActionModel instances"""
		converted_actions = []
		action_model = self.ActionModel
		for action_dict in actions:
			# Each action_dict should have a single key-value pair
			action_name = next(iter(action_dict))
			params = action_dict[action_name]

			# Get the parameter model for this action from registry
			action_info = self.controller.registry.registry.actions[action_name]
			param_model = action_info.param_model

			# Create validated parameters using the appropriate param model
			validated_params = param_model(**params)

			# Create ActionModel instance with the validated parameters
			action_model = self.ActionModel(**{action_name: validated_params})
			converted_actions.append(action_model)

		return converted_actions

	def _verify_and_setup_llm(self):
		"""
		Verify that the LLM API keys are setup and the LLM API is responding properly.
		Also handles tool calling method detection if in auto mode.
		"""

		# Skip verification if already done
		if getattr(self.llm, '_verified_api_keys', None) is True or CONFIG.SKIP_LLM_API_KEY_VERIFICATION:
			setattr(self.llm, '_verified_api_keys', True)
			return True

	@property
	def message_manager(self) -> MessageManager:
		return self._message_manager

	async def close(self):
		"""Close all resources"""
		try:
			# First close browser resources
			assert self.browser_session is not None, 'BrowserSession is not set up'
			await self.browser_session.stop()

			# Force garbage collection
			gc.collect()

		except Exception as e:
			self.logger.error(f'Error during cleanup: {e}')

	async def _update_action_models_for_page(self, page) -> None:
		"""Update action models with page-specific actions"""
		# Create new action model with current page's filtered actions
		self.ActionModel = self.controller.registry.create_action_model(page=page)
		# Update output model with the new actions
		if self.settings.flash_mode:
			self.AgentOutput = AgentOutput.type_with_custom_actions_flash_mode(self.ActionModel)
		elif self.settings.use_thinking:
			self.AgentOutput = AgentOutput.type_with_custom_actions(self.ActionModel)
		else:
			self.AgentOutput = AgentOutput.type_with_custom_actions_no_thinking(self.ActionModel)

		# Update done action model too
		self.DoneActionModel = self.controller.registry.create_action_model(include_actions=['done'], page=page)
		if self.settings.flash_mode:
			self.DoneAgentOutput = AgentOutput.type_with_custom_actions_flash_mode(self.DoneActionModel)
		elif self.settings.use_thinking:
			self.DoneAgentOutput = AgentOutput.type_with_custom_actions(self.DoneActionModel)
		else:
			self.DoneAgentOutput = AgentOutput.type_with_custom_actions_no_thinking(self.DoneActionModel)

	def get_trace_object(self) -> dict[str, Any]:
		"""Get the trace and trace_details objects for the agent"""

		def extract_task_website(task_text: str) -> str | None:
			url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+|[^\s<>"\']+\.[a-z]{2,}(?:/[^\s<>"\']*)?'
			match = re.search(url_pattern, task_text, re.IGNORECASE)
			return match.group(0) if match else None

		def _get_complete_history_without_screenshots(history_data: dict[str, Any]) -> str:
			if 'history' in history_data:
				for item in history_data['history']:
					if 'state' in item and 'screenshot' in item['state']:
						item['state']['screenshot'] = None

			return json.dumps(history_data)

		# Generate autogenerated fields
		trace_id = uuid7str()
		timestamp = datetime.now().isoformat()

		# Only declare variables that are used multiple times
		structured_output = self.history.structured_output
		structured_output_json = json.dumps(structured_output.model_dump()) if structured_output else None
		final_result = self.history.final_result()
		git_info = get_git_info()
		action_history = self.history.action_history()
		action_errors = self.history.errors()
		urls = self.history.urls()
		usage = self.history.usage

		return {
			'trace': {
				# Autogenerated fields
				'trace_id': trace_id,
				'timestamp': timestamp,
				'browser_use_version': get_browser_use_version(),
				'git_info': json.dumps(git_info) if git_info else None,
				# Direct agent properties
				'model': self.llm.model,
				'settings': json.dumps(self.settings.model_dump()) if self.settings else None,
				'task_id': self.task_id,
				'task_truncated': self.task[:20000] if len(self.task) > 20000 else self.task,
				'task_website': extract_task_website(self.task),
				# AgentHistoryList methods
				'structured_output_truncated': (
					structured_output_json[:20000]
					if structured_output_json and len(structured_output_json) > 20000
					else structured_output_json
				),
				'action_history_truncated': json.dumps(action_history) if action_history else None,
				'action_errors': json.dumps(action_errors) if action_errors else None,
				'urls': json.dumps(urls) if urls else None,
				'final_result_response_truncated': (
					final_result[:20000] if final_result and len(final_result) > 20000 else final_result
				),
				'self_report_completed': 1 if self.history.is_done() else 0,
				'self_report_success': 1 if self.history.is_successful() else 0,
				'duration': self.history.total_duration_seconds(),
				'steps_taken': self.history.number_of_steps(),
				'usage': json.dumps(usage.model_dump()) if usage else None,
			},
			'trace_details': {
				# Autogenerated fields (ensure same as trace)
				'trace_id': trace_id,
				'timestamp': timestamp,
				# Direct agent properties
				'task': self.task,
				# AgentHistoryList methods
				'structured_output': structured_output_json,
				'final_result_response': final_result,
				'complete_history': _get_complete_history_without_screenshots(self.history.model_dump()),
			},
		}

# From agent/service.py
def log_response(response: AgentOutput, registry=None, logger=None) -> None:
	"""Utility function to log the model's response."""

	# Use module logger if no logger provided
	if logger is None:
		logger = logging.getLogger(__name__)

	# Only log thinking if it's present
	if response.current_state.thinking:
		logger.info(f'💡 Thinking:\n{response.current_state.thinking}')

	# Only log evaluation if it's not empty
	eval_goal = response.current_state.evaluation_previous_goal
	if eval_goal:
		if 'success' in eval_goal.lower():
			emoji = '👍'
		elif 'failure' in eval_goal.lower():
			emoji = '⚠️'
		else:
			emoji = '❔'
		logger.info(f'{emoji} Eval: {eval_goal}')

	# Always log memory if present
	if response.current_state.memory:
		logger.info(f'🧠 Memory: {response.current_state.memory}')

	# Only log next goal if it's not empty
	next_goal = response.current_state.next_goal
	if next_goal:
		logger.info(f'🎯 Next goal: {next_goal}\n')
	else:
		logger.info('')

# From agent/service.py
def logger(self) -> logging.Logger:
		"""Get instance-specific logger with task ID in the name"""

		_browser_session_id = self.browser_session.id if self.browser_session else self.id
		_current_page_id = str(id(self.browser_session and self.browser_session.agent_current_page))[-2:]
		return logging.getLogger(f'browser_use.Agent🅰 {self.task_id[-4:]} on 🆂 {_browser_session_id[-4:]} 🅟 {_current_page_id}')

# From agent/service.py
def browser(self) -> Browser:
		assert self.browser_session is not None, 'BrowserSession is not set up'
		assert self.browser_session.browser is not None, 'Browser is not set up'
		return self.browser_session.browser

# From agent/service.py
def browser_context(self) -> BrowserContext:
		assert self.browser_session is not None, 'BrowserSession is not set up'
		assert self.browser_session.browser_context is not None, 'BrowserContext is not set up'
		return self.browser_session.browser_context

# From agent/service.py
def browser_profile(self) -> BrowserProfile:
		assert self.browser_session is not None, 'BrowserSession is not set up'
		return self.browser_session.browser_profile

# From agent/service.py
def save_file_system_state(self) -> None:
		"""Save current file system state to agent state"""
		if self.file_system:
			self.state.file_system_state = self.file_system.get_state()
		else:
			logger.error('💾 File system is not set up. Cannot save state.')
			raise ValueError('File system is not set up. Cannot save state.')

# From agent/service.py
def add_new_task(self, new_task: str) -> None:
		"""Add a new task to the agent, keeping the same task_id as tasks are continuous"""
		# Simply delegate to message manager - no need for new task_id or events
		# The task continues with new instructions, it doesn't end and start a new one
		self.task = new_task
		self._message_manager.add_new_task(new_task)

# From agent/service.py
def save_history(self, file_path: str | Path | None = None) -> None:
		"""Save the history to a file"""
		if not file_path:
			file_path = 'AgentHistory.json'
		self.history.save_to_file(file_path)

# From agent/service.py
def pause(self) -> None:
		"""Pause the agent before the next step"""
		print(
			'\n\n⏸️  Got [Ctrl+C], paused the agent and left the browser open.\n\tPress [Enter] to resume or [Ctrl+C] again to quit.'
		)
		self.state.paused = True
		self._external_pause_event.clear()

# From agent/service.py
def resume(self) -> None:
		"""Resume the agent"""
		print('----------------------------------------------------------------------')
		print('▶️  Got Enter, resuming agent execution where it left off...\n')
		self.state.paused = False
		self._external_pause_event.set()

		# Task resumed

		# The signal handler should have already reset the flags
		# through its reset() method when called from run()

		# playwright browser is always immediately killed by the first Ctrl+C (no way to stop that)
		# so we need to restart the browser if user wants to continue
		# the _init() method exists, even through its shows a linter error
		if self.browser:
			self.logger.info('🌎 Restarting/reconnecting to browser...')
			loop = asyncio.get_event_loop()
			loop.create_task(self.browser._init())  # type: ignore
			loop.create_task(asyncio.sleep(5))

# From agent/service.py
def stop(self) -> None:
		"""Stop the agent"""
		self.logger.info('⏹️ Agent stopping')
		self.state.stopped = True

# From agent/service.py
def message_manager(self) -> MessageManager:
		return self._message_manager

# From agent/service.py
def get_trace_object(self) -> dict[str, Any]:
		"""Get the trace and trace_details objects for the agent"""

		def extract_task_website(task_text: str) -> str | None:
			url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+|[^\s<>"\']+\.[a-z]{2,}(?:/[^\s<>"\']*)?'
			match = re.search(url_pattern, task_text, re.IGNORECASE)
			return match.group(0) if match else None

		def _get_complete_history_without_screenshots(history_data: dict[str, Any]) -> str:
			if 'history' in history_data:
				for item in history_data['history']:
					if 'state' in item and 'screenshot' in item['state']:
						item['state']['screenshot'] = None

			return json.dumps(history_data)

		# Generate autogenerated fields
		trace_id = uuid7str()
		timestamp = datetime.now().isoformat()

		# Only declare variables that are used multiple times
		structured_output = self.history.structured_output
		structured_output_json = json.dumps(structured_output.model_dump()) if structured_output else None
		final_result = self.history.final_result()
		git_info = get_git_info()
		action_history = self.history.action_history()
		action_errors = self.history.errors()
		urls = self.history.urls()
		usage = self.history.usage

		return {
			'trace': {
				# Autogenerated fields
				'trace_id': trace_id,
				'timestamp': timestamp,
				'browser_use_version': get_browser_use_version(),
				'git_info': json.dumps(git_info) if git_info else None,
				# Direct agent properties
				'model': self.llm.model,
				'settings': json.dumps(self.settings.model_dump()) if self.settings else None,
				'task_id': self.task_id,
				'task_truncated': self.task[:20000] if len(self.task) > 20000 else self.task,
				'task_website': extract_task_website(self.task),
				# AgentHistoryList methods
				'structured_output_truncated': (
					structured_output_json[:20000]
					if structured_output_json and len(structured_output_json) > 20000
					else structured_output_json
				),
				'action_history_truncated': json.dumps(action_history) if action_history else None,
				'action_errors': json.dumps(action_errors) if action_errors else None,
				'urls': json.dumps(urls) if urls else None,
				'final_result_response_truncated': (
					final_result[:20000] if final_result and len(final_result) > 20000 else final_result
				),
				'self_report_completed': 1 if self.history.is_done() else 0,
				'self_report_success': 1 if self.history.is_successful() else 0,
				'duration': self.history.total_duration_seconds(),
				'steps_taken': self.history.number_of_steps(),
				'usage': json.dumps(usage.model_dump()) if usage else None,
			},
			'trace_details': {
				# Autogenerated fields (ensure same as trace)
				'trace_id': trace_id,
				'timestamp': timestamp,
				# Direct agent properties
				'task': self.task,
				# AgentHistoryList methods
				'structured_output': structured_output_json,
				'final_result_response': final_result,
				'complete_history': _get_complete_history_without_screenshots(self.history.model_dump()),
			},
		}

# From agent/service.py
def on_force_exit_log_telemetry():
			self._log_agent_event(max_steps=max_steps, agent_run_error='SIGINT: Cancelled by user')
			# NEW: Call the flush method on the telemetry instance
			if hasattr(self, 'telemetry') and self.telemetry:
				self.telemetry.flush()
			self._force_exit_telemetry_logged = True

# From agent/service.py
def extract_task_website(task_text: str) -> str | None:
			url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+|[^\s<>"\']+\.[a-z]{2,}(?:/[^\s<>"\']*)?'
			match = re.search(url_pattern, task_text, re.IGNORECASE)
			return match.group(0) if match else None

from __future__ import annotations
import io
from browser_use.browser.views import PLACEHOLDER_4PX_SCREENSHOT
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# From agent/gif.py
def decode_unicode_escapes_to_utf8(text: str) -> str:
	"""Handle decoding any unicode escape sequences embedded in a string (needed to render non-ASCII languages like chinese or arabic in the GIF overlay text)"""

	if r'\u' not in text:
		# doesn't have any escape sequences that need to be decoded
		return text

	try:
		# Try to decode Unicode escape sequences
		return text.encode('latin1').decode('unicode_escape')
	except (UnicodeEncodeError, UnicodeDecodeError):
		# logger.debug(f"Failed to decode unicode escape sequences while generating gif text: {text}")
		return text

# From agent/gif.py
def create_history_gif(
	task: str,
	history: AgentHistoryList,
	#
	output_path: str = 'agent_history.gif',
	duration: int = 3000,
	show_goals: bool = True,
	show_task: bool = True,
	show_logo: bool = False,
	font_size: int = 40,
	title_font_size: int = 56,
	goal_font_size: int = 44,
	margin: int = 40,
	line_spacing: float = 1.5,
) -> None:
	"""Create a GIF from the agent's history with overlaid task and goal text."""
	if not history.history:
		logger.warning('No history to create GIF from')
		return

	from PIL import Image, ImageFont

	images = []

	# if history is empty, we can't create a gif
	if not history.history:
		logger.warning('No history to create GIF from')
		return

	# Get all screenshots from history (including None placeholders)
	screenshots = history.screenshots(return_none_if_not_screenshot=True)

	if not screenshots:
		logger.warning('No screenshots found in history')
		return

	# Find the first non-placeholder screenshot
	first_real_screenshot = None
	for screenshot in screenshots:
		if screenshot and screenshot != PLACEHOLDER_4PX_SCREENSHOT:
			first_real_screenshot = screenshot
			break

	if not first_real_screenshot:
		logger.warning('No valid screenshots found (all are placeholders)')
		return

	# Try to load nicer fonts
	try:
		# Try different font options in order of preference
		# ArialUni is a font that comes with Office and can render most non-alphabet characters
		font_options = [
			'Microsoft YaHei',  # 微软雅黑
			'SimHei',  # 黑体
			'SimSun',  # 宋体
			'Noto Sans CJK SC',  # 思源黑体
			'WenQuanYi Micro Hei',  # 文泉驿微米黑
			'Helvetica',
			'Arial',
			'DejaVuSans',
			'Verdana',
		]
		font_loaded = False

		for font_name in font_options:
			try:
				if platform.system() == 'Windows':
					# Need to specify the abs font path on Windows
					font_name = os.path.join(CONFIG.WIN_FONT_DIR, font_name + '.ttf')
				regular_font = ImageFont.truetype(font_name, font_size)
				title_font = ImageFont.truetype(font_name, title_font_size)
				goal_font = ImageFont.truetype(font_name, goal_font_size)
				font_loaded = True
				break
			except OSError:
				continue

		if not font_loaded:
			raise OSError('No preferred fonts found')

	except OSError:
		regular_font = ImageFont.load_default()
		title_font = ImageFont.load_default()

		goal_font = regular_font

	# Load logo if requested
	logo = None
	if show_logo:
		try:
			logo = Image.open('./static/browser-use.png')
			# Resize logo to be small (e.g., 40px height)
			logo_height = 150
			aspect_ratio = logo.width / logo.height
			logo_width = int(logo_height * aspect_ratio)
			logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
		except Exception as e:
			logger.warning(f'Could not load logo: {e}')

	# Create task frame if requested
	if show_task and task:
		# Find the first non-placeholder screenshot for the task frame
		first_real_screenshot = None
		for item in history.history:
			screenshot_b64 = item.state.get_screenshot()
			if screenshot_b64 and screenshot_b64 != PLACEHOLDER_4PX_SCREENSHOT:
				first_real_screenshot = screenshot_b64
				break

		if first_real_screenshot:
			task_frame = _create_task_frame(
				task,
				first_real_screenshot,
				title_font,  # type: ignore
				regular_font,  # type: ignore
				logo,
				line_spacing,
			)
			images.append(task_frame)
		else:
			logger.warning('No real screenshots found for task frame, skipping task frame')

	# Process each history item with its corresponding screenshot
	for i, (item, screenshot) in enumerate(zip(history.history, screenshots), 1):
		if not screenshot:
			continue

		# Skip placeholder screenshots from about:blank pages
		# These are 4x4 white PNGs encoded as a specific base64 string
		if screenshot == PLACEHOLDER_4PX_SCREENSHOT:
			logger.debug(f'Skipping placeholder screenshot from about:blank page at step {i}')
			continue

		# Convert base64 screenshot to PIL Image
		img_data = base64.b64decode(screenshot)
		image = Image.open(io.BytesIO(img_data))

		if show_goals and item.model_output:
			image = _add_overlay_to_image(
				image=image,
				step_number=i,
				goal_text=item.model_output.current_state.next_goal,
				regular_font=regular_font,  # type: ignore
				title_font=title_font,  # type: ignore
				margin=margin,
				logo=logo,
			)

		images.append(image)

	if images:
		# Save the GIF
		images[0].save(
			output_path,
			save_all=True,
			append_images=images[1:],
			duration=duration,
			loop=0,
			optimize=False,
		)
		logger.info(f'Created GIF at {output_path}')
	else:
		logger.warning('No images found in history to create GIF')

from pydantic import model_validator
from typing_extensions import TypeVar
from browser_use.agent.message_manager.views import MessageManagerState
from browser_use.browser.views import BrowserStateHistory
from browser_use.dom.history_tree_processor.service import DOMElementNode
from browser_use.filesystem.file_system import FileSystemState

# From agent/views.py
class AgentSettings(BaseModel):
	"""Configuration options for the Agent"""

	use_vision: bool = True
	vision_detail_level: Literal['auto', 'low', 'high'] = 'auto'
	use_vision_for_planner: bool = False
	save_conversation_path: str | Path | None = None
	save_conversation_path_encoding: str | None = 'utf-8'
	max_failures: int = 3
	retry_delay: int = 10
	validate_output: bool = False
	generate_gif: bool | str = False
	override_system_message: str | None = None
	extend_system_message: str | None = None
	include_attributes: list[str] = [
		'title',
		'type',
		'name',
		'role',
		'tabindex',
		'aria-label',
		'placeholder',
		'value',
		'alt',
		'aria-expanded',
	]
	max_actions_per_step: int = 10
	use_thinking: bool = True
	flash_mode: bool = False  # If enabled, disables evaluation_previous_goal and next_goal, and sets use_thinking = False
	max_history_items: int | None = None

	page_extraction_llm: BaseChatModel | None = None
	planner_llm: BaseChatModel | None = None
	planner_interval: int = 1  # Run planner every N steps
	is_planner_reasoning: bool = False  # type: ignore
	extend_planner_system_message: str | None = None
	calculate_cost: bool = False
	include_tool_call_examples: bool = False
	llm_timeout: int = 60  # Timeout in seconds for LLM calls
	step_timeout: int = 180

# From agent/views.py
class AgentState(BaseModel):
	"""Holds all state information for an Agent"""

	agent_id: str = Field(default_factory=uuid7str)
	n_steps: int = 1
	consecutive_failures: int = 0
	last_result: list[ActionResult] | None = None
	last_plan: str | None = None
	last_model_output: AgentOutput | None = None
	paused: bool = False
	stopped: bool = False

	message_manager_state: MessageManagerState = Field(default_factory=MessageManagerState)
	file_system_state: FileSystemState | None = None

# From agent/views.py
class AgentStepInfo:
	step_number: int
	max_steps: int

	def is_last_step(self) -> bool:
		"""Check if this is the last step"""
		return self.step_number >= self.max_steps - 1

# From agent/views.py
class ActionResult(BaseModel):
	"""Result of executing an action"""

	# For done action
	is_done: bool | None = False
	success: bool | None = None

	# Error handling - always include in long term memory
	error: str | None = None

	# Files
	attachments: list[str] | None = None  # Files to display in the done message

	# Always include in long term memory
	long_term_memory: str | None = None  # Memory of this action

	# if update_only_read_state is True we add the extracted_content to the agent context only once for the next step
	# if update_only_read_state is False we add the extracted_content to the agent long term memory if no long_term_memory is provided
	extracted_content: str | None = None
	include_extracted_content_only_once: bool = False  # Whether the extracted content should be used to update the read_state

	# Deprecated
	include_in_memory: bool = False  # whether to include in extracted_content inside long_term_memory

	@model_validator(mode='after')
	def validate_success_requires_done(self):
		"""Ensure success=True can only be set when is_done=True"""
		if self.success is True and self.is_done is not True:
			raise ValueError(
				'success=True can only be set when is_done=True. '
				'For regular actions that succeed, leave success as None. '
				'Use success=False only for actions that fail.'
			)
		return self

# From agent/views.py
class StepMetadata(BaseModel):
	"""Metadata for a single step including timing and token information"""

	step_start_time: float
	step_end_time: float
	step_number: int

	@property
	def duration_seconds(self) -> float:
		"""Calculate step duration in seconds"""
		return self.step_end_time - self.step_start_time

# From agent/views.py
class AgentBrain(BaseModel):
	thinking: str | None = None
	evaluation_previous_goal: str
	memory: str
	next_goal: str

# From agent/views.py
class AgentOutput(BaseModel):
	model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

	thinking: str | None = None
	evaluation_previous_goal: str | None = None
	memory: str | None = None
	next_goal: str | None = None
	action: list[ActionModel] = Field(
		...,
		description='List of actions to execute',
		json_schema_extra={'min_items': 1},  # Ensure at least one action is provided
	)

	@classmethod
	def model_json_schema(cls, **kwargs):
		schema = super().model_json_schema(**kwargs)
		schema['required'] = ['evaluation_previous_goal', 'memory', 'next_goal', 'action']
		return schema

	@property
	def current_state(self) -> AgentBrain:
		"""For backward compatibility - returns an AgentBrain with the flattened properties"""
		return AgentBrain(
			thinking=self.thinking,
			evaluation_previous_goal=self.evaluation_previous_goal if self.evaluation_previous_goal else '',
			memory=self.memory if self.memory else '',
			next_goal=self.next_goal if self.next_goal else '',
		)

	@staticmethod
	def type_with_custom_actions(custom_actions: type[ActionModel]) -> type[AgentOutput]:
		"""Extend actions with custom actions"""

		model_ = create_model(
			'AgentOutput',
			__base__=AgentOutput,
			action=(
				list[custom_actions],  # type: ignore
				Field(..., description='List of actions to execute', json_schema_extra={'min_items': 1}),
			),
			__module__=AgentOutput.__module__,
		)
		model_.__doc__ = 'AgentOutput model with custom actions'
		return model_

	@staticmethod
	def type_with_custom_actions_no_thinking(custom_actions: type[ActionModel]) -> type[AgentOutput]:
		"""Extend actions with custom actions and exclude thinking field"""

		class AgentOutputNoThinking(AgentOutput):
			@classmethod
			def model_json_schema(cls, **kwargs):
				schema = super().model_json_schema(**kwargs)
				del schema['properties']['thinking']
				schema['required'] = ['evaluation_previous_goal', 'memory', 'next_goal', 'action']
				return schema

		model = create_model(
			'AgentOutput',
			__base__=AgentOutputNoThinking,
			action=(
				list[custom_actions],  # type: ignore
				Field(..., description='List of actions to execute', json_schema_extra={'min_items': 1}),
			),
			__module__=AgentOutputNoThinking.__module__,
		)

		model.__doc__ = 'AgentOutput model with custom actions'
		return model

	@staticmethod
	def type_with_custom_actions_flash_mode(custom_actions: type[ActionModel]) -> type[AgentOutput]:
		"""Extend actions with custom actions for flash mode - memory and action fields only"""

		class AgentOutputFlashMode(AgentOutput):
			@classmethod
			def model_json_schema(cls, **kwargs):
				schema = super().model_json_schema(**kwargs)
				# Remove thinking, evaluation_previous_goal, and next_goal fields
				del schema['properties']['thinking']
				del schema['properties']['evaluation_previous_goal']
				del schema['properties']['next_goal']
				# Update required fields to only include remaining properties
				schema['required'] = ['memory', 'action']
				return schema

		model = create_model(
			'AgentOutput',
			__base__=AgentOutputFlashMode,
			action=(
				list[custom_actions],  # type: ignore
				Field(..., description='List of actions to execute', json_schema_extra={'min_items': 1}),
			),
			__module__=AgentOutputFlashMode.__module__,
		)

		model.__doc__ = 'AgentOutput model with custom actions'
		return model

# From agent/views.py
class AgentHistory(BaseModel):
	"""History item for agent actions"""

	model_output: AgentOutput | None
	result: list[ActionResult]
	state: BrowserStateHistory
	metadata: StepMetadata | None = None

	model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

	@staticmethod
	def get_interacted_element(model_output: AgentOutput, selector_map: SelectorMap) -> list[DOMHistoryElement | None]:
		elements = []
		for action in model_output.action:
			index = action.get_index()
			if index is not None and index in selector_map:
				el: DOMElementNode = selector_map[index]
				elements.append(HistoryTreeProcessor.convert_dom_element_to_history_element(el))
			else:
				elements.append(None)
		return elements

	def model_dump(self, **kwargs) -> dict[str, Any]:
		"""Custom serialization handling circular references"""

		# Handle action serialization
		model_output_dump = None
		if self.model_output:
			action_dump = [action.model_dump(exclude_none=True) for action in self.model_output.action]
			model_output_dump = {
				'evaluation_previous_goal': self.model_output.evaluation_previous_goal,
				'memory': self.model_output.memory,
				'next_goal': self.model_output.next_goal,
				'action': action_dump,  # This preserves the actual action data
			}
			# Only include thinking if it's present
			if self.model_output.thinking is not None:
				model_output_dump['thinking'] = self.model_output.thinking

		return {
			'model_output': model_output_dump,
			'result': [r.model_dump(exclude_none=True) for r in self.result],
			'state': self.state.to_dict(),
			'metadata': self.metadata.model_dump() if self.metadata else None,
		}

# From agent/views.py
class AgentHistoryList(BaseModel, Generic[AgentStructuredOutput]):
	"""List of AgentHistory messages, i.e. the history of the agent's actions and thoughts."""

	history: list[AgentHistory]
	usage: UsageSummary | None = None

	_output_model_schema: type[AgentStructuredOutput] | None = None

	def total_duration_seconds(self) -> float:
		"""Get total duration of all steps in seconds"""
		total = 0.0
		for h in self.history:
			if h.metadata:
				total += h.metadata.duration_seconds
		return total

	def __len__(self) -> int:
		"""Return the number of history items"""
		return len(self.history)

	def __str__(self) -> str:
		"""Representation of the AgentHistoryList object"""
		return f'AgentHistoryList(all_results={self.action_results()}, all_model_outputs={self.model_actions()})'

	def add_item(self, history_item: AgentHistory) -> None:
		"""Add a history item to the list"""
		self.history.append(history_item)

	def __repr__(self) -> str:
		"""Representation of the AgentHistoryList object"""
		return self.__str__()

	def save_to_file(self, filepath: str | Path) -> None:
		"""Save history to JSON file with proper serialization"""
		try:
			Path(filepath).parent.mkdir(parents=True, exist_ok=True)
			data = self.model_dump()
			with open(filepath, 'w', encoding='utf-8') as f:
				json.dump(data, f, indent=2)
		except Exception as e:
			raise e

	# def save_as_playwright_script(
	# 	self,
	# 	output_path: str | Path,
	# 	sensitive_data_keys: list[str] | None = None,
	# 	browser_config: BrowserConfig | None = None,
	# 	context_config: BrowserContextConfig | None = None,
	# ) -> None:
	# 	"""
	# 	Generates a Playwright script based on the agent's history and saves it to a file.
	# 	Args:
	# 		output_path: The path where the generated Python script will be saved.
	# 		sensitive_data_keys: A list of keys used as placeholders for sensitive data
	# 							 (e.g., ['username_placeholder', 'password_placeholder']).
	# 							 These will be loaded from environment variables in the
	# 							 generated script.
	# 		browser_config: Configuration of the original Browser instance.
	# 		context_config: Configuration of the original BrowserContext instance.
	# 	"""
	# 	from browser_use.agent.playwright_script_generator import PlaywrightScriptGenerator

	# 	try:
	# 		serialized_history = self.model_dump()['history']
	# 		generator = PlaywrightScriptGenerator(serialized_history, sensitive_data_keys, browser_config, context_config)

	# 		script_content = generator.generate_script_content()
	# 		path_obj = Path(output_path)
	# 		path_obj.parent.mkdir(parents=True, exist_ok=True)
	# 		with open(path_obj, 'w', encoding='utf-8') as f:
	# 			f.write(script_content)
	# 	except Exception as e:
	# 		raise e

	def model_dump(self, **kwargs) -> dict[str, Any]:
		"""Custom serialization that properly uses AgentHistory's model_dump"""
		return {
			'history': [h.model_dump(**kwargs) for h in self.history],
		}

	@classmethod
	def load_from_file(cls, filepath: str | Path, output_model: type[AgentOutput]) -> AgentHistoryList:
		"""Load history from JSON file"""
		with open(filepath, encoding='utf-8') as f:
			data = json.load(f)
		# loop through history and validate output_model actions to enrich with custom actions
		for h in data['history']:
			if h['model_output']:
				if isinstance(h['model_output'], dict):
					h['model_output'] = output_model.model_validate(h['model_output'])
				else:
					h['model_output'] = None
			if 'interacted_element' not in h['state']:
				h['state']['interacted_element'] = None
		history = cls.model_validate(data)
		return history

	def last_action(self) -> None | dict:
		"""Last action in history"""
		if self.history and self.history[-1].model_output:
			return self.history[-1].model_output.action[-1].model_dump(exclude_none=True)
		return None

	def errors(self) -> list[str | None]:
		"""Get all errors from history, with None for steps without errors"""
		errors = []
		for h in self.history:
			step_errors = [r.error for r in h.result if r.error]

			# each step can have only one error
			errors.append(step_errors[0] if step_errors else None)
		return errors

	def final_result(self) -> None | str:
		"""Final result from history"""
		if self.history and self.history[-1].result[-1].extracted_content:
			return self.history[-1].result[-1].extracted_content
		return None

	def is_done(self) -> bool:
		"""Check if the agent is done"""
		if self.history and len(self.history[-1].result) > 0:
			last_result = self.history[-1].result[-1]
			return last_result.is_done is True
		return False

	def is_successful(self) -> bool | None:
		"""Check if the agent completed successfully - the agent decides in the last step if it was successful or not. None if not done yet."""
		if self.history and len(self.history[-1].result) > 0:
			last_result = self.history[-1].result[-1]
			if last_result.is_done is True:
				return last_result.success
		return None

	def has_errors(self) -> bool:
		"""Check if the agent has any non-None errors"""
		return any(error is not None for error in self.errors())

	def urls(self) -> list[str | None]:
		"""Get all unique URLs from history"""
		return [h.state.url if h.state.url is not None else None for h in self.history]

	def screenshot_paths(self, n_last: int | None = None, return_none_if_not_screenshot: bool = True) -> list[str | None]:
		"""Get all screenshot paths from history"""
		if n_last == 0:
			return []
		if n_last is None:
			if return_none_if_not_screenshot:
				return [h.state.screenshot_path if h.state.screenshot_path is not None else None for h in self.history]
			else:
				return [h.state.screenshot_path for h in self.history if h.state.screenshot_path is not None]
		else:
			if return_none_if_not_screenshot:
				return [h.state.screenshot_path if h.state.screenshot_path is not None else None for h in self.history[-n_last:]]
			else:
				return [h.state.screenshot_path for h in self.history[-n_last:] if h.state.screenshot_path is not None]

	def screenshots(self, n_last: int | None = None, return_none_if_not_screenshot: bool = True) -> list[str | None]:
		"""Get all screenshots from history as base64 strings"""
		if n_last == 0:
			return []

		history_items = self.history if n_last is None else self.history[-n_last:]
		screenshots = []

		for item in history_items:
			screenshot_b64 = item.state.get_screenshot()
			if screenshot_b64:
				screenshots.append(screenshot_b64)
			else:
				if return_none_if_not_screenshot:
					screenshots.append(None)
				# If return_none_if_not_screenshot is False, we skip None values

		return screenshots

	def action_names(self) -> list[str]:
		"""Get all action names from history"""
		action_names = []
		for action in self.model_actions():
			actions = list(action.keys())
			if actions:
				action_names.append(actions[0])
		return action_names

	def model_thoughts(self) -> list[AgentBrain]:
		"""Get all thoughts from history"""
		return [h.model_output.current_state for h in self.history if h.model_output]

	def model_outputs(self) -> list[AgentOutput]:
		"""Get all model outputs from history"""
		return [h.model_output for h in self.history if h.model_output]

	# get all actions with params
	def model_actions(self) -> list[dict]:
		"""Get all actions from history"""
		outputs = []

		for h in self.history:
			if h.model_output:
				# Guard against None interacted_element before zipping
				interacted_elements = h.state.interacted_element or [None] * len(h.model_output.action)
				for action, interacted_element in zip(h.model_output.action, interacted_elements):
					output = action.model_dump(exclude_none=True)
					output['interacted_element'] = interacted_element
					outputs.append(output)
		return outputs

	def action_history(self) -> list[list[dict]]:
		"""Get truncated action history with only essential fields"""
		step_outputs = []

		for h in self.history:
			step_actions = []
			if h.model_output:
				# Guard against None interacted_element before zipping
				interacted_elements = h.state.interacted_element or [None] * len(h.model_output.action)
				# Zip actions with interacted elements and results
				for action, interacted_element, result in zip(h.model_output.action, interacted_elements, h.result):
					action_output = action.model_dump(exclude_none=True)
					action_output['interacted_element'] = interacted_element
					# Only keep long_term_memory from result
					action_output['result'] = result.long_term_memory if result and result.long_term_memory else None
					step_actions.append(action_output)
			step_outputs.append(step_actions)

		return step_outputs

	def action_results(self) -> list[ActionResult]:
		"""Get all results from history"""
		results = []
		for h in self.history:
			results.extend([r for r in h.result if r])
		return results

	def extracted_content(self) -> list[str]:
		"""Get all extracted content from history"""
		content = []
		for h in self.history:
			content.extend([r.extracted_content for r in h.result if r.extracted_content])
		return content

	def model_actions_filtered(self, include: list[str] | None = None) -> list[dict]:
		"""Get all model actions from history as JSON"""
		if include is None:
			include = []
		outputs = self.model_actions()
		result = []
		for o in outputs:
			for i in include:
				if i == list(o.keys())[0]:
					result.append(o)
		return result

	def number_of_steps(self) -> int:
		"""Get the number of steps in the history"""
		return len(self.history)

	@property
	def structured_output(self) -> AgentStructuredOutput | None:
		"""Get the structured output from the history

		Returns:
			The structured output if both final_result and _output_model_schema are available,
			otherwise None
		"""
		final_result = self.final_result()
		if final_result is not None and self._output_model_schema is not None:
			return self._output_model_schema.model_validate_json(final_result)

		return None

# From agent/views.py
class AgentError:
	"""Container for agent error handling"""

	VALIDATION_ERROR = 'Invalid model output format. Please follow the correct schema.'
	RATE_LIMIT_ERROR = 'Rate limit reached. Waiting before retry.'
	NO_VALID_ACTION = 'No valid action found'

	@staticmethod
	def format_error(error: Exception, include_trace: bool = False) -> str:
		"""Format error message based on error type and optionally include trace"""
		message = ''
		if isinstance(error, ValidationError):
			return f'{AgentError.VALIDATION_ERROR}\nDetails: {str(error)}'
		if isinstance(error, RateLimitError):
			return AgentError.RATE_LIMIT_ERROR
		if include_trace:
			return f'{str(error)}\nStacktrace:\n{traceback.format_exc()}'
		return f'{str(error)}'

# From agent/views.py
class AgentOutputNoThinking(AgentOutput):
			@classmethod
			def model_json_schema(cls, **kwargs):
				schema = super().model_json_schema(**kwargs)
				del schema['properties']['thinking']
				schema['required'] = ['evaluation_previous_goal', 'memory', 'next_goal', 'action']
				return schema

# From agent/views.py
class AgentOutputFlashMode(AgentOutput):
			@classmethod
			def model_json_schema(cls, **kwargs):
				schema = super().model_json_schema(**kwargs)
				# Remove thinking, evaluation_previous_goal, and next_goal fields
				del schema['properties']['thinking']
				del schema['properties']['evaluation_previous_goal']
				del schema['properties']['next_goal']
				# Update required fields to only include remaining properties
				schema['required'] = ['memory', 'action']
				return schema

# From agent/views.py
def is_last_step(self) -> bool:
		"""Check if this is the last step"""
		return self.step_number >= self.max_steps - 1

# From agent/views.py
def validate_success_requires_done(self):
		"""Ensure success=True can only be set when is_done=True"""
		if self.success is True and self.is_done is not True:
			raise ValueError(
				'success=True can only be set when is_done=True. '
				'For regular actions that succeed, leave success as None. '
				'Use success=False only for actions that fail.'
			)
		return self

# From agent/views.py
def duration_seconds(self) -> float:
		"""Calculate step duration in seconds"""
		return self.step_end_time - self.step_start_time

# From agent/views.py
def model_json_schema(cls, **kwargs):
		schema = super().model_json_schema(**kwargs)
		schema['required'] = ['evaluation_previous_goal', 'memory', 'next_goal', 'action']
		return schema

# From agent/views.py
def current_state(self) -> AgentBrain:
		"""For backward compatibility - returns an AgentBrain with the flattened properties"""
		return AgentBrain(
			thinking=self.thinking,
			evaluation_previous_goal=self.evaluation_previous_goal if self.evaluation_previous_goal else '',
			memory=self.memory if self.memory else '',
			next_goal=self.next_goal if self.next_goal else '',
		)

# From agent/views.py
def type_with_custom_actions(custom_actions: type[ActionModel]) -> type[AgentOutput]:
		"""Extend actions with custom actions"""

		model_ = create_model(
			'AgentOutput',
			__base__=AgentOutput,
			action=(
				list[custom_actions],  # type: ignore
				Field(..., description='List of actions to execute', json_schema_extra={'min_items': 1}),
			),
			__module__=AgentOutput.__module__,
		)
		model_.__doc__ = 'AgentOutput model with custom actions'
		return model_

# From agent/views.py
def type_with_custom_actions_no_thinking(custom_actions: type[ActionModel]) -> type[AgentOutput]:
		"""Extend actions with custom actions and exclude thinking field"""

		class AgentOutputNoThinking(AgentOutput):
			@classmethod
			def model_json_schema(cls, **kwargs):
				schema = super().model_json_schema(**kwargs)
				del schema['properties']['thinking']
				schema['required'] = ['evaluation_previous_goal', 'memory', 'next_goal', 'action']
				return schema

		model = create_model(
			'AgentOutput',
			__base__=AgentOutputNoThinking,
			action=(
				list[custom_actions],  # type: ignore
				Field(..., description='List of actions to execute', json_schema_extra={'min_items': 1}),
			),
			__module__=AgentOutputNoThinking.__module__,
		)

		model.__doc__ = 'AgentOutput model with custom actions'
		return model

# From agent/views.py
def type_with_custom_actions_flash_mode(custom_actions: type[ActionModel]) -> type[AgentOutput]:
		"""Extend actions with custom actions for flash mode - memory and action fields only"""

		class AgentOutputFlashMode(AgentOutput):
			@classmethod
			def model_json_schema(cls, **kwargs):
				schema = super().model_json_schema(**kwargs)
				# Remove thinking, evaluation_previous_goal, and next_goal fields
				del schema['properties']['thinking']
				del schema['properties']['evaluation_previous_goal']
				del schema['properties']['next_goal']
				# Update required fields to only include remaining properties
				schema['required'] = ['memory', 'action']
				return schema

		model = create_model(
			'AgentOutput',
			__base__=AgentOutputFlashMode,
			action=(
				list[custom_actions],  # type: ignore
				Field(..., description='List of actions to execute', json_schema_extra={'min_items': 1}),
			),
			__module__=AgentOutputFlashMode.__module__,
		)

		model.__doc__ = 'AgentOutput model with custom actions'
		return model

# From agent/views.py
def get_interacted_element(model_output: AgentOutput, selector_map: SelectorMap) -> list[DOMHistoryElement | None]:
		elements = []
		for action in model_output.action:
			index = action.get_index()
			if index is not None and index in selector_map:
				el: DOMElementNode = selector_map[index]
				elements.append(HistoryTreeProcessor.convert_dom_element_to_history_element(el))
			else:
				elements.append(None)
		return elements

# From agent/views.py
def model_dump(self, **kwargs) -> dict[str, Any]:
		"""Custom serialization handling circular references"""

		# Handle action serialization
		model_output_dump = None
		if self.model_output:
			action_dump = [action.model_dump(exclude_none=True) for action in self.model_output.action]
			model_output_dump = {
				'evaluation_previous_goal': self.model_output.evaluation_previous_goal,
				'memory': self.model_output.memory,
				'next_goal': self.model_output.next_goal,
				'action': action_dump,  # This preserves the actual action data
			}
			# Only include thinking if it's present
			if self.model_output.thinking is not None:
				model_output_dump['thinking'] = self.model_output.thinking

		return {
			'model_output': model_output_dump,
			'result': [r.model_dump(exclude_none=True) for r in self.result],
			'state': self.state.to_dict(),
			'metadata': self.metadata.model_dump() if self.metadata else None,
		}

# From agent/views.py
def total_duration_seconds(self) -> float:
		"""Get total duration of all steps in seconds"""
		total = 0.0
		for h in self.history:
			if h.metadata:
				total += h.metadata.duration_seconds
		return total

# From agent/views.py
def add_item(self, history_item: AgentHistory) -> None:
		"""Add a history item to the list"""
		self.history.append(history_item)

# From agent/views.py
def last_action(self) -> None | dict:
		"""Last action in history"""
		if self.history and self.history[-1].model_output:
			return self.history[-1].model_output.action[-1].model_dump(exclude_none=True)
		return None

# From agent/views.py
def errors(self) -> list[str | None]:
		"""Get all errors from history, with None for steps without errors"""
		errors = []
		for h in self.history:
			step_errors = [r.error for r in h.result if r.error]

			# each step can have only one error
			errors.append(step_errors[0] if step_errors else None)
		return errors

# From agent/views.py
def final_result(self) -> None | str:
		"""Final result from history"""
		if self.history and self.history[-1].result[-1].extracted_content:
			return self.history[-1].result[-1].extracted_content
		return None

# From agent/views.py
def is_done(self) -> bool:
		"""Check if the agent is done"""
		if self.history and len(self.history[-1].result) > 0:
			last_result = self.history[-1].result[-1]
			return last_result.is_done is True
		return False

# From agent/views.py
def is_successful(self) -> bool | None:
		"""Check if the agent completed successfully - the agent decides in the last step if it was successful or not. None if not done yet."""
		if self.history and len(self.history[-1].result) > 0:
			last_result = self.history[-1].result[-1]
			if last_result.is_done is True:
				return last_result.success
		return None

# From agent/views.py
def has_errors(self) -> bool:
		"""Check if the agent has any non-None errors"""
		return any(error is not None for error in self.errors())

# From agent/views.py
def urls(self) -> list[str | None]:
		"""Get all unique URLs from history"""
		return [h.state.url if h.state.url is not None else None for h in self.history]

# From agent/views.py
def screenshot_paths(self, n_last: int | None = None, return_none_if_not_screenshot: bool = True) -> list[str | None]:
		"""Get all screenshot paths from history"""
		if n_last == 0:
			return []
		if n_last is None:
			if return_none_if_not_screenshot:
				return [h.state.screenshot_path if h.state.screenshot_path is not None else None for h in self.history]
			else:
				return [h.state.screenshot_path for h in self.history if h.state.screenshot_path is not None]
		else:
			if return_none_if_not_screenshot:
				return [h.state.screenshot_path if h.state.screenshot_path is not None else None for h in self.history[-n_last:]]
			else:
				return [h.state.screenshot_path for h in self.history[-n_last:] if h.state.screenshot_path is not None]

# From agent/views.py
def screenshots(self, n_last: int | None = None, return_none_if_not_screenshot: bool = True) -> list[str | None]:
		"""Get all screenshots from history as base64 strings"""
		if n_last == 0:
			return []

		history_items = self.history if n_last is None else self.history[-n_last:]
		screenshots = []

		for item in history_items:
			screenshot_b64 = item.state.get_screenshot()
			if screenshot_b64:
				screenshots.append(screenshot_b64)
			else:
				if return_none_if_not_screenshot:
					screenshots.append(None)
				# If return_none_if_not_screenshot is False, we skip None values

		return screenshots

# From agent/views.py
def action_names(self) -> list[str]:
		"""Get all action names from history"""
		action_names = []
		for action in self.model_actions():
			actions = list(action.keys())
			if actions:
				action_names.append(actions[0])
		return action_names

# From agent/views.py
def model_thoughts(self) -> list[AgentBrain]:
		"""Get all thoughts from history"""
		return [h.model_output.current_state for h in self.history if h.model_output]

# From agent/views.py
def model_outputs(self) -> list[AgentOutput]:
		"""Get all model outputs from history"""
		return [h.model_output for h in self.history if h.model_output]

# From agent/views.py
def model_actions(self) -> list[dict]:
		"""Get all actions from history"""
		outputs = []

		for h in self.history:
			if h.model_output:
				# Guard against None interacted_element before zipping
				interacted_elements = h.state.interacted_element or [None] * len(h.model_output.action)
				for action, interacted_element in zip(h.model_output.action, interacted_elements):
					output = action.model_dump(exclude_none=True)
					output['interacted_element'] = interacted_element
					outputs.append(output)
		return outputs

# From agent/views.py
def action_history(self) -> list[list[dict]]:
		"""Get truncated action history with only essential fields"""
		step_outputs = []

		for h in self.history:
			step_actions = []
			if h.model_output:
				# Guard against None interacted_element before zipping
				interacted_elements = h.state.interacted_element or [None] * len(h.model_output.action)
				# Zip actions with interacted elements and results
				for action, interacted_element, result in zip(h.model_output.action, interacted_elements, h.result):
					action_output = action.model_dump(exclude_none=True)
					action_output['interacted_element'] = interacted_element
					# Only keep long_term_memory from result
					action_output['result'] = result.long_term_memory if result and result.long_term_memory else None
					step_actions.append(action_output)
			step_outputs.append(step_actions)

		return step_outputs

# From agent/views.py
def action_results(self) -> list[ActionResult]:
		"""Get all results from history"""
		results = []
		for h in self.history:
			results.extend([r for r in h.result if r])
		return results

# From agent/views.py
def extracted_content(self) -> list[str]:
		"""Get all extracted content from history"""
		content = []
		for h in self.history:
			content.extend([r.extracted_content for r in h.result if r.extracted_content])
		return content

# From agent/views.py
def model_actions_filtered(self, include: list[str] | None = None) -> list[dict]:
		"""Get all model actions from history as JSON"""
		if include is None:
			include = []
		outputs = self.model_actions()
		result = []
		for o in outputs:
			for i in include:
				if i == list(o.keys())[0]:
					result.append(o)
		return result

# From agent/views.py
def number_of_steps(self) -> int:
		"""Get the number of steps in the history"""
		return len(self.history)

# From agent/views.py
def structured_output(self) -> AgentStructuredOutput | None:
		"""Get the structured output from the history

		Returns:
			The structured output if both final_result and _output_model_schema are available,
			otherwise None
		"""
		final_result = self.final_result()
		if final_result is not None and self._output_model_schema is not None:
			return self._output_model_schema.model_validate_json(final_result)

		return None

# From agent/views.py
def format_error(error: Exception, include_trace: bool = False) -> str:
		"""Format error message based on error type and optionally include trace"""
		message = ''
		if isinstance(error, ValidationError):
			return f'{AgentError.VALIDATION_ERROR}\nDetails: {str(error)}'
		if isinstance(error, RateLimitError):
			return AgentError.RATE_LIMIT_ERROR
		if include_trace:
			return f'{str(error)}\nStacktrace:\n{traceback.format_exc()}'
		return f'{str(error)}'

from posthog import Posthog
from browser_use.telemetry.views import BaseTelemetryEvent
from browser_use.utils import singleton

# From telemetry/service.py
class ProductTelemetry:
	"""
	Service for capturing anonymized telemetry data.

	If the environment variable `ANONYMIZED_TELEMETRY=False`, anonymized telemetry will be disabled.
	"""

	USER_ID_PATH = str(CONFIG.BROWSER_USE_CONFIG_DIR / 'device_id')
	PROJECT_API_KEY = 'phc_F8JMNjW1i2KbGUTaW1unnDdLSPCoyc52SGRU0JecaUh'
	HOST = 'https://eu.i.posthog.com'
	UNKNOWN_USER_ID = 'UNKNOWN'

	_curr_user_id = None

	def __init__(self) -> None:
		telemetry_disabled = not CONFIG.ANONYMIZED_TELEMETRY
		self.debug_logging = CONFIG.BROWSER_USE_LOGGING_LEVEL == 'debug'

		if telemetry_disabled:
			self._posthog_client = None
		else:
			logger.info(
				'Anonymized telemetry enabled. See https://docs.browser-use.com/development/telemetry for more information.'
			)
			self._posthog_client = Posthog(
				project_api_key=self.PROJECT_API_KEY,
				host=self.HOST,
				disable_geoip=False,
				enable_exception_autocapture=True,
			)

			# Silence posthog's logging
			if not self.debug_logging:
				posthog_logger = logging.getLogger('posthog')
				posthog_logger.disabled = True

		if self._posthog_client is None:
			logger.debug('Telemetry disabled')

	def capture(self, event: BaseTelemetryEvent) -> None:
		if self._posthog_client is None:
			return

		self._direct_capture(event)

	def _direct_capture(self, event: BaseTelemetryEvent) -> None:
		"""
		Should not be thread blocking because posthog magically handles it
		"""
		if self._posthog_client is None:
			return

		try:
			self._posthog_client.capture(
				distinct_id=self.user_id,
				event=event.name,
				properties={**event.properties, **POSTHOG_EVENT_SETTINGS},
			)
		except Exception as e:
			logger.error(f'Failed to send telemetry event {event.name}: {e}')

	def flush(self) -> None:
		if self._posthog_client:
			try:
				self._posthog_client.flush()
				logger.debug('PostHog client telemetry queue flushed.')
			except Exception as e:
				logger.error(f'Failed to flush PostHog client: {e}')
		else:
			logger.debug('PostHog client not available, skipping flush.')

	@property
	def user_id(self) -> str:
		if self._curr_user_id:
			return self._curr_user_id

		# File access may fail due to permissions or other reasons. We don't want to
		# crash so we catch all exceptions.
		try:
			if not os.path.exists(self.USER_ID_PATH):
				os.makedirs(os.path.dirname(self.USER_ID_PATH), exist_ok=True)
				with open(self.USER_ID_PATH, 'w') as f:
					new_user_id = uuid7str()
					f.write(new_user_id)
				self._curr_user_id = new_user_id
			else:
				with open(self.USER_ID_PATH) as f:
					self._curr_user_id = f.read()
		except Exception:
			self._curr_user_id = 'UNKNOWN_USER_ID'
		return self._curr_user_id

# From telemetry/service.py
def capture(self, event: BaseTelemetryEvent) -> None:
		if self._posthog_client is None:
			return

		self._direct_capture(event)

# From telemetry/service.py
def flush(self) -> None:
		if self._posthog_client:
			try:
				self._posthog_client.flush()
				logger.debug('PostHog client telemetry queue flushed.')
			except Exception as e:
				logger.error(f'Failed to flush PostHog client: {e}')
		else:
			logger.debug('PostHog client not available, skipping flush.')

from collections.abc import Sequence
from dataclasses import asdict

# From telemetry/views.py
class BaseTelemetryEvent(ABC):
	@property
	@abstractmethod
	def name(self) -> str:
		pass

	@property
	def properties(self) -> dict[str, Any]:
		return {k: v for k, v in asdict(self).items() if k != 'name'}

# From telemetry/views.py
class AgentTelemetryEvent(BaseTelemetryEvent):
	# start details
	task: str
	model: str
	model_provider: str
	planner_llm: str | None
	max_steps: int
	max_actions_per_step: int
	use_vision: bool
	use_validation: bool
	version: str
	source: str
	cdp_url: str | None
	# step details
	action_errors: Sequence[str | None]
	action_history: Sequence[list[dict] | None]
	urls_visited: Sequence[str | None]
	# end details
	steps: int
	total_input_tokens: int
	total_duration_seconds: float
	success: bool | None
	final_result_response: str | None
	error_message: str | None

	name: str = 'agent_event'

# From telemetry/views.py
class MCPClientTelemetryEvent(BaseTelemetryEvent):
	"""Telemetry event for MCP client usage"""

	server_name: str
	command: str
	tools_discovered: int
	version: str
	action: str  # 'connect', 'disconnect', 'tool_call'
	tool_name: str | None = None
	duration_seconds: float | None = None
	error_message: str | None = None

	name: str = 'mcp_client_event'

# From telemetry/views.py
class MCPServerTelemetryEvent(BaseTelemetryEvent):
	"""Telemetry event for MCP server usage"""

	version: str
	action: str  # 'start', 'stop', 'tool_call'
	tool_name: str | None = None
	duration_seconds: float | None = None
	error_message: str | None = None
	parent_process_cmdline: str | None = None

	name: str = 'mcp_server_event'

# From telemetry/views.py
class CLITelemetryEvent(BaseTelemetryEvent):
	"""Telemetry event for CLI usage"""

	version: str
	action: str  # 'start', 'message_sent', 'task_completed', 'error'
	mode: str  # 'interactive', 'oneshot', 'mcp_server'
	model: str | None = None
	model_provider: str | None = None
	duration_seconds: float | None = None
	error_message: str | None = None

	name: str = 'cli_event'

# From telemetry/views.py
def name(self) -> str:
		pass

# From telemetry/views.py
def properties(self) -> dict[str, Any]:
		return {k: v for k, v in asdict(self).items() if k != 'name'}

from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession



# From browser/utils.py
def normalize_url(url: str) -> str:
	"""
	Normalize a URL by adding https:// protocol if needed, while preserving special URLs.

	This function safely adds https:// to URLs that lack a protocol, but preserves
	special URLs like "about:blank", "chrome://new-tab-page", "mailto:...", "tel:...", etc.
	that should not be prefixed with https://.

	Args:
	    url: The URL string to normalize

	Returns:
	    str: The normalized URL with protocol if needed

	Examples:
	    >>> normalize_url('example.com')
	    'https://example.com'
	    >>> normalize_url('about:blank')
	    'about:blank'
	    >>> normalize_url('mailto:test@example.com')
	    'mailto:test@example.com'
	    >>> normalize_url('https://example.com')
	    'https://example.com'
	"""
	normalized_url = url.strip()

	# If URL already has a protocol, return as-is
	if '://' in normalized_url:
		return normalized_url

	# Check for special protocols that should not be prefixed with https://
	special_protocols = ['about:', 'mailto:', 'tel:', 'ftp:', 'file:', 'data:', 'javascript:']
	for protocol in special_protocols:
		if normalized_url.startswith(protocol):
			return normalized_url

	# For everything else, add https://
	return f'https://{normalized_url}'

from patchright._impl._errors import TargetClosedError
from patchright.async_api import Browser
from patchright.async_api import BrowserContext
from patchright.async_api import ElementHandle
from patchright.async_api import FrameLocator
from patchright.async_api import Page
from patchright.async_api import Playwright
from patchright.async_api import async_playwright
from playwright._impl._errors import TargetClosedError
from playwright.async_api import Browser
from playwright.async_api import BrowserContext
from playwright.async_api import ElementHandle
from playwright.async_api import FrameLocator
from playwright.async_api import Page
from playwright.async_api import Playwright
from playwright.async_api import async_playwright
from playwright._impl._api_structures import ClientCertificate
from playwright._impl._api_structures import Geolocation
from playwright._impl._api_structures import HttpCredentials
from playwright._impl._api_structures import ProxySettings
from playwright._impl._api_structures import StorageState
from playwright._impl._api_structures import ViewportSize
from typing_extensions import TypedDict


import atexit
from typing import Self
from typing_extensions import deprecated
from browser_use.utils import _log_pretty_url
from utils import normalize_url
from pydantic import AliasChoices
from pydantic import InstanceOf
from pydantic import PrivateAttr
from browser_use.browser.profile import BROWSERUSE_DEFAULT_CHANNEL
from browser_use.browser.profile import BrowserChannel
from browser_use.browser.types import ElementHandle
from browser_use.browser.types import FrameLocator
from browser_use.browser.types import Patchright
from browser_use.browser.types import PlaywrightOrPatchright
from browser_use.browser.types import async_patchright
from browser_use.browser.types import async_playwright
from browser_use.browser.views import PageInfo
from browser_use.browser.views import TabInfo
from browser_use.browser.views import URLNotAllowedError
from browser_use.utils import match_url_with_domain_pattern
from browser_use.utils import merge_dicts
from browser_use.dom.clickable_element_processor.service import ClickableElementProcessor
from browser_use.dom.service import DomService
from browser_use.browser.types import TargetClosedError
import socket

# From browser/session.py
class CachedClickableElementHashes:
	"""
	Clickable elements hashes for the last state
	"""

	url: str
	hashes: set[str]

# From browser/session.py
class BrowserSession(BaseModel):
	"""
	Represents an active browser session with a running browser process somewhere.

	Chromium flags should be passed via extra_launch_args.
	Extra Playwright launch options (e.g., handle_sigterm, handle_sigint) can be passed as kwargs to BrowserSession and will be forwarded to the launch() call.
	"""

	model_config = ConfigDict(
		extra='allow',
		validate_assignment=False,
		revalidate_instances='always',
		frozen=False,
		arbitrary_types_allowed=True,
		validate_by_alias=True,
		validate_by_name=True,
	)
	# this class accepts arbitrary extra **kwargs in init because of the extra='allow' pydantic option
	# they are saved on the model, then applied to self.browser_profile via .apply_session_overrides_to_profile()

	# Persistent ID for this browser session
	id: str = Field(default_factory=uuid7str)

	# template profile for the BrowserSession, will be copied at init/validation time, and overrides applied to the copy
	browser_profile: InstanceOf[BrowserProfile] = Field(
		default=DEFAULT_BROWSER_PROFILE,
		description='BrowserProfile() instance containing config for the BrowserSession',
		validation_alias=AliasChoices(
			'profile', 'config', 'new_context_config'
		),  # abbreviations = 'profile', old deprecated names = 'config', 'new_context_config'
	)

	# runtime props/state: these can be passed in as props at init, or get auto-setup by BrowserSession.start()
	wss_url: str | None = Field(
		default=None,
		description='WSS URL of the node.js playwright browser server to connect to, outputted by (await chromium.launchServer()).wsEndpoint()',
	)
	cdp_url: str | None = Field(
		default=None,
		description='CDP URL of the browser to connect to, e.g. http://localhost:9222 or ws://127.0.0.1:9222/devtools/browser/387adf4c-243f-4051-a181-46798f4a46f4',
	)
	browser_pid: int | None = Field(
		default=None,
		description='pid of a running chromium-based browser process to connect to on localhost',
		validation_alias=AliasChoices('chrome_pid'),  # old deprecated name = chrome_pid
	)
	playwright: PlaywrightOrPatchright | None = Field(
		default=None,
		description='Playwright library object returned by: await (playwright or patchright).async_playwright().start()',
		exclude=True,
	)
	browser: Browser | None = Field(
		default=None,
		description='playwright Browser object to use (optional)',
		validation_alias=AliasChoices('playwright_browser'),
		exclude=True,
	)
	browser_context: BrowserContext | None = Field(
		default=None,
		description='playwright BrowserContext object to use (optional)',
		validation_alias=AliasChoices('playwright_browser_context', 'context'),
		exclude=True,
	)

	# runtime state: state that changes during the lifecycle of a BrowserSession(), updated by the methods below
	initialized: bool = Field(
		default=False,
		description='Mark BrowserSession launch/connection as already ready and skip setup (not recommended)',
		validation_alias=AliasChoices('is_initialized'),
	)
	agent_current_page: Page | None = Field(  # mutated by self.create_new_tab(url)
		default=None,
		description='Foreground Page that the agent is focused on',
		validation_alias=AliasChoices('current_page', 'page'),  # alias page= allows passing in a playwright Page object easily
		exclude=True,
	)
	human_current_page: Page | None = Field(  # mutated by self._setup_current_page_change_listeners()
		default=None,
		description='Foreground Page that the human is focused on',
		exclude=True,
	)

	_cached_browser_state_summary: BrowserStateSummary | None = PrivateAttr(default=None)
	_cached_clickable_element_hashes: CachedClickableElementHashes | None = PrivateAttr(default=None)
	_tab_visibility_callback: Any = PrivateAttr(default=None)
	_logger: logging.Logger | None = PrivateAttr(default=None)
	_downloaded_files: list[str] = PrivateAttr(default_factory=list)
	_original_browser_session: Any = PrivateAttr(default=None)  # Reference to prevent GC of the original session when copied
	_owns_browser_resources: bool = PrivateAttr(default=True)  # True if this instance owns and should clean up browser resources
	_auto_download_pdfs: bool = PrivateAttr(default=True)  # Auto-download PDFs when detected
	_subprocess: Any = PrivateAttr(default=None)  # Chrome subprocess reference for error handling
	_current_page_loading_status: str | None = PrivateAttr(default=None)  # Track loading status for current page

	@model_validator(mode='after')
	def apply_session_overrides_to_profile(self) -> Self:
		"""Apply any extra **kwargs passed to BrowserSession(...) as session-specific config overrides on top of browser_profile"""
		session_own_fields = type(self).model_fields.keys()

		# get all the extra kwarg overrides passed to BrowserSession(...) that are actually
		# config Fields tracked by BrowserProfile, instead of BrowserSession's own args
		profile_overrides = self.model_dump(exclude=set(session_own_fields))

		# FOR REPL DEBUGGING ONLY, NEVER ALLOW CIRCULAR REFERENCES IN REAL CODE:
		# self.browser_profile._in_use_by_session = self

		self.browser_profile = self.browser_profile.model_copy(update=profile_overrides)

		# FOR REPL DEBUGGING ONLY, NEVER ALLOW CIRCULAR REFERENCES IN REAL CODE:
		# self.browser_profile._in_use_by_session = self

		return self

	@model_validator(mode='after')
	def set_browser_ownership(self) -> Self:
		"""Set _owns_browser_resources based on whether we're connecting to an external browser"""
		# If user provided CDP URL, WSS URL, or existing browser/context, we don't own the browser
		if self.cdp_url or self.wss_url or self.browser or self.browser_context:
			self._owns_browser_resources = False
		return self

	@property
	def logger(self) -> logging.Logger:
		"""Get instance-specific logger with session ID in the name"""
		if (
			self._logger is None or self.browser_context is None
		):  # keep updating the name pre-init because our id and str(self) can change
			self._logger = logging.getLogger(f'browser_use.{self}')
		return self._logger

	def __repr__(self) -> str:
		is_copy = '©' if self._original_browser_session else '#'
		port_number_or_pid = (
			(self.cdp_url or self.wss_url or str(self.browser_pid) or 'playwright').rsplit(':', 1)[-1].split('/', 1)[0]
		)
		return f'BrowserSession🆂 {self.id[-4:]}:{port_number_or_pid} {is_copy}{str(id(self))[-2:]} ({self._connection_str}, profile={self.browser_profile})'

	def __str__(self) -> str:
		is_copy = '©' if self._original_browser_session else '#'
		port_number_or_pid = (
			(self.cdp_url or self.wss_url or str(self.browser_pid) or 'playwright').rsplit(':', 1)[-1].split('/', 1)[0]
		)
		return f'BrowserSession🆂 {self.id[-4:]}:{port_number_or_pid} {is_copy}{str(id(self))[-2:]}'  # ' 🅟 {str(id(self.agent_current_page))[-2:]}'

	# better to force people to get it from the right object, "only one way to do it" is better python
	# def __getattr__(self, key: str) -> Any:
	# 	"""
	# 	fall back to getting any attrs from the underlying self.browser_profile when not defined on self.
	# 	(extra kwargs passed e.g. BrowserSession(extra_kwarg=124) on init get saved into self.browser_profile on validation,
	# 	so this also allows you to read those: browser_session.extra_kwarg => browser_session.browser_profile.extra_kwarg)
	# 	"""
	# 	return getattr(self.browser_profile, key)

	@observe_debug(ignore_input=True, ignore_output=True, name='browser.session.start')
	async def start(self) -> Self:
		"""
		Starts the browser session by either connecting to an existing browser or launching a new one.
		Precedence order for launching/connecting:
			1. page=Page playwright object, will use its page.context as browser_context
			2. browser_context=PlaywrightBrowserContext object, will use its browser
			3. browser=PlaywrightBrowser object, will use its first available context
			4. browser_pid=int, will connect to a local chromium-based browser via pid
			5. wss_url=str, will connect to a remote playwright browser server via WSS
			6. cdp_url=str, will connect to a remote chromium-based browser via CDP
			7. playwright=Playwright object, will use its chromium instance to launch a new browser
		"""

		# if we're already initialized and the connection is still valid, return the existing session state and start from scratch

		# Use timeout to prevent indefinite waiting on lock acquisition

		# Quick return if already connected
		if self.initialized and await self.is_connected():
			return self

		# Reset if we were initialized but lost connection
		if self.initialized:
			self.logger.warning(f'💔 Browser {self._connection_str} has gone away, attempting to reconnect...')
			self._reset_connection_state()

		try:
			# Setup
			self.browser_profile.detect_display_configuration()
			# Note: prepare_user_data_dir() is called later in _unsafe_setup_new_browser_context()
			# after the temp directory is created. Calling it here is premature.

			# Get playwright object (has its own retry/semaphore)
			await self.setup_playwright()

			# Try to connect/launch browser (each has appropriate retry logic)
			await self._connect_or_launch_browser()

			# Ensure we have a context
			assert self.browser_context, f'Failed to create BrowserContext for browser={self.browser}'

			# Configure browser - run some setup tasks in parallel for speed
			setup_results = await asyncio.gather(
				self._setup_viewports(),
				self._setup_current_page_change_listeners(),
				self._start_context_tracing(),
				return_exceptions=True,
			)

			# Check for exceptions in setup results
			for i, result in enumerate(setup_results):
				if isinstance(result, Exception):
					setup_task_names = ['_setup_viewports', '_setup_current_page_change_listeners', '_start_context_tracing']
					raise Exception(f'Browser setup failed in {setup_task_names[i]}: {result}') from result

			self.initialized = True
			return self

		except BaseException:
			self.initialized = False
			raise

	@property
	def _connection_str(self) -> str:
		"""Get a logging-ready string describing the connection method e.g. browser=playwright+google-chrome-canary (local)"""
		binary_name = (
			Path(self.browser_profile.executable_path).name.lower().replace(' ', '-').replace('.exe', '')
			if self.browser_profile.executable_path
			else (self.browser_profile.channel or BROWSERUSE_DEFAULT_CHANNEL).name.lower().replace('_', '-').replace(' ', '-')
		)  # Google Chrome Canary.exe -> google-chrome-canary
		driver_name = 'playwright'
		if self.browser_profile.stealth:
			driver_name = 'patchright'
		return (
			f'cdp_url={self.cdp_url}'
			if self.cdp_url
			else f'wss_url={self.wss_url}'
			if self.wss_url
			else f'browser_pid={self.browser_pid}'
			if self.browser_pid
			else f'browser={driver_name}:{binary_name}'
		)

	async def stop(self, _hint: str = '') -> None:
		"""Shuts down the BrowserSession, killing the browser process (only works if keep_alive=False)"""

		# Save cookies to disk if configured
		if self.browser_context:
			try:
				await self.save_storage_state()
			except Exception as e:
				self.logger.warning(f'⚠️ Failed to save auth storage state before stopping: {type(e).__name__}: {e}')

		if self.browser_profile.keep_alive:
			self.logger.info(
				'🕊️ BrowserSession.stop() called but keep_alive=True, leaving the browser running. Use .kill() to force close.'
			)
			return  # nothing to do if keep_alive=True, leave the browser running

		# Only the owner can actually stop the browser
		if not self._owns_browser_resources:
			self.logger.debug(f'🔗 BrowserSession.stop() called on a copy, not closing shared browser resources {_hint}')
			# Still reset our references though
			self._reset_connection_state()
			return

		if self.browser_context or self.browser:
			self.logger.info(f'🛑 Closing {self._connection_str} browser context {_hint} {self.browser or self.browser_context}')

			# Save trace recording if configured
			if self.browser_profile.traces_dir and self.browser_context:
				try:
					await self._save_trace_recording()
				except Exception as e:
					# TargetClosedError is expected when browser has already been closed
					from browser_use.browser.types import TargetClosedError

					if isinstance(e, TargetClosedError):
						self.logger.debug('Browser context already closed, trace may have been saved automatically')
					else:
						self.logger.error(f'❌ Error saving browser context trace: {type(e).__name__}: {e}')

			# Log video/HAR save operations (saved automatically on close)
			if self.browser_profile.record_video_dir:
				self.logger.info(f'🎥 Saving video recording to record_video_dir= {self.browser_profile.record_video_dir}...')
			if self.browser_profile.record_har_path:
				self.logger.info(f'🎥 Saving HAR file to record_har_path= {self.browser_profile.record_har_path}...')

			# Close browser context and browser using retry-decorated methods
			try:
				# IMPORTANT: Close context first to ensure HAR/video files are saved
				await self._close_browser_context()
				await self._close_browser()
			except Exception as e:
				if 'browser has been closed' not in str(e):
					self.logger.warning(f'❌ Error closing browser: {type(e).__name__}: {e}')
			finally:
				# Always clear references to ensure a fresh start next time
				self.browser_context = None
				self.browser = None

		# Kill the chrome subprocess if we started it
		if self.browser_pid:
			try:
				await self._terminate_browser_process(_hint='(stop() called)')
			except psutil.NoSuchProcess:
				self.browser_pid = None
			except (TimeoutError, psutil.TimeoutExpired):
				# If graceful termination failed, force kill
				try:
					proc = psutil.Process(pid=self.browser_pid)
					self.logger.warning(f'⏱️ Process did not terminate gracefully, force killing browser_pid={self.browser_pid}')
					proc.kill()
				except psutil.NoSuchProcess:
					pass
				self.browser_pid = None
			except Exception as e:
				if 'NoSuchProcess' not in type(e).__name__:
					self.logger.debug(f'❌ Error terminating subprocess: {type(e).__name__}: {e}')
				self.browser_pid = None

		# Clean up temporary user data directory
		if self.browser_profile.user_data_dir and Path(self.browser_profile.user_data_dir).name.startswith('browseruse-tmp'):
			shutil.rmtree(self.browser_profile.user_data_dir, ignore_errors=True)

		# Clear CDP/WSS URLs when stopping the browser
		self.cdp_url = None
		self.wss_url = None

		self._reset_connection_state()

	async def close(self) -> None:
		"""Deprecated: Provides backwards-compatibility with old method Browser().close() and playwright BrowserContext.close()"""
		await self.stop(_hint='(close() called)')

	async def kill(self) -> None:
		"""Stop the BrowserSession even if keep_alive=True"""
		# self.logger.debug(
		# 	f'⏹️ Browser browser_pid={self.browser_pid} user_data_dir= {_log_pretty_path(self.browser_profile.user_data_dir) or "<incognito>"} keep_alive={self.browser_profile.keep_alive} (close() called)'
		# )
		self.browser_profile.keep_alive = False
		await self.stop(_hint='(kill() called)')

		# do not stop self.playwright here as its likely used by other parallel browser_sessions
		# let it be cleaned up by the garbage collector when no refs use it anymore

	async def new_context(self, **kwargs):
		"""Deprecated: Provides backwards-compatibility with old class method Browser().new_context()."""
		# TODO: remove this after >=0.3.0
		return self

	async def __aenter__(self) -> BrowserSession:
		await self.start()
		return self

	def __eq__(self, other: object) -> bool:
		"""Check if two BrowserSession instances are using the same browser."""

		if not isinstance(other, BrowserSession):
			return False

		# Two sessions are considered equal if they're connected to the same browser
		# All three connection identifiers must match
		return self.browser_pid == other.browser_pid and self.cdp_url == other.cdp_url and self.wss_url == other.wss_url

	async def __aexit__(self, exc_type, exc_val, exc_tb):
		# self.logger.debug(
		# 	f'⏹️ Stopping gracefully browser_pid={self.browser_pid} user_data_dir= {_log_pretty_path(self.browser_profile.user_data_dir) or "<incognito>"} keep_alive={self.browser_profile.keep_alive} (context manager exit)'
		# )
		await self.stop(_hint='(context manager exit)')

	def model_copy(self, **kwargs) -> Self:
		"""Create a copy of this BrowserSession that shares the browser resources but doesn't own them.

		This method creates a copy that:
		- Shares the same browser, browser_context, and playwright objects
		- Doesn't own the browser resources (won't close them when garbage collected)
		- Keeps a reference to the original to prevent premature garbage collection
		"""
		# Create the copy using the parent class method
		copy = super().model_copy(**kwargs)

		# The copy doesn't own the browser resources
		copy._owns_browser_resources = False

		# Keep a reference to the original to prevent garbage collection
		copy._original_browser_session = self

		# Manually copy over the excluded fields that are needed for browser connection
		# These fields are excluded in the model config but need to be shared
		copy.playwright = self.playwright
		copy.browser = self.browser
		copy.browser_context = self.browser_context
		copy.agent_current_page = self.agent_current_page
		copy.human_current_page = self.human_current_page
		copy.browser_pid = self.browser_pid

		return copy

	def __del__(self):
		profile = getattr(self, 'browser_profile', None)
		keep_alive = getattr(profile, 'keep_alive', None)
		user_data_dir = getattr(profile, 'user_data_dir', None)
		owns_browser = getattr(self, '_owns_browser_resources', True)
		status = f'🪓 killing pid={self.browser_pid}...' if (self.browser_pid and owns_browser) else '☠️'
		self.logger.debug(
			f'🗑️ Garbage collected BrowserSession 🆂 {self.id[-4:]}.{str(id(self.agent_current_page))[-2:]} ref #{str(id(self))[-4:]} keep_alive={keep_alive} owns_browser={owns_browser} {status}'
		)
		# Only kill browser processes if this instance owns them
		if owns_browser:
			# Avoid keeping references in __del__ that might prevent garbage collection
			try:
				self._kill_child_processes(_hint='(garbage collected)')
			except TimeoutError:
				# Never let __del__ raise Timeout exceptions
				pass

	def _kill_child_processes(self, _hint: str = '') -> None:
		"""Kill any child processes that might be related to the browser"""

		if not self.browser_profile.keep_alive and self.browser_pid:
			try:
				browser_proc = psutil.Process(self.browser_pid)
				try:
					browser_proc.terminate()
					browser_proc.wait(
						timeout=5
					)  # wait up to 5 seconds for the process to exit cleanly and commit its user_data_dir changes
					self.logger.debug(f'🍂 Killed browser subprocess gracefully browser_pid={self.browser_pid} {_hint}')
				except (psutil.NoSuchProcess, psutil.AccessDenied, TimeoutError):
					pass

				# Kill all child processes first (recursive)
				for child in browser_proc.children(recursive=True):
					try:
						# self.logger.debug(f'Force killing child process: {child.pid} ({child.name()})')
						child.kill()
						self.logger.debug(f'☠️ Force-killed hung browser helper subprocess pid={child.pid} {_hint}')
					except (psutil.NoSuchProcess, psutil.AccessDenied):
						pass

				# Kill the main browser process
				# self.logger.debug(f'Force killing browser process: {self.browser_pid}')
				browser_proc.kill()
				self.logger.debug(f'☠️ Force-killed hung browser subprocess browser_pid={self.browser_pid} {_hint}')
			except psutil.NoSuchProcess:
				pass
			except Exception as e:
				self.logger.warning(f'⚠️ Error force-killing browser in BrowserSession.__del__: {type(e).__name__}: {e}')

	@staticmethod
	async def _start_global_playwright_subprocess(is_stealth: bool) -> PlaywrightOrPatchright:
		"""Create and return a new playwright or patchright node.js subprocess / API connector"""
		global GLOBAL_PLAYWRIGHT_API_OBJECT, GLOBAL_PATCHRIGHT_API_OBJECT
		global GLOBAL_PLAYWRIGHT_EVENT_LOOP, GLOBAL_PATCHRIGHT_EVENT_LOOP

		try:
			current_loop = asyncio.get_running_loop()
		except RuntimeError:
			current_loop = None

		if is_stealth:
			GLOBAL_PATCHRIGHT_API_OBJECT = await async_patchright().start()
			GLOBAL_PATCHRIGHT_EVENT_LOOP = current_loop
			return GLOBAL_PATCHRIGHT_API_OBJECT
		else:
			GLOBAL_PLAYWRIGHT_API_OBJECT = await async_playwright().start()
			GLOBAL_PLAYWRIGHT_EVENT_LOOP = current_loop
			return GLOBAL_PLAYWRIGHT_API_OBJECT

	async def _unsafe_get_or_start_playwright_object(self) -> PlaywrightOrPatchright:
		"""Get existing or create new global playwright object with proper locking."""
		global GLOBAL_PLAYWRIGHT_API_OBJECT, GLOBAL_PATCHRIGHT_API_OBJECT
		global GLOBAL_PLAYWRIGHT_EVENT_LOOP, GLOBAL_PATCHRIGHT_EVENT_LOOP

		# Get current event loop
		try:
			current_loop = asyncio.get_running_loop()
		except RuntimeError:
			current_loop = None

		is_stealth = self.browser_profile.stealth
		driver_name = 'patchright' if is_stealth else 'playwright'
		global_api_object = GLOBAL_PATCHRIGHT_API_OBJECT if is_stealth else GLOBAL_PLAYWRIGHT_API_OBJECT
		global_event_loop = GLOBAL_PATCHRIGHT_EVENT_LOOP if is_stealth else GLOBAL_PLAYWRIGHT_EVENT_LOOP

		# Check if we need to create or recreate the global object
		should_recreate = False

		if global_api_object and global_event_loop != current_loop:
			self.logger.debug(
				f'Detected event loop change. Previous {driver_name} instance was created in a different event loop. '
				'Creating new instance to avoid disconnection when the previous loop closes.'
			)
			should_recreate = True

		# Also check if the object exists but is no longer functional
		if global_api_object and not should_recreate:
			try:
				# Try to access the chromium property to verify the object is still valid
				_ = global_api_object.chromium.executable_path
			except Exception as e:
				self.logger.debug(f'Detected invalid {driver_name} instance: {type(e).__name__}. Creating new instance.')
				should_recreate = True

		# If we already have a valid object, use it
		if global_api_object and not should_recreate:
			return global_api_object

		# Create new playwright object
		return await self._start_global_playwright_subprocess(is_stealth=is_stealth)

	@retry(wait=1, retries=2, timeout=45, semaphore_limit=1, semaphore_scope='self', semaphore_lax=False)
	async def _close_browser_context(self) -> None:
		"""Close browser context with retry logic."""
		await self._unsafe_close_browser_context()

	async def _unsafe_close_browser_context(self) -> None:
		"""Unsafe browser context close logic without retry protection."""
		if self.browser_context:
			await self.browser_context.close()
			self.browser_context = None

	@retry(wait=1, retries=2, timeout=10, semaphore_limit=1, semaphore_scope='self', semaphore_lax=False)
	async def _close_browser(self) -> None:
		"""Close browser instance with retry logic."""
		await self._unsafe_close_browser()

	async def _unsafe_close_browser(self) -> None:
		"""Unsafe browser close logic without retry protection."""
		if self.browser and self.browser.is_connected():
			await self.browser.close()
			self.browser = None

	@retry(
		wait=0.5,
		retries=3,
		timeout=5,
		semaphore_limit=1,
		semaphore_scope='self',
		semaphore_lax=True,
		retry_on=(TimeoutError, psutil.TimeoutExpired),  # Only retry on timeouts, not NoSuchProcess
	)
	async def _terminate_browser_process(self, _hint: str = '') -> None:
		"""Terminate browser process with retry logic."""
		await self._unsafe_terminate_browser_process(_hint='(terminate() called)')

	async def _unsafe_terminate_browser_process(self, _hint: str = '') -> None:
		"""Unsafe browser process termination without retry protection."""
		if self.browser_pid:
			try:
				proc = psutil.Process(pid=self.browser_pid)
				cmdline = proc.cmdline()
				executable_path = cmdline[0] if cmdline else 'unknown'
				self.logger.info(f' ↳ Killing browser_pid={self.browser_pid} {_log_pretty_path(executable_path)} {_hint}')

				# Try graceful termination first
				proc.terminate()
				self._kill_child_processes(_hint=_hint)
				await asyncio.to_thread(proc.wait, timeout=4)
			except psutil.NoSuchProcess:
				# Process already gone, that's fine
				pass
			finally:
				self.browser_pid = None

	@retry(wait=0.5, retries=2, timeout=30, semaphore_limit=1, semaphore_scope='self', semaphore_lax=True)
	async def _save_trace_recording(self) -> None:
		"""Save browser trace recording."""
		# TEMPORARILY DISABLED: Trace recording causing test timeouts
		return
		# if self.browser_profile.traces_dir and self.browser_context is not None:
		# 	traces_path = Path(self.browser_profile.traces_dir)
		# 	if traces_path.suffix:
		# 		# Path has extension, use as-is (user specified exact file path)
		# 		final_trace_path = traces_path
		# 	else:
		# 		# Path has no extension, treat as directory and create filename
		# 		trace_filename = f'BrowserSession_{self.id}.zip'
		# 		final_trace_path = traces_path / trace_filename

		# 	self.logger.info(f'🎥 Saving browser_context trace to {final_trace_path}...')
		# 	await self.browser_context.tracing.stop(path=str(final_trace_path))

	@observe_debug(ignore_input=True, ignore_output=True, name='connect_or_launch_browser')
	async def _connect_or_launch_browser(self, retry_count: int = 0) -> None:
		"""Try all connection methods in order of precedence.

		Args:
			retry_count: Number of retries already attempted (max 2)
		"""
		# Try connecting via passed objects first
		await self.setup_browser_via_passed_objects()
		if self.browser_context:
			return

		# Try connecting via browser PID
		await self.setup_browser_via_browser_pid()
		if self.browser_context:
			return

		# Try connecting via WSS URL
		await self.setup_browser_via_wss_url()
		if self.browser_context:
			return

		# Try connecting via CDP URL
		await self.setup_browser_via_cdp_url()
		if self.browser_context:
			return

		# Launch new browser as last resort
		await self.setup_new_browser_context(retry_count)

	# Removed _take_screenshot_hybrid - merged into take_screenshot

	@observe_debug(ignore_input=True, ignore_output=True, name='setup_playwright')
	@retry(
		wait=1,
		retries=3,
		timeout=10,
		semaphore_limit=1,
		semaphore_name='playwright_global_object',
		semaphore_scope='global',
		semaphore_lax=False,
		semaphore_timeout=5,  # 5s to wait for global playwright object
	)
	async def setup_playwright(self) -> None:
		"""
		Set up playwright library client object: usually the result of (await async_playwright().start())
		Override to customize the set up of the playwright or patchright library object
		"""
		is_stealth = self.browser_profile.stealth

		# Configure browser channel based on stealth mode
		if is_stealth:
			# use patchright + chrome when stealth=True
			self.browser_profile.channel = self.browser_profile.channel or BrowserChannel.CHROME
			self.logger.info(f'🕶️ Activated stealth mode using patchright {self.browser_profile.channel.name.lower()} browser...')
		else:
			# use playwright + chromium by default
			self.browser_profile.channel = self.browser_profile.channel or BrowserChannel.CHROMIUM

		# Get or create the global playwright object
		self.playwright = self.playwright or await self._unsafe_get_or_start_playwright_object()

		# Log stealth best-practices warnings if applicable
		if is_stealth:
			if self.browser_profile.channel and self.browser_profile.channel != BrowserChannel.CHROME:
				self.logger.info(
					' 🪄 For maximum stealth, BrowserSession(...) should be passed channel=None or BrowserChannel.CHROME'
				)
			if not self.browser_profile.user_data_dir:
				self.logger.info(' 🪄 For maximum stealth, BrowserSession(...) should be passed a persistent user_data_dir=...')
			if self.browser_profile.headless or not self.browser_profile.no_viewport:
				self.logger.info(' 🪄 For maximum stealth, BrowserSession(...) should be passed headless=False & viewport=None')

		# register a shutdown hook to stop the shared global playwright node.js client when the program exits (if an event loop is still running)
		def shudown_playwright():
			if not self.playwright:
				return
			try:
				loop = asyncio.get_running_loop()
				self.logger.debug('🛑 Shutting down shared global playwright node.js client')
				task = loop.create_task(self.playwright.stop())
				if hasattr(task, '_log_destroy_pending'):
					task._log_destroy_pending = False  # type: ignore
			except Exception:
				pass
			self.playwright = None

		atexit.register(shudown_playwright)

	@observe_debug(ignore_input=True, ignore_output=True, name='setup_browser_via_passed_objects')
	async def setup_browser_via_passed_objects(self) -> None:
		"""Override to customize the set up of the connection to an existing browser"""

		# 1. check for a passed Page object, if present, it always takes priority, set browser_context = page.context
		if self.agent_current_page:
			try:
				# Test if the page is still usable by evaluating simple JS
				await self.agent_current_page.evaluate('() => true')
				self.browser_context = self.agent_current_page.context
			except Exception:
				# Page is closed or unusable, clear it
				self.agent_current_page = None
				self.browser_context = None

		# 2. Check if the current browser connection is valid, if not clear the invalid objects
		if self.browser_context:
			try:
				# Try to access a property that would fail if the context is closed
				_ = self.browser_context.pages
				# Additional check: verify the browser is still connected
				if self.browser_context.browser and not self.browser_context.browser.is_connected():
					self.browser_context = None
			except Exception:
				# Context is closed, clear it
				self.browser_context = None

		# 3. if we have a browser object but it's disconnected, clear it and the context because we cant use either
		if self.browser and not self.browser.is_connected():
			if self.browser_context and (self.browser_context.browser is self.browser):
				self.browser_context = None
			self.browser = None

		# 4. if we have a context now, it always takes precedence, set browser = context.browser, otherwise use the passed browser
		browser_from_context = self.browser_context and self.browser_context.browser
		if browser_from_context and browser_from_context.is_connected():
			self.browser = browser_from_context

		if self.browser or self.browser_context:
			self.logger.info(f'🎭 Connected to existing user-provided browser: {self.browser_context}')
			self._set_browser_keep_alive(True)  # we connected to an existing browser, dont kill it at the end

	@observe_debug(ignore_input=True, ignore_output=True, name='setup_browser_via_browser_pid')
	async def setup_browser_via_browser_pid(self) -> None:
		"""if browser_pid is provided, calcuclate its CDP URL by looking for --remote-debugging-port=... in its CLI args, then connect to it"""

		if self.browser or self.browser_context:
			return  # already connected to a browser
		if not self.browser_pid:
			return  # no browser_pid provided, nothing to do

		# check that browser_pid process is running, otherwise we cannot connect to it
		try:
			chrome_process = psutil.Process(pid=self.browser_pid)
			if not chrome_process.is_running():
				self.logger.warning(f'⚠️ Expected Chrome process with pid={self.browser_pid} is not running')
				return
			args = chrome_process.cmdline()
		except psutil.NoSuchProcess:
			self.logger.warning(f'⚠️ Expected Chrome process with pid={self.browser_pid} not found, unable to (re-)connect')
			return
		except Exception as e:
			self.browser_pid = None
			self.logger.warning(f'⚠️ Error accessing chrome process with pid={self.browser_pid}: {type(e).__name__}: {e}')
			return

		# check that browser_pid process is exposing a debug port we can connect to, otherwise we cannot connect to it
		debug_port = next((arg for arg in args if arg.startswith('--remote-debugging-port=')), '').split('=')[-1].strip()
		# self.logger.debug(f'👾 Found Chrome subprocess browser_pid={self.browser_pid} open CDP port: --remote-debugging-port={debug_port}')
		if not debug_port:
			# provided pid is unusable, it's either not running or doesnt have an open debug port we can connect to
			if '--remote-debugging-pipe' in args:
				self.logger.error(
					f'❌ Found --remote-debugging-pipe in browser launch args for browser_pid={self.browser_pid} but it was started by a different BrowserSession, cannot connect to it'
				)
			else:
				self.logger.error(
					f'❌ Could not find --remote-debugging-port=... to connect to in browser launch args for browser_pid={self.browser_pid}: {" ".join(args)}'
				)
			self.browser_pid = None
			return

		self.cdp_url = self.cdp_url or f'http://127.0.0.1:{debug_port}/'

		# Wait for CDP port to become available (Chrome might still be starting)
		import httpx

		# No initial sleep needed - the polling loop below handles waiting if Chrome isn't ready yet

		async with httpx.AsyncClient() as client:
			for i in range(30):  # timeout
				# First check if the Chrome process has exited
				try:
					chrome_process = psutil.Process(pid=self.browser_pid)
					if not chrome_process.is_running():
						# If we have a subprocess reference, try to get stderr
						if hasattr(self, '_subprocess') and self._subprocess:
							stderr_output = ''
							if self._subprocess.stderr:
								try:
									stderr_bytes = await self._subprocess.stderr.read()
									stderr_output = stderr_bytes.decode('utf-8', errors='replace')
								except Exception:
									pass
							if 'Failed parsing extensions' in stderr_output:
								self.logger.error(f'❌ Chrome process {self.browser_pid} exited: Failed parsing extensions')
								raise RuntimeError('Failed parsing extensions: Chrome profile incompatibility detected')
							elif 'SingletonLock' in stderr_output or 'ProcessSingleton' in stderr_output:
								# Chrome exited due to singleton lock
								self.logger.error(
									f'❌ Chrome process {self.browser_pid} crashed due to SingletonLock error: {stderr_output[:500]}'
								)
								# Kill the subprocess
								try:
									self._subprocess.terminate()
									await self._subprocess.wait()
								except Exception:
									pass
								self.browser_pid = None
								# Throw hard error instead of restarting
								raise RuntimeError(f'Chrome process crashed due to SingletonLock error: {stderr_output[:500]}')
							else:
								# Chrome exited for unknown reason
								self.logger.error(
									f'❌ Chrome process {self.browser_pid} crashed unexpectedly. Error: {stderr_output[:500] if stderr_output else "No error output"}'
								)
								# Kill the subprocess
								try:
									self._subprocess.terminate()
									await self._subprocess.wait()
								except Exception:
									pass
								self.browser_pid = None
								# Throw hard error instead of restarting
								raise RuntimeError(
									f'Chrome process crashed unexpectedly: {stderr_output[:500] if stderr_output else "No error output"}'
								)
						self.logger.error(f'❌ Chrome process {self.browser_pid} exited unexpectedly')
						self.browser_pid = None
						return
				except psutil.NoSuchProcess:
					self.logger.error(f'❌ Chrome process {self.browser_pid} no longer exists')
					self.browser_pid = None
					return

				try:
					response = await client.get(f'{self.cdp_url}json/version', timeout=1.0)
					if response.status_code == 200:
						break
					else:
						# FIX: Always sleep if status != 200
						if i == 0:
							self.logger.debug(f'⏳ Waiting for Chrome CDP port {debug_port} to become available...')
						await asyncio.sleep(0.5)
				except (httpx.ConnectError, httpx.TimeoutException):
					if i == 0:
						self.logger.debug(f'⏳ Waiting for Chrome CDP port {debug_port} to become available...')
					await asyncio.sleep(0.5)
			else:
				self.logger.error(f'❌ Chrome CDP port {debug_port} did not become available after 30 seconds')
				self.browser_pid = None
				raise RuntimeError(f'Chrome CDP port {debug_port} did not become available - browser process may have crashed')

		# Determine if this is a newly spawned subprocess or an existing process
		if hasattr(self, '_subprocess') and self._subprocess and self._subprocess.pid == self.browser_pid:
			self.logger.info(
				f'🌎 Connecting to newly spawned browser via CDP {self.cdp_url} -> browser_pid={self.browser_pid} (local)'
			)
		else:
			self.logger.info(
				f'🌎 Connecting to existing browser via CDP  {self.cdp_url} -> browser_pid={self.browser_pid} (local)'
			)
		assert self.playwright is not None, 'playwright instance is None'
		self.browser = self.browser or await self.playwright.chromium.connect_over_cdp(
			self.cdp_url,
			**self.browser_profile.kwargs_for_connect().model_dump(),
		)
		self._set_browser_keep_alive(True)  # we connected to an existing browser, dont kill it at the end

	@observe_debug(ignore_input=True, ignore_output=True, name='setup_browser_via_wss_url')
	async def setup_browser_via_wss_url(self) -> None:
		"""check for a passed wss_url, connect to a remote playwright browser server via WSS"""

		if self.browser or self.browser_context:
			return  # already connected to a browser
		if not self.wss_url:
			return  # no wss_url provided, nothing to do

		self.logger.info(
			f'🌎 Connecting to existing playwright node.js browser server over WSS: {self.wss_url} -> (remote playwright)'
		)
		assert self.playwright is not None, 'playwright instance is None'
		self.browser = self.browser or await self.playwright.chromium.connect(
			self.wss_url,
			**self.browser_profile.kwargs_for_connect().model_dump(),
		)
		self._set_browser_keep_alive(True)  # we connected to an existing browser, dont kill it at the end

	async def setup_browser_via_cdp_url(self) -> None:
		"""check for a passed cdp_url, connect to a remote chromium-based browser via CDP"""

		if self.browser or self.browser_context:
			return  # already connected to a browser
		if not self.cdp_url:
			return  # no cdp_url provided, nothing to do

		self.logger.info(f'🌎 Connecting to existing chromium-based browser via CDP: {self.cdp_url} -> (remote browser)')
		assert self.playwright is not None, 'playwright instance is None'
		self.browser = self.browser or await self.playwright.chromium.connect_over_cdp(
			self.cdp_url,
			**self.browser_profile.kwargs_for_connect().model_dump(),
		)
		self._set_browser_keep_alive(True)  # we connected to an existing browser, dont kill it at the end

	@observe_debug(ignore_input=True, ignore_output=True, name='setup_new_browser_context')
	@retry(wait=0.1, retries=5, timeout=45, semaphore_limit=1, semaphore_scope='self', semaphore_lax=False)
	async def setup_new_browser_context(self, retry_count: int = 0) -> None:
		"""Launch a new browser and browser_context

		Args:
			retry_count: Number of retries already attempted (max 2)
		"""
		# Double-check after semaphore acquisition to prevent duplicate browser launches
		if self.browser_context:
			try:
				# Check if context is still valid and has pages
				if self.browser_context.pages and not all(page.is_closed() for page in self.browser_context.pages):
					# self.logger.debug('Browser context already exists after semaphore acquisition, skipping launch')
					return
			except Exception:
				# If we can't check pages, assume context is invalid and continue with launch
				pass
		await self._unsafe_setup_new_browser_context(retry_count)

	@observe_debug(ignore_input=True, ignore_output=True, name='_unsafe_setup_new_browser_context')
	async def _unsafe_setup_new_browser_context(self, retry_count: int = 0) -> None:
		"""Unsafe browser context setup without retry protection.

		Args:
			retry_count: Number of retries already attempted (max 2)
		"""

		# Note: cdp_url might be set from a previous attempt that failed and is being retried
		# Only assert if we don't own browser resources (meaning cdp_url was user-provided for external browser)
		# AND we don't already have a browser (which means we need to get/create a context)
		if self.cdp_url and not self._owns_browser_resources and not self.browser:
			raise AssertionError(
				'Should never try to set up a new local browser when connecting to an external browser via cdp_url'
			)

		# if we have a browser object but no browser_context, use the first context discovered or make a new one
		if self.browser and not self.browser_context:
			# If HAR recording or video recording is requested, we need to create a new context with recording enabled
			# Cannot reuse existing context as recording must be configured at context creation
			if (self.browser_profile.record_har_path or self.browser_profile.record_video_dir) and self.browser.contexts:
				recording_types = []
				if self.browser_profile.record_har_path:
					recording_types.append('HAR')
				if self.browser_profile.record_video_dir:
					recording_types.append('video')
				self.logger.info(
					f'🎥 Creating new browser_context with {" and ".join(recording_types)} recording enabled (cannot reuse existing context)'
				)
				self.browser_context = await self.browser.new_context(
					**self.browser_profile.kwargs_for_new_context().model_dump(mode='json')
				)
			elif self.browser.contexts:
				self.browser_context = self.browser.contexts[0]
				# Check if this is a newly spawned subprocess
				if hasattr(self, '_subprocess') and self._subprocess and self._subprocess.pid == self.browser_pid:
					self.logger.debug(f'👤 Using default browser_context opened in newly spawned browser: {self.browser_context}')
				else:
					self.logger.info(f'👤 Using first browser_context found in existing browser: {self.browser_context}')
			else:
				self.browser_context = await self.browser.new_context(
					**self.browser_profile.kwargs_for_new_context().model_dump(mode='json')
				)
				storage_info = (
					f' + loaded storage_state={len(self.browser_profile.storage_state) if self.browser_profile.storage_state else 0} cookies'
					if self.browser_profile.storage_state and isinstance(self.browser_profile.storage_state, dict)
					else ''
				)
				self.logger.info(
					f'🌎 Created new empty browser_context in existing browser{storage_info}: {self.browser_context}'
				)

		# if we still have no browser_context by now, launch a new local one using launch_persistent_context()
		if not self.browser_context:
			assert self.browser_profile.channel is not None, 'browser_profile.channel is None'
			self.logger.info(
				f'🎭 Launching new local browser '
				f'{str(type(self.playwright).__module__).split(".")[0]}:{self.browser_profile.channel.name.lower()} keep_alive={self.browser_profile.keep_alive or False} '
				f'user_data_dir= {_log_pretty_path(self.browser_profile.user_data_dir) or "<incognito>"}'
			)

			# if no user_data_dir is provided, generate a unique one for this temporary browser_context (will be used to uniquely identify the browser_pid later)
			if not self.browser_profile.user_data_dir:
				# self.logger.debug('🌎 Launching local browser in incognito mode')
				# if no user_data_dir is provided, generate a unique one for this temporary browser_context (will be used to uniquely identify the browser_pid later)
				self.browser_profile.user_data_dir = self.browser_profile.user_data_dir or Path(
					tempfile.mkdtemp(prefix='browseruse-tmp-')
				)
			# If we're reconnecting and using a temp directory, create a new one
			# This avoids conflicts with the previous browser process that might still be shutting down
			elif self.browser_profile.user_data_dir and Path(self.browser_profile.user_data_dir).name.startswith(
				'browseruse-tmp-'
			):
				old_dir = self.browser_profile.user_data_dir
				self.browser_profile.user_data_dir = Path(tempfile.mkdtemp(prefix='browseruse-tmp-'))
				self.logger.debug(
					f'🗑️ Cleaning up old tmp user_data_dir= {_log_pretty_path(old_dir)} and using fresh one:{_log_pretty_path(self.browser_profile.user_data_dir)}'
				)
				try:
					shutil.rmtree(old_dir)
				except Exception:
					self.logger.warning(f'🗑️ Failed to cleanup old tmp user_data_dir= {_log_pretty_path(old_dir)}')

			# user data dir was provided, prepare it for use (handles conflicts automatically)
			self.prepare_user_data_dir()

			# if a user_data_dir is provided, launch Chrome as subprocess then connect via CDP
			try:
				async with asyncio.timeout(self.browser_profile.timeout / 1000):
					try:
						assert self.playwright is not None, 'playwright instance is None'

						# Find an available port for remote debugging
						import socket

						with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
							s.bind(('127.0.0.1', 0))
							s.listen(1)
							debug_port = s.getsockname()[1]

						# Get chromium executable path from browser profile or fall back to to playwright default
						chromium_path = self.browser_profile.executable_path or self.playwright.chromium.executable_path

						# Build chrome launch command with all args
						chrome_args = self.browser_profile.get_args()

						# Add/replace remote-debugging-port with our chosen port
						final_args = []
						for arg in chrome_args:
							if not arg.startswith('--remote-debugging-port='):
								final_args.append(arg)
						final_args.extend(
							[
								f'--remote-debugging-port={debug_port}',
								f'--user-data-dir={self.browser_profile.user_data_dir}',
							]
						)

						# Build final command
						chrome_launch_cmd = [chromium_path] + final_args

						# Launch chrome as subprocess
						self.logger.info(
							f' ↳ Spawning Chrome subprocess listening on CDP http://127.0.0.1:{debug_port}/ with user_data_dir= {_log_pretty_path(self.browser_profile.user_data_dir)}'
						)
						process = await asyncio.create_subprocess_exec(
							*chrome_launch_cmd,
							stdout=asyncio.subprocess.PIPE,
							stderr=asyncio.subprocess.PIPE,
						)

						# Store the subprocess reference for error handling
						self._subprocess = process

						# Store the browser PID
						self.browser_pid = process.pid
						self._set_browser_keep_alive(False)  # We launched it, so we should close it
						self._owns_browser_resources = True  # We launched it, so we own it
						# self.logger.debug(f'👶 Chrome subprocess launched with browser_pid={process.pid}')

						# Use the existing setup_browser_via_browser_pid method to connect
						# It will wait for the CDP port to become available
						await self.setup_browser_via_browser_pid()

						# If connection failed, browser will be None
						if not self.browser:
							# Try to get error info from the process
							if process.returncode is not None:
								# Chrome exited, try to read stderr for error message
								stderr_output = ''
								if process.stderr:
									try:
										stderr_bytes = await process.stderr.read()
										stderr_output = stderr_bytes.decode('utf-8', errors='replace')
									except Exception:
										pass

								# Check for common Chrome errors
								if 'Failed parsing extensions' in stderr_output:
									raise RuntimeError(
										f'Failed parsing extensions: Chrome profile incompatibility detected. Chrome exited with code {process.returncode}'
									)
								elif 'SingletonLock' in stderr_output or 'ProcessSingleton' in stderr_output:
									raise RuntimeError(f'SingletonLock error: {stderr_output[:500]}')
								else:
									# For any other error, raise hard error
									self.logger.error(
										f'❌ Chrome subprocess crashed with code {process.returncode}. Error: {stderr_output[:500] if stderr_output else "No error output"}'
									)
									raise RuntimeError(
										f'Chrome subprocess crashed with code {process.returncode}. Error output: {stderr_output[:500] if stderr_output else "No error output"}'
									)
							else:
								# Kill the subprocess if it's still running but we couldn't connect
								try:
									process.terminate()
									await process.wait()
								except Exception:
									pass
								raise RuntimeError(f'Failed to connect to Chrome subprocess on port {debug_port}')

					except Exception as e:
						# Check if it's a SingletonLock error or Chrome subprocess exit error
						if (
							'SingletonLock' in str(e)
							or 'ProcessSingleton' in str(e)
							or 'Chrome subprocess exited' in str(e)
							or isinstance(e, RuntimeError)
						):
							# Chrome has crashed - throw hard error instead of restarting
							self.logger.error(f'❌ Chrome process crashed and cannot be recovered: {str(e)}')
							# Kill the failed subprocess if it exists
							if hasattr(self, '_subprocess') and self._subprocess:
								try:
									self._subprocess.terminate()
									await self._subprocess.wait()
								except Exception:
									pass
							# Re-raise to be caught by outer exception handler for fallback
							raise
						# Re-raise if not a timeout
						elif not isinstance(e, asyncio.TimeoutError):
							raise
			except TimeoutError:
				self.logger.error(
					'❌ Browser operation timed out. This may indicate the playwright instance is invalid or the browser has crashed.'
				)
				# Try fallback to temp profile in case it's a profile lock issue
				# But only if we're trying to launch a local browser (not connecting to external)
				if retry_count < 2 and self._owns_browser_resources:
					self.logger.warning(
						f'⚠️ Chrome subprocess failed to start (timeout). Profile at {_log_pretty_path(self.browser_profile.user_data_dir)} may be locked. Using temporary profile instead.'
					)
					self._fallback_to_temp_profile('Chrome subprocess timeout')
					# Retry with temp profile
					return await self.setup_new_browser_context(retry_count + 1)
				else:
					# Max retries reached or external browser - throw hard error
					raise RuntimeError('Browser operation timed out - browser may have crashed or become unresponsive')
			except Exception as e:
				# Check if it's a SingletonLock error or any Chrome subprocess failure
				if 'SingletonLock' in str(e) or 'ProcessSingleton' in str(e) or isinstance(e, RuntimeError):
					# Chrome crashed - fallback to temp profile
					# But only if we're trying to launch a local browser (not connecting to external)
					if retry_count < 2 and self._owns_browser_resources:
						self.logger.warning(
							f'⚠️ Chrome subprocess failed to start detected. Profile at {_log_pretty_path(self.browser_profile.user_data_dir)} is locked. Using temporary profile instead.'
						)
						self._fallback_to_temp_profile()
						# Retry with temp profile
						return await self._connect_or_launch_browser(retry_count + 1)
					else:
						# Max retries reached or external browser - throw error
						self.logger.error(f'❌ Chrome launch failed after {retry_count} retries: {str(e)}')
						raise RuntimeError(f'Chrome launch failed: {str(e)}')

				# show a nice logger hint explaining what went wrong with the user_data_dir
				# calculate the version of the browser that the user_data_dir is for, and the version of the browser we are running with
				user_data_dir_chrome_version = '???'
				test_browser_version = '???'
				try:
					# user_data_dir is corrupted or unreadable because it was migrated to a newer version of chrome than we are running with
					user_data_dir_chrome_version = (Path(self.browser_profile.user_data_dir) / 'Last Version').read_text().strip()
				except Exception:
					pass  # let the logger below handle it
				try:
					assert self.playwright is not None, 'playwright instance is None'
					test_browser = await self.playwright.chromium.launch(headless=True)
					test_browser_version = test_browser.version
					await test_browser.close()
				except Exception:
					pass

				# failed to parse extensions == most common error text when user_data_dir is corrupted / has an unusable schema
				reason = 'due to bad' if 'Failed parsing extensions' in str(e) else 'for unknown reason with'
				driver = str(type(self.playwright).__module__).split('.')[0].lower()
				browser_channel = (
					Path(self.browser_profile.executable_path).name.replace(' ', '-').replace('.exe', '').lower()
					if self.browser_profile.executable_path
					else (self.browser_profile.channel or BROWSERUSE_DEFAULT_CHANNEL).name.lower()
				)
				self.logger.error(
					f'❌ Launching new local browser {driver}:{browser_channel} (v{test_browser_version}) failed!'
					f'\n\tFailed {reason} user_data_dir= {_log_pretty_path(self.browser_profile.user_data_dir)} (created with v{user_data_dir_chrome_version})'
					'\n\tTry using a different browser version/channel or delete the user_data_dir to start over with a fresh profile.'
					'\n\t(can happen if different versions of Chrome/Chromium/Brave/etc. tried to share one dir)'
					f'\n\n{type(e).__name__} {e}'
				)
				raise

		# Only restore browser from context if it's connected, otherwise keep it None to force new launch
		browser_from_context = self.browser_context and self.browser_context.browser
		if browser_from_context and browser_from_context.is_connected():
			self.browser = browser_from_context
		# ^ self.browser can unfortunately still be None at the end ^
		# playwright does not give us a browser object at all when we use launch_persistent_context()!

		# PID detection is no longer needed since we get PIDs directly from subprocesses or passed objects

		if self.browser:
			assert self.browser.is_connected(), (
				f'Browser is not connected, did the browser process crash or get killed? (connection method: {self._connection_str})'
			)
			# Only log final connection if we didn't already log it via setup_browser_via_browser_pid
			if not (hasattr(self, '_subprocess') and self._subprocess and self._subprocess.pid == self.browser_pid):
				self.logger.debug(f'🪢 Browser {self._connection_str} connected {self.browser or self.browser_context}')
		elif self.browser_context and not self.browser:
			# For launch_persistent_context case where we don't get a browser object
			self.logger.debug(f'🪢 Browser context {self._connection_str} connected {self.browser_context}')

		assert self.browser_context, (
			f'{self} Failed to create a playwright BrowserContext {self.browser_context} for browser={self.browser}'
		)

		# self.logger.debug('Setting up init scripts in browser')

		init_script = """
			// check to make sure we're not inside the PDF viewer
			window.isPdfViewer = !!document?.body?.querySelector('body > embed[type="application/pdf"][width="100%"]')
			if (!window.isPdfViewer) {

				// Permissions
				const originalQuery = window.navigator.permissions.query;
				window.navigator.permissions.query = (parameters) => (
					parameters.name === 'notifications' ?
						Promise.resolve({ state: Notification.permission }) :
						originalQuery(parameters)
				);
				(() => {
					if (window._eventListenerTrackerInitialized) return;
					window._eventListenerTrackerInitialized = true;

					const originalAddEventListener = EventTarget.prototype.addEventListener;
					const eventListenersMap = new WeakMap();

					EventTarget.prototype.addEventListener = function(type, listener, options) {
						if (typeof listener === "function") {
							let listeners = eventListenersMap.get(this);
							if (!listeners) {
								listeners = [];
								eventListenersMap.set(this, listeners);
							}

							listeners.push({
								type,
								listener,
								listenerPreview: listener.toString().slice(0, 100),
								options
							});
						}

						return originalAddEventListener.call(this, type, listener, options);
					};

					window.getEventListenersForNode = (node) => {
						const listeners = eventListenersMap.get(node) || [];
						return listeners.map(({ type, listenerPreview, options }) => ({
							type,
							listenerPreview,
							options
						}));
					};
				})();
			}
		"""

		# Expose anti-detection scripts
		try:
			await self.browser_context.add_init_script(init_script)
		except Exception as e:
			if 'Target page, context or browser has been closed' in str(e):
				self.logger.warning('⚠️ Browser context was closed before init script could be added')
				# Reset connection state since browser is no longer valid
				self._reset_connection_state()
			else:
				raise

		if self.browser_profile.stealth and not isinstance(self.playwright, Patchright):
			self.logger.warning('⚠️ Failed to set up stealth mode. (...) got normal playwright objects as input.')

	# async def _fork_locked_user_data_dir(self) -> None:
	# 	"""Fork an in-use user_data_dir by cloning it to a new location to allow a second browser to use it"""
	# 	# TODO: implement copy-on-write using overlayfs or zfs or something
	# 	suffix_num = str(self.browser_profile.user_data_dir).rsplit('.', 1)[-1] or '1'
	# 	suffix_num = int(suffix_num) if suffix_num.isdigit() else 1
	# 	dir_name = self.browser_profile.user_data_dir.name
	# 	incremented_name = dir_name.replace(f'.{suffix_num}', f'.{suffix_num + 1}')
	# 	fork_path = self.browser_profile.user_data_dir.parent / incremented_name

	# 	# keep incrementing the suffix_num until we find a path that doesn't exist
	# 	while fork_path.exists():
	# 		suffix_num += 1
	# 		fork_path = self.browser_profile.user_data_dir.parent / (dir_name.rsplit('.', 1)[0] + f'.{suffix_num}')

	# 	# use shutil to recursively copy the user_data_dir to a new location
	# 	shutil.copytree(
	# 		str(self.browser_profile.user_data_dir),
	# 		str(fork_path),
	# 		symlinks=True,
	# 		ignore_dangling_symlinks=True,
	# 		dirs_exist_ok=False,
	# 	)
	# 	self.browser_profile.user_data_dir = fork_path
	# 	self.browser_profile.prepare_user_data_dir()

	@observe_debug(ignore_input=True, ignore_output=True, name='setup_current_page_change_listeners')
	async def _setup_current_page_change_listeners(self) -> None:
		# Uses a combination of:
		# - visibilitychange events
		# - window focus/blur events
		# - pointermove events

		# This annoying multi-method approach is needed for more reliable detection across browsers because playwright provides no API for this.

		# TODO: pester the playwright team to add a new event that fires when a headful tab is focused.
		# OR implement a browser-use chrome extension that acts as a bridge to the chrome.tabs API.

		#         - https://github.com/microsoft/playwright/issues/1290
		#         - https://github.com/microsoft/playwright/issues/2286
		#         - https://github.com/microsoft/playwright/issues/3570
		#         - https://github.com/microsoft/playwright/issues/13989

		# set up / detect foreground page
		assert self.browser_context is not None, 'BrowserContext object is not set'
		pages = self.browser_context.pages
		foreground_page = None
		if pages:
			foreground_page = pages[0]
			self.logger.debug(
				f'👁️‍🗨️ Found {len(pages)} existing tabs in browser, Agent 🅰 {self.id[-4:]} is on Page 🅟 {str(id(foreground_page))[-2:]}: {_log_pretty_url(foreground_page.url)}'  # type: ignore
			)
		else:
			foreground_page = await self.browser_context.new_page()
			pages = [foreground_page]
			self.logger.debug('➕ Opened new tab in empty browser context...')

		self.agent_current_page = self.agent_current_page or foreground_page
		self.human_current_page = self.human_current_page or foreground_page
		# self.logger.debug('About to define _BrowserUseonTabVisibilityChange callback')

		def _BrowserUseonTabVisibilityChange(source: dict[str, Page]):
			"""hook callback fired when init script injected into a page detects a focus event"""
			new_page = source['page']

			# Update human foreground tab state
			old_foreground = self.human_current_page
			assert self.browser_context is not None, 'BrowserContext object is not set'
			assert old_foreground is not None, 'Old foreground page is not set'
			old_tab_idx = self.browser_context.pages.index(old_foreground)  # type: ignore
			self.human_current_page = new_page
			new_tab_idx = self.browser_context.pages.index(new_page)  # type: ignore

			# Log before and after for debugging
			old_url = old_foreground and old_foreground.url or 'about:blank'
			new_url = new_page and new_page.url or 'about:blank'
			agent_url = self.agent_current_page and self.agent_current_page.url or 'about:blank'
			agent_tab_idx = self.browser_context.pages.index(self.agent_current_page)  # type: ignore
			if old_url != new_url:
				self.logger.info(
					f'👁️ Foregound tab changed by human from [{old_tab_idx}]{_log_pretty_url(old_url)} '
					f'➡️ [{new_tab_idx}]{_log_pretty_url(new_url)} '
					f'(agent will stay on [{agent_tab_idx}]{_log_pretty_url(agent_url)})'
				)

		# Store the callback so we can potentially clean it up later
		self._tab_visibility_callback = _BrowserUseonTabVisibilityChange

		# self.logger.info('About to call expose_binding')
		try:
			await self.browser_context.expose_binding('_BrowserUseonTabVisibilityChange', _BrowserUseonTabVisibilityChange)
			# self.logger.debug('window._BrowserUseonTabVisibilityChange binding attached via browser_context')
		except Exception as e:
			if 'Function "_BrowserUseonTabVisibilityChange" has been already registered' in str(e):
				self.logger.debug(
					'⚠️ Function "_BrowserUseonTabVisibilityChange" has been already registered, '
					'this is likely because the browser was already started with an existing BrowserSession()'
				)

			else:
				raise

		update_tab_focus_script = """
			// --- Method 1: visibilitychange event (unfortunately *all* tabs are always marked visible by playwright, usually does not fire) ---
			document.addEventListener('visibilitychange', async () => {
				if (document.visibilityState === 'visible') {
					await window._BrowserUseonTabVisibilityChange({ source: 'visibilitychange', url: document.location.href });
					console.log('BrowserUse Foreground tab change event fired', document.location.href);
				}
			});
			
			// --- Method 2: focus/blur events, most reliable method for headful browsers ---
			window.addEventListener('focus', async () => {
				await window._BrowserUseonTabVisibilityChange({ source: 'focus', url: document.location.href });
				console.log('BrowserUse Foreground tab change event fired', document.location.href);
			});
			
			// --- Method 3: pointermove events (may be fired by agent if we implement AI hover movements, also very noisy) ---
			// Use a throttled handler to avoid excessive calls
			// let lastMove = 0;
			// window.addEventListener('pointermove', async () => {
			// 	const now = Date.now();
			// 	if (now - lastMove > 1000) {  // Throttle to once per second
			// 		lastMove = now;
			// 		await window._BrowserUseonTabVisibilityChange({ source: 'pointermove', url: document.location.href });
			//      console.log('BrowserUse Foreground tab change event fired', document.location.href);
			// 	}
			// });
		"""
		try:
			await self.browser_context.add_init_script(update_tab_focus_script)
		except Exception as e:
			self.logger.warning(f'⚠️ Failed to register init script for tab focus detection: {e}')

		# Set up visibility listeners for all existing tabs
		# self.logger.info(f'Setting up visibility listeners for {len(self.browser_context.pages)} pages')
		for page in self.browser_context.pages:
			# self.logger.info(f'Processing page with URL: {repr(page.url)}')
			# Skip new tab pages as they can hang when evaluating scripts
			if is_new_tab_page(page.url):
				continue

			try:
				await page.evaluate(update_tab_focus_script)
				# self.logger.debug(f'👁️ Added visibility listener to existing tab: {page.url}')
			except Exception as e:
				page_idx = self.browser_context.pages.index(page)  # type: ignore
				self.logger.debug(
					f'⚠️ Failed to add visibility listener to existing tab, is it crashed or ignoring CDP commands?: [{page_idx}]{page.url}: {type(e).__name__}: {e}'
				)

	@observe_debug(
		ignore_input=True, ignore_output=True, name='setup_viewports', metadata={'browser_profile': '{{browser_profile}}'}
	)
	async def _setup_viewports(self) -> None:
		"""Resize any existing page viewports to match the configured size, set up storage_state, permissions, geolocation, etc."""

		assert self.browser_context, 'BrowserSession.browser_context must already be set up before calling _setup_viewports()'

		# log the viewport settings to terminal
		viewport = self.browser_profile.viewport
		self.logger.debug(
			'📐 Setting up viewport: '
			+ f'headless={self.browser_profile.headless} '
			+ (
				f'window={self.browser_profile.window_size["width"]}x{self.browser_profile.window_size["height"]}px '
				if self.browser_profile.window_size
				else '(no window) '
			)
			+ (
				f'screen={self.browser_profile.screen["width"]}x{self.browser_profile.screen["height"]}px '
				if self.browser_profile.screen
				else ''
			)
			+ (f'viewport={viewport["width"]}x{viewport["height"]}px ' if viewport else '(no viewport) ')
			+ f'device_scale_factor={self.browser_profile.device_scale_factor or 1.0} '
			+ f'is_mobile={self.browser_profile.is_mobile} '
			+ (f'color_scheme={self.browser_profile.color_scheme.value} ' if self.browser_profile.color_scheme else '')
			+ (f'locale={self.browser_profile.locale} ' if self.browser_profile.locale else '')
			+ (f'timezone_id={self.browser_profile.timezone_id} ' if self.browser_profile.timezone_id else '')
			+ (f'geolocation={self.browser_profile.geolocation} ' if self.browser_profile.geolocation else '')
			+ (f'permissions={",".join(self.browser_profile.permissions or ["<none>"])} ')
			+ f'storage_state={_log_pretty_path(str(self.browser_profile.storage_state or self.browser_profile.cookies_file or "<none>"))} '
		)

		# if we have any viewport settings in the profile, make sure to apply them to the entire browser_context as defaults
		if self.browser_profile.permissions:
			try:
				await self.browser_context.grant_permissions(self.browser_profile.permissions)
			except Exception as e:
				self.logger.warning(
					f'⚠️ Failed to grant browser permissions {self.browser_profile.permissions}: {type(e).__name__}: {e}'
				)
		try:
			if self.browser_profile.default_timeout:
				self.browser_context.set_default_timeout(self.browser_profile.default_timeout)
			if self.browser_profile.default_navigation_timeout:
				self.browser_context.set_default_navigation_timeout(self.browser_profile.default_navigation_timeout)
		except Exception as e:
			self.logger.warning(
				f'⚠️ Failed to set playwright timeout settings '
				f'cdp_api={self.browser_profile.default_timeout} '
				f'navigation={self.browser_profile.default_navigation_timeout}: {type(e).__name__}: {e}'
			)
		try:
			if self.browser_profile.extra_http_headers:
				await self.browser_context.set_extra_http_headers(self.browser_profile.extra_http_headers)
		except Exception as e:
			self.logger.warning(
				f'⚠️ Failed to setup playwright extra_http_headers: {type(e).__name__}: {e}'
			)  # dont print the secret header contents in the logs!

		try:
			if self.browser_profile.geolocation:
				await self.browser_context.set_geolocation(self.browser_profile.geolocation)
		except Exception as e:
			self.logger.warning(
				f'⚠️ Failed to update browser geolocation {self.browser_profile.geolocation}: {type(e).__name__}: {e}'
			)

		await self.load_storage_state()

		page = None

		for page in self.browser_context.pages:
			# apply viewport size settings to any existing pages
			if viewport:
				await page.set_viewport_size(viewport)

			# show browser-use dvd screensaver-style bouncing loading animation on any new tab pages
			if is_new_tab_page(page.url):
				# Navigate to about:blank if we're on chrome://new-tab-page to avoid security restrictions
				if page.url.startswith('chrome://new-tab-page'):
					try:
						# can raise exception if nav is interrupted by another agent nav or human, harmless but annoying
						await page.goto('about:blank', wait_until='load', timeout=5000)
					except Exception:
						pass
				await self._show_dvd_screensaver_loading_animation(page)

		page = page or (await self.browser_context.new_page())

		if (not viewport) and (self.browser_profile.window_size is not None) and not self.browser_profile.headless:
			# attempt to resize the actual browser window

			# cdp api: https://chromedevtools.github.io/devtools-protocol/tot/Browser/#method-setWindowBounds
			try:
				cdp_session = await page.context.new_cdp_session(page)  # type: ignore
				window_id_result = await cdp_session.send('Browser.getWindowForTarget')
				await cdp_session.send(
					'Browser.setWindowBounds',
					{
						'windowId': window_id_result['windowId'],
						'bounds': {
							**self.browser_profile.window_size,
							'windowState': 'normal',  # Ensure window is not minimized/maximized
						},
					},
				)
				try:
					await asyncio.wait_for(cdp_session.detach(), timeout=1.0)
				except (TimeoutError, Exception):
					pass
			except Exception as e:
				_log_size = lambda size: f'{size["width"]}x{size["height"]}px'
				try:
					# fallback to javascript resize if cdp setWindowBounds fails
					await page.evaluate(
						"""(width, height) => {window.resizeTo(width, height)}""",
						[self.browser_profile.window_size['width'], self.browser_profile.window_size['height']],
					)
					return
				except Exception:
					pass

				self.logger.warning(
					f'⚠️ Failed to resize browser window to {_log_size(self.browser_profile.window_size)} via CDP setWindowBounds: {type(e).__name__}: {e}'
				)

	def _set_browser_keep_alive(self, keep_alive: bool | None) -> None:
		"""set the keep_alive flag on the browser_profile, defaulting to True if keep_alive is None"""
		if self.browser_profile.keep_alive is None:
			self.browser_profile.keep_alive = keep_alive

	@observe_debug(ignore_input=True, ignore_output=True, name='is_connected')
	async def is_connected(self, restart: bool = True) -> bool:
		"""
		Check if the browser session has valid, connected browser and context objects.
		Returns False if any of the following conditions are met:
		- No browser_context exists
		- Browser exists but is disconnected
		- Browser_context's browser exists but is disconnected
		- Browser_context itself is closed/unusable

		Args:
			restart: If True, will attempt to create a new tab if no pages exist (valid contexts must always have at least one page open).
			        If False, will only check connection status without side effects.
		"""
		if not self.browser_context:
			return False

		if self.browser_context.browser and not self.browser_context.browser.is_connected():
			return False

		# Check if the browser_context itself is closed/unusable
		try:
			# The only reliable way to check if a browser context is still valid
			# is to try to use it. We'll try a simple page.evaluate() call.
			if self.browser_context.pages:
				# Use the first available page to test the connection
				test_page = self.browser_context.pages[0]
				# Try a simple evaluate to check if the connection is alive
				result = await test_page.evaluate('() => true')
				return result is True
			elif restart:
				# Create new page directly to avoid using decorated methods
				new_page = await self.browser_context.new_page()
				self.agent_current_page = new_page
				if (not self.human_current_page) or self.human_current_page.is_closed():
					self.human_current_page = new_page
				# Test the new tab
				if self.browser_context.pages:
					test_page = self.browser_context.pages[0]
					result = await test_page.evaluate('() => true')
					return result is True
				return False
			else:
				return False
		except Exception:
			# Any exception means the context is closed or invalid
			return False

	def _reset_connection_state(self) -> None:
		"""Reset the browser connection state when disconnection is detected"""

		already_disconnected = not any(
			(
				self.initialized,
				self.browser,
				self.browser_context,
				self.agent_current_page,
				self.human_current_page,
				self._cached_clickable_element_hashes,
				self._cached_browser_state_summary,
			)
		)

		self.initialized = False
		self.browser = None
		self.browser_context = None
		self.agent_current_page = None
		self.human_current_page = None
		self._cached_clickable_element_hashes = None
		# Reset CDP connection info when browser is stopped
		self.browser_pid = None
		self._cached_browser_state_summary = None
		# Don't clear self.playwright here - it should be cleared explicitly in kill()

		if self.browser_pid:
			try:
				# browser_pid is different from all the other state objects, it's closer to cdp_url or wss_url
				# because we might still be able to reconnect to the same browser even if self.browser_context died
				# if we have a self.browser_pid, check if it's still alive and serving a remote debugging port
				# if so, don't clear it because there's a chance we can re-use it by just reconnecting to the same pid's port
				proc = psutil.Process(self.browser_pid)
				proc_is_alive = proc.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
				assert proc_is_alive and '--remote-debugging-port' in ' '.join(proc.cmdline())
			except Exception:
				self.logger.info(f' ↳ Browser browser_pid={self.browser_pid} process is no longer running')
				# process has gone away or crashed, pid is no longer valid so we clear it
				self.browser_pid = None

		if not already_disconnected:
			self.logger.debug(f'⚰️ Browser {self._connection_str} disconnected')

	def _check_for_singleton_lock_conflict(self) -> bool:
		"""Check if the user data directory has a conflicting browser process.

		Returns:
			True if there's a conflict (active process using this profile), False otherwise
		"""
		if not self.browser_profile.user_data_dir:
			return False

		# Normalize the path for comparison
		target_dir = str(Path(self.browser_profile.user_data_dir).expanduser().resolve())

		# Check for running processes using this user data dir
		for proc in psutil.process_iter(['pid', 'cmdline']):
			# Skip our own browser process
			if hasattr(self, 'browser_pid') and self.browser_pid and proc.info['pid'] == self.browser_pid:
				continue

			cmdline = proc.info['cmdline'] or []

			# Check both formats: --user-data-dir=/path and --user-data-dir /path
			for i, arg in enumerate(cmdline):
				# Combined format: --user-data-dir=/path
				if arg.startswith('--user-data-dir='):
					try:
						cmd_path = str(Path(arg.split('=', 1)[1]).expanduser().resolve())
						if cmd_path == target_dir:
							self.logger.debug(
								f'🔍 Found conflicting Chrome process PID {proc.info["pid"]} using profile {_log_pretty_path(self.browser_profile.user_data_dir)}'
							)
							return True
					except Exception:
						# Fallback to string comparison if path resolution fails
						if arg.split('=', 1)[1] == str(self.browser_profile.user_data_dir):
							self.logger.debug(
								f'🔍 Found conflicting Chrome process PID {proc.info["pid"]} using profile {_log_pretty_path(self.browser_profile.user_data_dir)}'
							)
							return True
				# Separate format: --user-data-dir /path
				elif arg == '--user-data-dir' and i + 1 < len(cmdline):
					try:
						cmd_path = str(Path(cmdline[i + 1]).expanduser().resolve())
						if cmd_path == target_dir:
							self.logger.debug(
								f'🔍 Found conflicting Chrome process PID {proc.info["pid"]} using profile {_log_pretty_path(self.browser_profile.user_data_dir)}'
							)
							return True
					except Exception:
						# Fallback to string comparison if path resolution fails
						if cmdline[i + 1] == str(self.browser_profile.user_data_dir):
							self.logger.debug(
								f'🔍 Found conflicting Chrome process PID {proc.info["pid"]} using profile {_log_pretty_path(self.browser_profile.user_data_dir)}'
							)
							return True

		# Note: We don't consider a SingletonLock file alone as a conflict
		# because it might be stale. Only actual running processes count as conflicts.
		return False

	def _fallback_to_temp_profile(self, reason: str = 'SingletonLock conflict') -> None:
		"""Fallback to a temporary profile directory when the current one is locked.

		Args:
			reason: Human-readable reason for the fallback
		"""
		old_dir = self.browser_profile.user_data_dir
		self.browser_profile.user_data_dir = Path(tempfile.mkdtemp(prefix='browseruse-tmp-singleton-'))
		self.logger.warning(
			f'⚠️ {reason} detected. Profile at {_log_pretty_path(old_dir)} is locked. '
			f'Using temporary profile instead: {_log_pretty_path(self.browser_profile.user_data_dir)}'
		)

	@observe_debug(ignore_input=True, ignore_output=True, name='prepare_user_data_dir')
	def prepare_user_data_dir(self, check_conflicts: bool = True) -> None:
		"""Create and prepare the user data dir, handling conflicts if needed.

		Args:
			check_conflicts: Whether to check for and handle singleton lock conflicts
		"""
		if self.browser_profile.user_data_dir:
			try:
				self.browser_profile.user_data_dir = Path(self.browser_profile.user_data_dir).expanduser().resolve()
				self.browser_profile.user_data_dir.mkdir(parents=True, exist_ok=True)
				(self.browser_profile.user_data_dir / '.browseruse_profile_id').write_text(self.browser_profile.id)
			except Exception as e:
				raise ValueError(
					f'Unusable path provided for user_data_dir= {_log_pretty_path(self.browser_profile.user_data_dir)} (check for typos/permissions issues)'
				) from e

			# Remove stale singleton lock file ONLY if no process is using this profile
			# This must happen BEFORE checking for conflicts to avoid false positives
			singleton_lock = self.browser_profile.user_data_dir / 'SingletonLock'
			if singleton_lock.exists():
				# Check if any process is actually using this user_data_dir
				has_active_process = False
				target_dir = str(self.browser_profile.user_data_dir)
				for proc in psutil.process_iter(['pid', 'cmdline']):
					# Skip our own browser process
					if hasattr(self, 'browser_pid') and self.browser_pid and proc.info['pid'] == self.browser_pid:
						continue

					cmdline = proc.info['cmdline'] or []
					# Check both formats: --user-data-dir=/path and --user-data-dir /path
					for i, arg in enumerate(cmdline):
						if arg.startswith('--user-data-dir='):
							try:
								if str(Path(arg.split('=', 1)[1]).expanduser().resolve()) == target_dir:
									has_active_process = True
									break
							except Exception:
								if arg.split('=', 1)[1] == str(self.browser_profile.user_data_dir):
									has_active_process = True
									break
						elif arg == '--user-data-dir' and i + 1 < len(cmdline):
							try:
								if str(Path(cmdline[i + 1]).expanduser().resolve()) == target_dir:
									has_active_process = True
									break
							except Exception:
								if cmdline[i + 1] == str(self.browser_profile.user_data_dir):
									has_active_process = True
									break
					if has_active_process:
						break

				if not has_active_process:
					# No active process, safe to remove stale lock
					try:
						# Handle both regular files and symlinks
						if singleton_lock.is_symlink() or singleton_lock.exists():
							singleton_lock.unlink()
							self.logger.debug(
								f'🧹 Removed stale SingletonLock file from {_log_pretty_path(self.browser_profile.user_data_dir)} (no active Chrome process found)'
							)
					except Exception:
						pass  # Ignore errors removing lock file

			# Check for conflicts and fallback if needed (AFTER cleaning stale locks)
			if check_conflicts and self._check_for_singleton_lock_conflict():
				self._fallback_to_temp_profile()
				# Recursive call without conflict checking to prepare the new temp dir
				return self.prepare_user_data_dir(check_conflicts=False)

		# Create directories for all paths that need them
		dir_paths = {
			'downloads_path': self.browser_profile.downloads_path,
			'record_video_dir': self.browser_profile.record_video_dir,
			'traces_dir': self.browser_profile.traces_dir,
		}

		file_paths = {
			'record_har_path': self.browser_profile.record_har_path,
		}

		# Handle directory creation
		for path_name, path_value in dir_paths.items():
			if path_value:
				try:
					path_obj = Path(path_value).expanduser().resolve()
					path_obj.mkdir(parents=True, exist_ok=True)
					setattr(self.browser_profile, path_name, str(path_obj) if path_name == 'traces_dir' else path_obj)
				except Exception as e:
					self.logger.error(f'❌ Failed to create {path_name} directory {path_value}: {e}')

		# Handle file path parent directory creation
		for path_name, path_value in file_paths.items():
			if path_value:
				try:
					path_obj = Path(path_value).expanduser().resolve()
					path_obj.parent.mkdir(parents=True, exist_ok=True)
				except Exception as e:
					self.logger.error(f'❌ Failed to create parent directory for {path_name} {path_value}: {e}')

	# --- Tab management ---
	@observe_debug(ignore_input=True, ignore_output=True, name='get_current_page')
	async def get_current_page(self) -> Page:
		"""Get the current page + ensure it's not None / closed"""

		if not self.initialized:
			await self.start()

		# get-or-create the browser_context if it's not already set up
		if not self.browser_context:
			await self.start()
			assert self.browser_context, 'BrowserContext is not set up'

		# if either focused page is closed, clear it so we dont use a dead object
		if (not self.human_current_page) or self.human_current_page.is_closed():
			self.human_current_page = None
		if (not self.agent_current_page) or self.agent_current_page.is_closed():
			self.agent_current_page = None

		# if either one is None, fallback to using the other one for both
		self.agent_current_page = self.agent_current_page or self.human_current_page or None
		self.human_current_page = self.human_current_page or self.agent_current_page or None

		# if both are still None, fallback to using the first open tab we can find
		if self.agent_current_page is None:
			if self.browser_context.pages:
				first_available_tab = self.browser_context.pages[0]
				self.agent_current_page = first_available_tab
				self.human_current_page = first_available_tab
			else:
				# if all tabs are closed, open a new one, never allow a context with 0 tabs
				new_page = await self.browser_context.new_page()
				self.agent_current_page = new_page
				self.human_current_page = new_page
				if self.browser_profile.viewport:
					await new_page.set_viewport_size(self.browser_profile.viewport)

		assert self.agent_current_page is not None, f'{self} Failed to find or create a new page for the agent'
		assert self.human_current_page is not None, f'{self} Failed to find or create a new page for the human'

		return self.agent_current_page

	@property
	def tabs(self) -> list[Page]:
		if not self.browser_context:
			return []
		return list(self.browser_context.pages)

	@require_healthy_browser(usable_page=False, reopen_page=False)
	async def switch_tab(self, tab_index: int) -> Page:
		assert self.browser_context is not None, 'BrowserContext is not set up'
		pages = self.browser_context.pages
		if not pages or tab_index >= len(pages):
			raise IndexError('Tab index out of range')
		page = pages[tab_index]
		self.agent_current_page = page

		# Invalidate cached state since we've switched to a different tab
		# The cached state contains DOM elements and selector map from the previous tab
		self._cached_browser_state_summary = None
		self._cached_clickable_element_hashes = None

		return page

	@require_healthy_browser(usable_page=True, reopen_page=True)
	async def wait_for_element(self, selector: str, timeout: int = 10000) -> None:
		page = await self.get_current_page()
		await page.wait_for_selector(selector, state='visible', timeout=timeout)

	@observe_debug(name='remove_highlights', ignore_output=True, ignore_input=True)
	@time_execution_async('--remove_highlights')
	@retry(timeout=2, retries=0)
	async def remove_highlights(self):
		"""
		Removes all highlight overlays and labels created by the highlightElement function.
		Handles cases where the page might be closed or inaccessible.
		"""
		page = await self.get_current_page()
		try:
			await page.evaluate(
				"""
				try {
					// Remove the highlight container and all its contents
					const container = document.getElementById('playwright-highlight-container');
					if (container) {
						container.remove();
					}

					// Remove highlight attributes from elements
					const highlightedElements = document.querySelectorAll('[browser-user-highlight-id^="playwright-highlight-"]');
					highlightedElements.forEach(el => {
						el.removeAttribute('browser-user-highlight-id');
					});
				} catch (e) {
					console.error('Failed to remove highlights:', e);
				}
				"""
			)
		except Exception as e:
			self.logger.debug(f'⚠️ Failed to remove highlights (this is usually ok): {type(e).__name__}: {e}')
			# Don't raise the error since this is not critical functionality

	@observe_debug(ignore_output=True, name='get_dom_element_by_index')
	@require_healthy_browser(usable_page=True, reopen_page=True)
	async def get_dom_element_by_index(self, index: int) -> DOMElementNode | None:
		"""Get DOM element by index."""
		selector_map = await self.get_selector_map()
		return selector_map.get(index)

	@time_execution_async('--click_element_node')
	@observe_debug(ignore_input=True, name='click_element_node')
	@require_healthy_browser(usable_page=True, reopen_page=True)
	async def _click_element_node(self, element_node: DOMElementNode) -> str | None:
		"""
		Optimized method to click an element using xpath.
		"""
		page = await self.get_current_page()
		try:
			# Highlight before clicking
			# if element_node.highlight_index is not None:
			# 	await self._update_state(focus_element=element_node.highlight_index)

			element_handle = await self.get_locate_element(element_node)

			if element_handle is None:
				self.logger.debug(f'Element: {repr(element_node)} not found')
				raise Exception('Element not found')

			async def perform_click(click_func):
				"""Performs the actual click, handling both download and navigation scenarios."""

				# only wait the 5s extra for potential downloads if they are enabled
				# TODO: instead of blocking for 5s, we should register a non-block page.on('download') event
				# and then check if the download has been triggered within the event handler
				if self.browser_profile.downloads_path:
					try:
						# Try short-timeout expect_download to detect a file download has been been triggered
						async with page.expect_download(timeout=5_000) as download_info:
							await click_func()
						download = await download_info.value
						# Determine file path
						suggested_filename = download.suggested_filename
						unique_filename = await self._get_unique_filename(self.browser_profile.downloads_path, suggested_filename)
						download_path = os.path.join(self.browser_profile.downloads_path, unique_filename)
						await download.save_as(download_path)
						self.logger.info(f'⬇️ Downloaded file to: {download_path}')

						# Track the downloaded file in the session
						self._downloaded_files.append(download_path)
						self.logger.info(f'📁 Added download to session tracking (total: {len(self._downloaded_files)} files)')

						return download_path
					except Exception:
						# If no download is triggered, treat as normal click
						self.logger.debug('No download triggered within timeout. Checking navigation...')
						try:
							await page.wait_for_load_state()
						except Exception as e:
							self.logger.warning(
								f'⚠️ Page {_log_pretty_url(page.url)} failed to finish loading after click: {type(e).__name__}: {e}'
							)
						await self._check_and_handle_navigation(page)
				else:
					# If downloads are disabled, just perform the click
					await click_func()
					try:
						await page.wait_for_load_state()
					except Exception as e:
						self.logger.warning(
							f'⚠️ Page {_log_pretty_url(page.url)} failed to finish loading after click: {type(e).__name__}: {e}'
						)
					await self._check_and_handle_navigation(page)

			try:
				return await perform_click(lambda: element_handle and element_handle.click(timeout=1_500))
			except URLNotAllowedError as e:
				raise e
			except Exception as e:
				# Check if it's a context error and provide more info
				if 'Cannot find context with specified id' in str(e) or 'Protocol error' in str(e):
					self.logger.warning(f'⚠️ Element context lost, attempting to re-locate element: {type(e).__name__}')
					# Try to re-locate the element
					element_handle = await self.get_locate_element(element_node)
					if element_handle is None:
						raise Exception(f'Element no longer exists in DOM after context loss: {repr(element_node)}')
					# Try click again with fresh element
					try:
						return await perform_click(lambda: element_handle.click(timeout=1_500))
					except Exception:
						# Fall back to JavaScript click
						return await perform_click(lambda: page.evaluate('(el) => el.click()', element_handle))
				else:
					# Original fallback for other errors
					try:
						return await perform_click(lambda: page.evaluate('(el) => el.click()', element_handle))
					except URLNotAllowedError as e:
						raise e
					except Exception as e:
						# Final fallback - try clicking by coordinates if available
						if element_node.viewport_coordinates and element_node.viewport_coordinates.center:
							try:
								self.logger.warning(
									f'⚠️ Element click failed, falling back to coordinate click at ({element_node.viewport_coordinates.center.x}, {element_node.viewport_coordinates.center.y})'
								)
								await page.mouse.click(
									element_node.viewport_coordinates.center.x, element_node.viewport_coordinates.center.y
								)
								try:
									await page.wait_for_load_state()
								except Exception:
									pass
								await self._check_and_handle_navigation(page)
								return None  # Success
							except Exception as coord_e:
								self.logger.error(f'Coordinate click also failed: {type(coord_e).__name__}: {coord_e}')
						raise Exception(f'Failed to click element: {type(e).__name__}: {e}')

		except URLNotAllowedError as e:
			raise e
		except Exception as e:
			raise Exception(f'Failed to click element. Error: {str(e)}')

	@time_execution_async('--get_tabs_info')
	@retry(timeout=3, retries=1)
	@require_healthy_browser(usable_page=False, reopen_page=False)
	async def get_tabs_info(self) -> list[TabInfo]:
		"""Get information about all tabs"""
		assert self.browser_context is not None, 'BrowserContext is not set up'
		tabs_info = []
		for page_id, page in enumerate(self.browser_context.pages):
			# Skip JS execution for chrome:// pages and new tab pages
			if is_new_tab_page(page.url) or page.url.startswith('chrome://'):
				# Use URL as title for chrome pages, or mark new tabs as unusable
				if is_new_tab_page(page.url):
					tab_info = TabInfo(page_id=page_id, url=page.url, title='ignore this tab and do not use it')
				else:
					# For chrome:// pages, use the URL itself as the title
					tab_info = TabInfo(page_id=page_id, url=page.url, title=page.url)
				tabs_info.append(tab_info)
				continue

			# Normal pages - try to get title with timeout
			try:
				title = await asyncio.wait_for(page.title(), timeout=2.0)
				tab_info = TabInfo(page_id=page_id, url=page.url, title=title)
			except Exception:
				# page.title() can hang forever on tabs that are crashed/disappeared/about:blank
				# but we should preserve the real URL and not mislead the LLM about tab availability
				self.logger.debug(
					f'⚠️ Failed to get tab info for tab #{page_id}: {_log_pretty_url(page.url)} (using fallback title)'
				)

				# Only mark as unusable if it's actually a new tab page, otherwise preserve the real URL
				if is_new_tab_page(page.url):
					tab_info = TabInfo(page_id=page_id, url=page.url, title='ignore this tab and do not use it')
				else:
					# Preserve the real URL and use a descriptive fallback title
					# fallback_title = '(title unavailable, page possibly crashed / unresponsive)'
					# tab_info = TabInfo(page_id=page_id, url=page.url, title=fallback_title)

					# harsh but good, just close the page here because if we cant get the title then we certainly cant do anything else useful with it, no point keeping it open
					try:
						await page.close()
						self.logger.debug(
							f'🪓 Force-closed 🅟 {str(id(page))[-2:]} because its JS engine is unresponsive via CDP: {_log_pretty_url(page.url)}'
						)
					except Exception:
						pass
					continue

			tabs_info.append(tab_info)

		return tabs_info

	@retry(timeout=20, retries=1, semaphore_limit=1, semaphore_scope='self')
	async def _set_viewport_size(self, page: Page, viewport: dict[str, int] | ViewportSize) -> None:
		"""Set viewport size with timeout protection."""
		if isinstance(viewport, dict):
			await page.set_viewport_size(ViewportSize(width=viewport['width'], height=viewport['height']))
		else:
			await page.set_viewport_size(viewport)

	@require_healthy_browser(usable_page=False, reopen_page=False)
	async def close_tab(self, tab_index: int | None = None) -> None:
		assert self.browser_context is not None, 'BrowserContext is not set up'
		pages = self.browser_context.pages
		if not pages:
			return

		if tab_index is None:
			# to tab_index passed, just close the current agent page
			page = await self.get_current_page()
		else:
			# otherwise close the tab at the given index
			if tab_index >= len(pages) or tab_index < 0:
				raise IndexError(f'Tab index {tab_index} out of range. Available tabs: {len(pages)}')
			page = pages[tab_index]

		await page.close()

		# reset the self.agent_current_page and self.human_current_page references to first available tab
		await self.get_current_page()

	# --- Page navigation ---
	@observe_debug(ignore_input=True, ignore_output=True)
	@retry(retries=0, timeout=30, wait=1, semaphore_timeout=10, semaphore_limit=1, semaphore_scope='self', semaphore_lax=True)
	@require_healthy_browser(usable_page=False, reopen_page=False)
	async def navigate(self, url: str = 'about:blank', new_tab: bool = False, timeout_ms: int | None = None) -> Page:
		"""
		Universal navigation method that handles all navigation scenarios.

		Args:
			url: URL to navigate to (defaults to 'about:blank')
			new_tab: If True, creates a new tab for navigation

		Returns:
			Page: The page that was navigated
		"""
		# Clear loading status from previous page
		self._current_page_loading_status = None

		# Normalize the URL
		normalized_url = normalize_url(url)

		# Check if URL is allowed
		if not self._is_url_allowed(normalized_url):
			raise BrowserError(f'⛔️ Navigation to non-allowed URL: {normalized_url}')
		# If timeout_ms is not None, use it (even if 0); else try profile.default_navigation_timeout (even if 0); else 12000
		if timeout_ms is not None:
			user_timeout_ms = int(timeout_ms)
		elif self.browser_profile.default_navigation_timeout is not None:
			user_timeout_ms = int(self.browser_profile.default_navigation_timeout)
		else:
			user_timeout_ms = 12000
		timeout_ms = min(3000, user_timeout_ms)

		# Handle new tab creation
		if new_tab:
			# Create new tab
			assert self.browser_context is not None, 'Browser context is not set'
			self.agent_current_page = await self.browser_context.new_page()

			# Update human tab reference if there is no human tab yet
			if (not self.human_current_page) or self.human_current_page.is_closed():
				self.human_current_page = self.agent_current_page

			# Set viewport for new tab
			if self.browser_profile.viewport:
				await self.agent_current_page.set_viewport_size(self.browser_profile.viewport)

			page = self.agent_current_page
		else:
			# Use existing page
			page = await self.get_current_page()

		# Navigate to URL
		try:
			# Use asyncio.wait to prevent hanging on a slow page loads
			# Don't cap the timeout - respect what was requested
			self.logger.debug(f'🧭 Starting navigation to {_log_pretty_url(normalized_url)} with timeout {timeout_ms}ms')
			nav_task = asyncio.create_task(page.goto(normalized_url, wait_until='load', timeout=timeout_ms))
			done, pending = await asyncio.wait([nav_task], timeout=(timeout_ms + 500) / 1000)

			if nav_task in pending:
				# Navigation timed out
				self.logger.warning(
					f"⚠️ Loading {_log_pretty_url(normalized_url)} didn't finish after {timeout_ms / 1000}s, continuing anyway..."
				)
				nav_task.cancel()
				try:
					await nav_task
				except asyncio.CancelledError:
					pass

				# Check if page is still usable after timeout
				if page and not page.is_closed():
					current_url = page.url
					# self.logger.debug(f'🤌 Checking responsiveness after navigation timeout (current URL: {current_url})')
					is_responsive = await self._is_page_responsive(page, timeout=3.0)
					if is_responsive:
						self.logger.debug(
							f'✅ Page is responsive and usable despite navigation loading timeout on: {_log_pretty_url(current_url)})'
						)
					else:
						self.logger.error(
							f'❌ Page is unresponsive after navigation stalled on: {_log_pretty_url(current_url)} WARNING! Subsequent operations will likely fail on this page, it must be reset...'
						)
						# Don't try complex recovery during navigate - just raise the error
						# The retry decorator will handle retries, and other methods with
						# @require_healthy_browser(reopen_page=True) will trigger proper recovery
						raise RuntimeError(
							f'Page JS engine is unresponsive after navigation / loading issue on: {_log_pretty_url(current_url)}). Agent cannot proceed with this page because its JS event loop is unresponsive.'
						)
			elif nav_task in done:
				# Navigation completed, check if it succeeded
				await nav_task  # This will raise if navigation failed
		except Exception as e:
			if 'timeout' in str(e).lower():
				# self.logger.warning(
				# 	f"⚠️ Loading {_log_pretty_url(normalized_url)} didn't finish and further operations may fail on this page..."
				# )
				pass  # allow agent to attempt to continue without raising hard error, it can use tools to work around it
			else:
				raise

		# Show DVD animation on new tab pages if no URL specified
		if new_tab and is_new_tab_page(page.url):
			# Navigate to about:blank if we're on chrome://new-tab-page to avoid security restrictions
			if page.url.startswith('chrome://new-tab-page'):
				try:
					await page.goto('about:blank', wait_until='load', timeout=timeout_ms)
				except Exception:
					pass
			await self._show_dvd_screensaver_loading_animation(page)

		return page

	@deprecated('Use BrowserSession.navigate(url) instead of .navigate_to(url)')
	async def navigate_to(self, url: str) -> Page:
		"""Backward compatibility alias for navigate()"""
		return await self.navigate(url=url, new_tab=False)

	@deprecated('Use BrowserSession.navigate(url=url, new_tab=True) instead of .create_new_tab(url)')
	async def create_new_tab(self, url: str | None = None) -> Page:
		"""Backward compatibility alias for navigate()"""
		return await self.navigate(url=url or 'about:blank', new_tab=True)

	@deprecated('Use BrowserSession.navigate(url=url, new_tab=True) instead of .new_tab(url)')
	async def new_tab(self, url: str | None = None) -> Page:
		"""Backward compatibility alias for navigate()"""
		return await self.navigate(url=url or 'about:blank', new_tab=True)

	@require_healthy_browser(usable_page=True, reopen_page=True)
	async def refresh(self) -> None:
		if self.agent_current_page and not self.agent_current_page.is_closed():
			await self.agent_current_page.reload()
		else:
			# Create new page directly
			assert self.browser_context is not None, 'Browser context is not set'
			new_page = await self.browser_context.new_page()
			self.agent_current_page = new_page
			if (not self.human_current_page) or self.human_current_page.is_closed():
				self.human_current_page = new_page
			if self.browser_profile.viewport:
				await new_page.set_viewport_size(self.browser_profile.viewport)

	@require_healthy_browser(usable_page=True, reopen_page=True)
	async def execute_javascript(self, script: str) -> Any:
		page = await self.get_current_page()
		return await page.evaluate(script)

	async def get_cookies(self) -> list[dict[str, Any]]:
		if self.browser_context:
			return [dict(x) for x in await self.browser_context.cookies()]
		return []

	@deprecated('Use BrowserSession.save_storage_state() instead')
	async def save_cookies(self, *args, **kwargs) -> None:
		"""
		Old name for the new save_storage_state() function.
		"""
		await self.save_storage_state(*args, **kwargs)

	async def _save_cookies_to_file(self, path: Path, cookies: list[dict[str, Any]] | None) -> None:
		if not (path or self.browser_profile.cookies_file):
			return

		if not cookies:
			return

		try:
			cookies_file_path = Path(path or self.browser_profile.cookies_file).expanduser().resolve()
			cookies_file_path.parent.mkdir(parents=True, exist_ok=True)

			# Write to a temporary file first
			cookies = cookies or []
			temp_path = cookies_file_path.with_suffix('.tmp')
			temp_path.write_text(json.dumps(cookies, indent=4))

			try:
				# backup any existing cookies_file if one is already present
				cookies_file_path.replace(cookies_file_path.with_suffix('.json.bak'))
			except Exception:
				pass
			temp_path.replace(cookies_file_path)

			self.logger.info(f'🍪 Saved {len(cookies)} cookies to cookies_file= {_log_pretty_path(cookies_file_path)}')
		except Exception as e:
			self.logger.warning(
				f'❌ Failed to save cookies to cookies_file= {_log_pretty_path(cookies_file_path)}: {type(e).__name__}: {e}'
			)

	async def _save_storage_state_to_file(self, path: str | Path, storage_state: dict[str, Any] | None) -> None:
		try:
			json_path = Path(path).expanduser().resolve()
			json_path.parent.mkdir(parents=True, exist_ok=True)
			assert self.browser_context is not None, 'BrowserContext is not set up'
			storage_state = storage_state or dict(await self.browser_context.storage_state())

			# always atomic merge storage states, never overwrite (so two browsers can share the same storage_state.json)
			merged_storage_state = storage_state
			if json_path.exists():
				try:
					existing_storage_state = json.loads(json_path.read_text())
					merged_storage_state = merge_dicts(existing_storage_state, storage_state)
				except Exception as e:
					self.logger.error(
						f'❌ Failed to merge cookie changes with existing storage_state= {_log_pretty_path(json_path)}: {type(e).__name__}: {e}'
					)
					return

			# write to .tmp file first to avoid partial writes, then mv original to .bak and .tmp to original
			temp_path = json_path.with_suffix('.json.tmp')
			temp_path.write_text(json.dumps(merged_storage_state, indent=4))
			try:
				json_path.replace(json_path.with_suffix('.json.bak'))
			except Exception:
				pass
			temp_path.replace(json_path)

			self.logger.info(
				f'🍪 Saved {len(storage_state["cookies"]) + len(storage_state.get("origins", []))} cookies to storage_state= {_log_pretty_path(json_path)}'
			)
		except Exception as e:
			self.logger.warning(f'❌ Failed to save cookies to storage_state= {_log_pretty_path(path)}: {type(e).__name__}: {e}')

	@retry(
		timeout=5, retries=1, semaphore_limit=1, semaphore_scope='self'
	)  # users can share JSON between browsers, this should really be 'multiprocess' not 'self
	async def save_storage_state(self, path: Path | None = None) -> None:
		"""
		Save cookies to the specified path or the configured cookies_file and/or storage_state.
		"""
		await self._unsafe_save_storage_state(path)

	async def _unsafe_save_storage_state(self, path: Path | None = None) -> None:
		"""
		Unsafe storage state save logic without retry protection.
		"""
		if not (path or self.browser_profile.storage_state or self.browser_profile.cookies_file):
			return

		assert self.browser_context is not None, 'BrowserContext is not set up'
		storage_state: dict[str, Any] = dict(await self.browser_context.storage_state())
		cookies = storage_state['cookies']
		has_any_auth_data = cookies or storage_state.get('origins', [])

		# they passed an explicit path, only save to that path and return
		if path and has_any_auth_data:
			if path.name == 'storage_state.json':
				await self._save_storage_state_to_file(path, storage_state)
				return
			else:
				# assume they're using the old API when path meant a cookies_file path,
				# also save new format next to it for convenience to help them migrate
				await self._save_cookies_to_file(path, cookies)
				await self._save_storage_state_to_file(path.parent / 'storage_state.json', storage_state)
				new_path = path.parent / 'storage_state.json'
				self.logger.warning(
					'⚠️ cookies_file is deprecated and will be removed in a future version. '
					f'Please use storage_state="{_log_pretty_path(new_path)}" instead for persisting cookies and other browser state. '
					'See: https://playwright.dev/python/docs/api/class-browsercontext#browser-context-storage-state'
				)
				return

		# save cookies_file if passed a cookies file path or if profile cookies_file is configured
		if cookies and self.browser_profile.cookies_file:
			# only show warning if they configured cookies_file (not if they passed in a path to this function as an arg)
			await self._save_cookies_to_file(self.browser_profile.cookies_file, cookies)
			new_path = self.browser_profile.cookies_file.parent / 'storage_state.json'
			await self._save_storage_state_to_file(new_path, storage_state)
			self.logger.warning(
				'⚠️ cookies_file is deprecated and will be removed in a future version. '
				f'Please use storage_state="{_log_pretty_path(new_path)}" instead for persisting cookies and other browser state. '
				'See: https://playwright.dev/python/docs/api/class-browsercontext#browser-context-storage-state'
			)

		if self.browser_profile.storage_state is None:
			return

		if isinstance(self.browser_profile.storage_state, dict):
			# cookies that never get updated rapidly expire or become invalid,
			# e.g. cloudflare bumps a nonce + does a tiny proof-of-work chain on every request that gets stored back into the cookie
			# if your cookies are frozen in time and don't update, they'll block you as a bot almost immediately
			# if they pass a dict in it means they have to get the updated cookies manually with browser_context.cookies()
			# and persist them manually on every change. most people don't realize they have to do that, so show a warning
			self.logger.warning(
				f'⚠️ storage_state was set as a {type(self.browser_profile.storage_state)} and will not be updated with any cookie changes, use a json file path instead to persist changes'
			)
			return

		if isinstance(self.browser_profile.storage_state, (str, Path)):
			await self._save_storage_state_to_file(self.browser_profile.storage_state, storage_state)
			return

		raise Exception(f'Got unexpected type for storage_state: {type(self.browser_profile.storage_state)}')

	async def load_storage_state(self) -> None:
		"""
		Load cookies from the storage_state or cookies_file and apply them to the browser context.
		"""

		assert self.browser_context, 'Browser context is not initialized, cannot load storage state'

		if self.browser_profile.cookies_file:
			# Show deprecation warning
			self.logger.warning(
				'⚠️ cookies_file is deprecated and will be removed in a future version. '
				'Please use storage_state instead for loading cookies and other browser state. '
				'See: https://playwright.dev/python/docs/api/class-browsercontext#browser-context-storage-state'
			)

			cookies_path = Path(self.browser_profile.cookies_file).expanduser()
			if not cookies_path.is_absolute():
				cookies_path = Path(self.browser_profile.downloads_path or '.').expanduser().resolve() / cookies_path.name

			try:
				cookies_data = json.loads(cookies_path.read_text())
				if cookies_data:
					await self.browser_context.add_cookies(cookies_data)
					self.logger.info(f'🍪 Loaded {len(cookies_data)} cookies from cookies_file= {_log_pretty_path(cookies_path)}')
			except Exception as e:
				self.logger.warning(
					f'❌ Failed to load cookies from cookies_file= {_log_pretty_path(cookies_path)}: {type(e).__name__}: {e}'
				)

		if self.browser_profile.storage_state:
			storage_state = self.browser_profile.storage_state
			if isinstance(storage_state, (str, Path)):
				try:
					storage_state_text = await anyio.Path(storage_state).read_text()
					storage_state = dict(json.loads(storage_state_text))
				except Exception as e:
					self.logger.warning(
						f'❌ Failed to load cookies from storage_state= {_log_pretty_path(storage_state)}: {type(e).__name__}: {e}'
					)
					return

			try:
				assert isinstance(storage_state, dict), f'Got unexpected type for storage_state: {type(storage_state)}'
				await self.browser_context.add_cookies(storage_state['cookies'])
				# TODO: also handle localStroage, IndexedDB, SessionStorage
				# playwright doesn't provide an API for setting these before launch
				# https://playwright.dev/python/docs/auth#session-storage
				# await self.browser_context.add_local_storage(storage_state['localStorage'])
				num_entries = len(storage_state['cookies']) + len(storage_state.get('origins', []))
				if num_entries:
					self.logger.info(f'🍪 Loaded {num_entries} cookies from storage_state= {storage_state}')
			except Exception as e:
				self.logger.warning(f'❌ Failed to load cookies from storage_state= {storage_state}: {type(e).__name__}: {e}')
				return

	async def load_cookies_from_file(self, *args, **kwargs) -> None:
		"""
		Old name for the new load_storage_state() function.
		"""
		await self.load_storage_state(*args, **kwargs)

	@property
	def downloaded_files(self) -> list[str]:
		"""
		Get list of all files downloaded during this browser session.

		Returns:
		    list[str]: List of absolute file paths to downloaded files
		"""
		self.logger.debug(f'📁 Retrieved {len(self._downloaded_files)} downloaded files from session tracking')
		return self._downloaded_files.copy()

	def set_auto_download_pdfs(self, enabled: bool) -> None:
		"""
		Enable or disable automatic PDF downloading when PDFs are encountered.

		Args:
		    enabled: Whether to automatically download PDFs
		"""
		self._auto_download_pdfs = enabled
		self.logger.info(f'📄 PDF auto-download {"enabled" if enabled else "disabled"}')

	@property
	def auto_download_pdfs(self) -> bool:
		"""Get current PDF auto-download setting."""
		return self._auto_download_pdfs

	# @property
	# def browser_extension_pages(self) -> list[Page]:
	# 	if not self.browser_context:
	# 		return []
	# 	return [p for p in self.browser_context.pages if p.url.startswith('chrome-extension://')]

	# @property
	# def saved_downloads(self) -> list[Path]:
	# 	"""
	# 	Return a list of files in the downloads_path.
	# 	"""
	# 	return list(Path(self.browser_profile.downloads_path).glob('*'))

	async def _wait_for_stable_network(self):
		pending_requests = set()
		last_activity = asyncio.get_event_loop().time()

		page = await self.get_current_page()

		# Define relevant resource types and content types
		RELEVANT_RESOURCE_TYPES = {
			'document',
			'stylesheet',
			'image',
			'font',
			'script',
			'iframe',
		}

		RELEVANT_CONTENT_TYPES = {
			'text/html',
			'text/css',
			'application/javascript',
			'image/',
			'font/',
			'application/json',
		}

		# Additional patterns to filter out
		IGNORED_URL_PATTERNS = {
			# Analytics and tracking
			'analytics',
			'tracking',
			'telemetry',
			'beacon',
			'metrics',
			# Ad-related
			'doubleclick',
			'adsystem',
			'adserver',
			'advertising',
			# Social media widgets
			'facebook.com/plugins',
			'platform.twitter',
			'linkedin.com/embed',
			# Live chat and support
			'livechat',
			'zendesk',
			'intercom',
			'crisp.chat',
			'hotjar',
			# Push notifications
			'push-notifications',
			'onesignal',
			'pushwoosh',
			# Background sync/heartbeat
			'heartbeat',
			'ping',
			'alive',
			# WebRTC and streaming
			'webrtc',
			'rtmp://',
			'wss://',
			# Common CDNs for dynamic content
			'cloudfront.net',
			'fastly.net',
		}

		async def on_request(request):
			# Filter by resource type
			if request.resource_type not in RELEVANT_RESOURCE_TYPES:
				return

			# Filter out streaming, websocket, and other real-time requests
			if request.resource_type in {
				'websocket',
				'media',
				'eventsource',
				'manifest',
				'other',
			}:
				return

			# Filter out by URL patterns
			url = request.url.lower()
			if any(pattern in url for pattern in IGNORED_URL_PATTERNS):
				return

			# Filter out data URLs and blob URLs
			if url.startswith(('data:', 'blob:')):
				return

			# Filter out requests with certain headers
			headers = request.headers
			if headers.get('purpose') == 'prefetch' or headers.get('sec-fetch-dest') in [
				'video',
				'audio',
			]:
				return

			nonlocal last_activity
			pending_requests.add(request)
			last_activity = asyncio.get_event_loop().time()
			# self.logger.debug(f'Request started: {request.url} ({request.resource_type})')

		async def on_response(response):
			request = response.request
			if request not in pending_requests:
				return

			# Filter by content type if available
			content_type = response.headers.get('content-type', '').lower()

			# Skip if content type indicates streaming or real-time data
			if any(
				t in content_type
				for t in [
					'streaming',
					'video',
					'audio',
					'webm',
					'mp4',
					'event-stream',
					'websocket',
					'protobuf',
				]
			):
				pending_requests.remove(request)
				return

			# Only process relevant content types
			if not any(ct in content_type for ct in RELEVANT_CONTENT_TYPES):
				pending_requests.remove(request)
				return

			# Skip if response is too large (likely not essential for page load)
			content_length = response.headers.get('content-length')
			if content_length and int(content_length) > 5 * 1024 * 1024:  # 5MB
				pending_requests.remove(request)
				return

			nonlocal last_activity
			pending_requests.remove(request)
			last_activity = asyncio.get_event_loop().time()
			# self.logger.debug(f'Request resolved: {request.url} ({content_type})')

		# Attach event listeners
		page.on('request', on_request)
		page.on('response', on_response)

		now = asyncio.get_event_loop().time()
		try:
			# Wait for idle time
			start_time = asyncio.get_event_loop().time()
			while True:
				await asyncio.sleep(0.1)
				now = asyncio.get_event_loop().time()
				if (
					len(pending_requests) == 0
					and (now - last_activity) >= self.browser_profile.wait_for_network_idle_page_load_time
				):
					# Clear loading status when page loads successfully
					self._current_page_loading_status = None
					break
				if now - start_time > self.browser_profile.maximum_wait_page_load_time:
					self.logger.debug(
						f'{self} Network timeout after {self.browser_profile.maximum_wait_page_load_time}s with {len(pending_requests)} '
						f'pending requests: {[r.url for r in pending_requests]}'
					)
					# Set loading status for LLM to see
					self._current_page_loading_status = f'Page loading was aborted after {self.browser_profile.maximum_wait_page_load_time}s with {len(pending_requests)} pending network requests. You may want to use the wait action to allow more time for the page to fully load.'
					break

		finally:
			# Clean up event listeners
			page.remove_listener('request', on_request)
			page.remove_listener('response', on_response)

		elapsed = now - start_time
		if elapsed > 1:
			self.logger.debug(f'💤 Page network traffic calmed down after {now - start_time:.2f} seconds')

	@observe_debug(ignore_input=True, ignore_output=True, name='wait_for_page_and_frames_load')
	async def _wait_for_page_and_frames_load(self, timeout_overwrite: float | None = None):
		"""
		Ensures page is fully loaded and stable before continuing.
		Waits for network idle, DOM stability, and minimum WAIT_TIME.
		Also checks if the loaded URL is allowed.

		Parameters:
		-----------
		timeout_overwrite: float | None
			Override the minimum wait time
		"""
		# Start timing
		start_time = time.time()

		# Wait for page load
		page = await self.get_current_page()

		# Skip network waiting for new tab pages (about:blank, chrome://new-tab-page, etc.)
		# These pages load instantly and don't need network idle time
		if is_new_tab_page(page.url):
			self.logger.debug(f'⚡ Skipping page load wait for new tab page: {page.url}')
			return

		try:
			await self._wait_for_stable_network()

			# Check if the loaded URL is allowed
			await self._check_and_handle_navigation(page)
		except URLNotAllowedError as e:
			raise e
		except Exception as e:
			self.logger.warning(
				f'⚠️ Page load for {_log_pretty_url(page.url)} failed due to {type(e).__name__}, continuing anyway...'
			)

		# Calculate remaining time to meet minimum WAIT_TIME
		elapsed = time.time() - start_time
		remaining = max((timeout_overwrite or self.browser_profile.minimum_wait_page_load_time) - elapsed, 0)

		# Skip expensive performance API logging - can cause significant delays on complex pages
		bytes_used = None

		try:
			tab_idx = self.tabs.index(page)
		except ValueError:
			tab_idx = '??'

		extra_delay = ''
		if remaining > 0:
			extra_delay = f', waiting +{remaining:.2f}s for all frames to finish'

		if bytes_used is not None:
			self.logger.info(
				f'➡️ Page navigation [{tab_idx}]{_log_pretty_url(page.url, 40)} used {bytes_used / 1024:.1f} KB in {elapsed:.2f}s{extra_delay}'
			)
		else:
			self.logger.info(f'➡️ Page navigation [{tab_idx}]{_log_pretty_url(page.url, 40)} took {elapsed:.2f}s{extra_delay}')

		# Sleep remaining time if needed
		if remaining > 0:
			await asyncio.sleep(remaining)

	def _is_url_allowed(self, url: str) -> bool:
		"""
		Check if a URL is allowed based on the whitelist configuration. SECURITY CRITICAL.

		Supports optional glob patterns and schemes in allowed_domains:
		- *.example.com will match sub.example.com and example.com
		- *google.com will match google.com, agoogle.com, and www.google.com
		- http*://example.com will match http://example.com, https://example.com
		- chrome-extension://* will match chrome-extension://aaaaaaaaaaaa and chrome-extension://bbbbbbbbbbbbb
		"""

		if not self.browser_profile.allowed_domains:
			return True  # allowed_domains are not configured, allow everything by default

		# Special case: Always allow new tab pages
		if is_new_tab_page(url):
			return True

		for allowed_domain in self.browser_profile.allowed_domains:
			try:
				if match_url_with_domain_pattern(url, allowed_domain, log_warnings=True):
					# If it's a pattern with wildcards, show a warning
					if '*' in allowed_domain:
						parsed_url = urlparse(url)
						domain = parsed_url.hostname.lower() if parsed_url.hostname else ''
						_log_glob_warning(domain, allowed_domain, self.logger)
					return True
			except AssertionError:
				# This would only happen if a new tab page is passed to match_url_with_domain_pattern,
				# which shouldn't occur since we check for it above
				continue

		return False

	async def _check_and_handle_navigation(self, page: Page) -> None:
		"""Check if current page URL is allowed and handle if not."""
		if not self._is_url_allowed(page.url):
			self.logger.warning(f'⛔️ Navigation to non-allowed URL detected: {page.url}')
			try:
				await self.go_back()
			except Exception as e:
				self.logger.error(f'⛔️ Failed to go back after detecting non-allowed URL: {type(e).__name__}: {e}')
			raise URLNotAllowedError(f'Navigation to non-allowed URL: {page.url}')

	@observe_debug()
	async def refresh_page(self):
		"""Refresh the agent's current page"""

		page = await self.get_current_page()
		await page.reload()
		try:
			await page.wait_for_load_state()
		except Exception as e:
			self.logger.warning(f'⚠️ Page {_log_pretty_url(page.url)} failed to fully load after refresh: {type(e).__name__}: {e}')
			assert await page.evaluate('1'), (
				f'Page {page.url} crashed after {type(e).__name__} and can no longer be used via CDP: {e}'
			)

	async def go_back(self):
		"""Navigate the agent's tab back in browser history"""
		try:
			# 10 ms timeout
			page = await self.get_current_page()
			await page.go_back(timeout=10_000, wait_until='load')

			# await self._wait_for_page_and_frames_load(timeout_overwrite=1.0)
		except Exception as e:
			# Continue even if its not fully loaded, because we wait later for the page to load
			self.logger.debug(f'⏮️ Error during go_back: {type(e).__name__}: {e}')
			# Verify page is still usable after navigation error
			if 'timeout' in str(e).lower():
				try:
					assert await page.evaluate('1'), (
						f'Page {page.url} crashed after go_back {type(e).__name__} and can no longer be used via CDP: {e}'
					)
				except Exception as eval_error:
					self.logger.error(f'❌ Page crashed after go_back timeout: {eval_error}')

	async def go_forward(self):
		"""Navigate the agent's tab forward in browser history"""
		try:
			page = await self.get_current_page()
			await page.go_forward(timeout=10_000, wait_until='load')
		except Exception as e:
			# Continue even if its not fully loaded, because we wait later for the page to load
			self.logger.debug(f'⏭️ Error during go_forward: {type(e).__name__}: {e}')
			# Verify page is still usable after navigation error
			if 'timeout' in str(e).lower():
				try:
					assert await page.evaluate('1'), (
						f'Page {page.url} crashed after go_forward {type(e).__name__} and can no longer be used via CDP: {e}'
					)
				except Exception as eval_error:
					self.logger.error(f'❌ Page crashed after go_forward timeout: {eval_error}')

	async def close_current_tab(self):
		"""Close the current tab that the agent is working with.

		This closes the tab that the agent is currently using (agent_current_page),
		not necessarily the tab that is visible to the user (human_current_page).
		If they are the same tab, both references will be updated.
		"""
		assert self.browser_context is not None, 'Browser context is not set'
		assert self.agent_current_page is not None, 'Agent current page is not set'

		# Check if this is the foreground tab as well
		is_foreground = self.agent_current_page == self.human_current_page

		# Close the tab
		try:
			await self.agent_current_page.close()
		except Exception as e:
			self.logger.debug(f'⛔️ Error during close_current_tab: {type(e).__name__}: {e}')

		# Clear agent's reference to the closed tab
		self.agent_current_page = None

		# Clear foreground reference if needed
		if is_foreground:
			self.human_current_page = None

		# Switch to the first available tab if any exist
		if self.browser_context.pages:
			await self.switch_to_tab(0)
			# switch_to_tab already updates both tab references

		# Otherwise, the browser will be closed

	async def get_page_html(self) -> str:
		"""Get the HTML content of the agent's current page"""
		page = await self.get_current_page()
		return await page.content()

	async def get_page_structure(self) -> str:
		"""Get a debug view of the page structure including iframes"""
		debug_script = """(() => {
			function getPageStructure(element = document, depth = 0, maxDepth = 10) {
				if (depth >= maxDepth) return '';

				const indent = '  '.repeat(depth);
				let structure = '';

				// Skip certain elements that clutter the output
				const skipTags = new Set(['script', 'style', 'link', 'meta', 'noscript']);

				// Add current element info if it's not the document
				if (element !== document) {
					const tagName = element.tagName.toLowerCase();

					// Skip uninteresting elements
					if (skipTags.has(tagName)) return '';

					const id = element.id ? `#${element.id}` : '';
					const classes = element.className && typeof element.className === 'string' ?
						`.${element.className.split(' ').filter(c => c).join('.')}` : '';

					// Get additional useful attributes
					const attrs = [];
					if (element.getAttribute('role')) attrs.push(`role="${element.getAttribute('role')}"`);
					if (element.getAttribute('aria-label')) attrs.push(`aria-label="${element.getAttribute('aria-label')}"`);
					if (element.getAttribute('type')) attrs.push(`type="${element.getAttribute('type')}"`);
					if (element.getAttribute('name')) attrs.push(`name="${element.getAttribute('name')}"`);
					if (element.getAttribute('src')) {
						const src = element.getAttribute('src');
						attrs.push(`src="${src.substring(0, 50)}${src.length > 50 ? '...' : ''}"`);
					}

					// Add element info
					structure += `${indent}${tagName}${id}${classes}${attrs.length ? ' [' + attrs.join(', ') + ']' : ''}\\n`;

					// Handle iframes specially
					if (tagName === 'iframe') {
						try {
							const iframeDoc = element.contentDocument || element.contentWindow?.document;
							if (iframeDoc) {
								structure += `${indent}  [IFRAME CONTENT]:\\n`;
								structure += getPageStructure(iframeDoc, depth + 2, maxDepth);
							} else {
								structure += `${indent}  [IFRAME: No access - likely cross-origin]\\n`;
							}
						} catch (e) {
							structure += `${indent}  [IFRAME: Access denied - ${e.message}]\\n`;
						}
					}
				}

				// Get all child elements
				const children = element.children || element.childNodes;
				for (const child of children) {
					if (child.nodeType === 1) { // Element nodes only
						structure += getPageStructure(child, depth + 1, maxDepth);
					}
				}

				return structure;
			}

			return getPageStructure();
		})()"""

		page = await self.get_current_page()
		structure = await page.evaluate(debug_script)
		return structure

	@observe_debug(ignore_input=True, ignore_output=True)
	@time_execution_async('--get_state_summary')
	@require_healthy_browser(usable_page=True, reopen_page=True)
	async def get_state_summary(
		self, cache_clickable_elements_hashes: bool, include_screenshot: bool = True
	) -> BrowserStateSummary:
		self.logger.debug('🔄 Starting get_state_summary...')
		"""Get a summary of the current browser state

		This method builds a BrowserStateSummary object that captures the current state
		of the browser, including url, title, tabs, screenshot, and DOM tree.

		Parameters:
		-----------
		cache_clickable_elements_hashes: bool
			If True, cache the clickable elements hashes for the current state.
			This is used to calculate which elements are new to the LLM since the last message,
			which helps reduce token usage.
		include_screenshot: bool
			If True, include screenshot in the state summary. Set to False to improve performance
			when screenshots are not needed (e.g., in multi_act element validation).
		"""

		updated_state = await self._get_updated_state(include_screenshot=include_screenshot)

		# Find out which elements are new
		# Do this only if url has not changed
		if cache_clickable_elements_hashes:
			# Lazy import heavy DOM service
			from browser_use.dom.clickable_element_processor.service import ClickableElementProcessor

			# if we are on the same url as the last state, we can use the cached hashes
			if self._cached_clickable_element_hashes and self._cached_clickable_element_hashes.url == updated_state.url:
				# Pointers, feel free to edit in place
				updated_state_clickable_elements = ClickableElementProcessor.get_clickable_elements(updated_state.element_tree)

				for dom_element in updated_state_clickable_elements:
					dom_element.is_new = (
						ClickableElementProcessor.hash_dom_element(dom_element)
						not in self._cached_clickable_element_hashes.hashes  # see which elements are new from the last state where we cached the hashes
					)
			# in any case, we need to cache the new hashes
			self._cached_clickable_element_hashes = CachedClickableElementHashes(
				url=updated_state.url,
				hashes=ClickableElementProcessor.get_clickable_elements_hashes(updated_state.element_tree),
			)

		assert updated_state
		self._cached_browser_state_summary = updated_state

		return self._cached_browser_state_summary

	@observe_debug(ignore_input=True, ignore_output=True, name='get_minimal_state_summary')
	@require_healthy_browser(usable_page=True, reopen_page=True)
	@time_execution_async('--get_minimal_state_summary')
	async def get_minimal_state_summary(self) -> BrowserStateSummary:
		"""Get basic page info without DOM processing, but try to capture screenshot"""
		from browser_use.browser.views import BrowserStateSummary
		from browser_use.dom.views import DOMElementNode

		page = await self.get_current_page()

		# Get basic info - no DOM parsing to avoid errors
		url = getattr(page, 'url', 'unknown')

		# Try to get title safely
		try:
			# timeout after 2 seconds
			title = await asyncio.wait_for(page.title(), timeout=2.0)
		except Exception:
			title = 'Page Load Error'

		# Try to get tabs info safely
		try:
			# timeout after 2 seconds
			tabs_info = await retry(timeout=2, retries=0)(self.get_tabs_info)()
		except Exception:
			tabs_info = []

		# Create minimal DOM element for error state
		minimal_element_tree = DOMElementNode(
			tag_name='body',
			xpath='/body',
			attributes={},
			children=[],
			is_visible=True,
			parent=None,
		)

		# Check if current page is a PDF viewer
		is_pdf_viewer = await self._is_pdf_viewer(page)

		return BrowserStateSummary(
			element_tree=minimal_element_tree,  # Minimal DOM tree
			selector_map={},  # Empty selector map
			url=url,
			title=title,
			tabs=tabs_info,
			pixels_above=0,
			pixels_below=0,
			browser_errors=[f'Page state retrieval failed, minimal recovery applied for {url}'],
			is_pdf_viewer=is_pdf_viewer,
			loading_status=self._current_page_loading_status,
		)

	@observe_debug(ignore_input=True, ignore_output=True, name='get_updated_state')
	async def _get_updated_state(self, focus_element: int = -1, include_screenshot: bool = True) -> BrowserStateSummary:
		"""Update and return state."""

		# Check if current page is still valid, if not switch to another available page
		page = await self.get_current_page()

		# Check if this is a new tab or chrome:// page early for optimization
		is_empty_page = is_new_tab_page(page.url) or page.url.startswith('chrome://')

		try:
			# Fast path for empty pages - skip all expensive operations
			if is_empty_page:
				self.logger.debug(f'⚡ Fast path for empty page: {page.url}')

				# Create minimal DOM state immediately
				from browser_use.dom.views import DOMElementNode, DOMState

				minimal_element_tree = DOMElementNode(
					tag_name='body',
					xpath='',
					attributes={},
					children=[],
					is_visible=False,
					parent=None,
				)
				content = DOMState(element_tree=minimal_element_tree, selector_map={})

				# Get minimal tab info
				tabs_info = await self.get_tabs_info()

				# Skip screenshot for empty pages
				screenshot_b64 = None

				# Use default viewport dimensions from browser profile
				viewport = self.browser_profile.viewport or {'width': 1280, 'height': 720}
				page_info = PageInfo(
					viewport_width=viewport['width'],
					viewport_height=viewport['height'],
					page_width=viewport['width'],
					page_height=viewport['height'],
					scroll_x=0,
					scroll_y=0,
					pixels_above=0,
					pixels_below=0,
					pixels_left=0,
					pixels_right=0,
				)

				# Return minimal state immediately
				self.browser_state_summary = BrowserStateSummary(
					element_tree=content.element_tree,
					selector_map=content.selector_map,
					url=page.url,
					title='New Tab' if is_new_tab_page(page.url) else 'Chrome Page',
					tabs=tabs_info,
					screenshot=screenshot_b64,
					page_info=page_info,
					pixels_above=0,
					pixels_below=0,
					browser_errors=[],
					is_pdf_viewer=False,
					loading_status=self._current_page_loading_status,
				)
				return self.browser_state_summary

			# Normal path for regular pages
			self.logger.debug('🧹 Removing highlights...')
			try:
				await self.remove_highlights()
			except TimeoutError:
				self.logger.debug('Timeout to remove highlights')

			# Check for PDF and auto-download if needed
			try:
				pdf_path = await self._auto_download_pdf_if_needed(page)
				if pdf_path:
					self.logger.info(f'📄 PDF auto-downloaded: {pdf_path}')
			except Exception as e:
				self.logger.debug(f'PDF auto-download check failed: {type(e).__name__}: {e}')

			self.logger.debug('🌳 Starting DOM processing...')
			from browser_use.dom.service import DomService

			dom_service = DomService(page, logger=self.logger)
			try:
				content = await asyncio.wait_for(
					dom_service.get_clickable_elements(
						focus_element=focus_element,
						viewport_expansion=self.browser_profile.viewport_expansion,
						highlight_elements=self.browser_profile.highlight_elements,
					),
					timeout=45.0,  # 45 second timeout for DOM processing - generous for complex pages
				)
				self.logger.debug('✅ DOM processing completed')
			except TimeoutError:
				self.logger.warning(f'DOM processing timed out after 45 seconds for {page.url}')
				self.logger.warning('🔄 Falling back to minimal DOM state to allow basic navigation...')

				# Create minimal DOM state for basic navigation
				from browser_use.dom.views import DOMElementNode

				minimal_element_tree = DOMElementNode(
					tag_name='body',
					xpath='/body',
					attributes={},
					children=[],
					is_visible=True,
					parent=None,
				)

				from browser_use.dom.views import DOMState

				content = DOMState(element_tree=minimal_element_tree, selector_map={})

			self.logger.debug('📋 Getting tabs info...')
			tabs_info = await self.get_tabs_info()
			self.logger.debug('✅ Tabs info completed')

			# Get all cross-origin iframes within the page and open them in new tabs
			# mark the titles of the new tabs so the LLM knows to check them for additional content
			# unfortunately too buggy for now, too many sites use invisible cross-origin iframes for ads, tracking, youtube videos, social media, etc.
			# and it distracts the bot by opening a lot of new tabs
			# iframe_urls = await dom_service.get_cross_origin_iframes()
			# outer_page = self.agent_current_page
			# for url in iframe_urls:
			# 	if url in [tab.url for tab in tabs_info]:
			# 		continue  # skip if the iframe if we already have it open in a tab
			# 	new_page_id = tabs_info[-1].page_id + 1
			# 	self.logger.debug(f'Opening cross-origin iframe in new tab #{new_page_id}: {url}')
			# 	await self.create_new_tab(url)
			# 	tabs_info.append(
			# 		TabInfo(
			# 			page_id=new_page_id,
			# 			url=url,
			# 			title=f'iFrame opened as new tab, treat as if embedded inside page {outer_page.url}: {page.url}',
			# 			parent_page_url=outer_page.url,
			# 		)
			# 	)

			if include_screenshot:
				try:
					self.logger.debug('📸 Capturing screenshot...')
					# Reasonable timeout for screenshot
					screenshot_b64 = await self.take_screenshot()
					# self.logger.debug('✅ Screenshot completed')
				except Exception as e:
					self.logger.warning(f'❌ Screenshot failed for {_log_pretty_url(page.url)}: {type(e).__name__} {e}')
					screenshot_b64 = None
			else:
				screenshot_b64 = None

			# Get comprehensive page information
			page_info = await self.get_page_info(page)
			try:
				self.logger.debug('📏 Getting scroll info...')
				pixels_above, pixels_below = await asyncio.wait_for(self.get_scroll_info(page), timeout=5.0)
				self.logger.debug('✅ Scroll info completed')
			except Exception as e:
				self.logger.warning(f'Failed to get scroll info: {type(e).__name__}')
				pixels_above, pixels_below = 0, 0

			try:
				title = await asyncio.wait_for(page.title(), timeout=3.0)
			except Exception:
				title = 'Title unavailable'

			# Check if this is a minimal fallback state
			browser_errors = []
			if not content.selector_map:  # Empty selector map indicates fallback state
				browser_errors.append(
					f'DOM processing timed out for {page.url} - using minimal state. Basic navigation still available via go_to_url, scroll, and search actions.'
				)

			# Check if current page is a PDF viewer
			is_pdf_viewer = await self._is_pdf_viewer(page)

			self.browser_state_summary = BrowserStateSummary(
				element_tree=content.element_tree,
				selector_map=content.selector_map,
				url=page.url,
				title=title,
				tabs=tabs_info,
				screenshot=screenshot_b64,
				page_info=page_info,
				pixels_above=pixels_above,
				pixels_below=pixels_below,
				browser_errors=browser_errors,
				is_pdf_viewer=is_pdf_viewer,
				loading_status=self._current_page_loading_status,
			)

			self.logger.debug('✅ get_state_summary completed successfully')
			return self.browser_state_summary
		except Exception as e:
			self.logger.error(f'❌ Failed to update browser_state_summary: {type(e).__name__}: {e}')
			# Return last known good state if available
			if hasattr(self, 'browser_state_summary'):
				return self.browser_state_summary
			raise

	# region - Page Health Check Helpers
	@observe_debug(ignore_input=True)
	async def _is_page_responsive(self, page: Page, timeout: float = 5.0) -> bool:
		"""Check if a page is responsive by trying to evaluate simple JavaScript."""
		eval_task = None
		try:
			eval_task = asyncio.create_task(page.evaluate('1'))
			done, pending = await asyncio.wait([eval_task], timeout=timeout)

			if eval_task in done:
				try:
					await eval_task  # This will raise if the evaluation failed
					return True
				except Exception:
					return False
			else:
				# Timeout - the page is unresponsive
				return False
		except Exception:
			return False
		finally:
			# Always clean up the eval task
			if eval_task and not eval_task.done():
				eval_task.cancel()
				try:
					await eval_task
				except (asyncio.CancelledError, Exception):
					pass

	async def _force_close_page_via_cdp(self, page_url: str) -> bool:
		"""Force close a crashed page using CDP from a clean temporary page."""
		try:
			# self.logger.info('🔨 Creating temporary page for CDP force-close...')

			# Create a clean page for CDP operations
			assert self.browser_context, 'Browser context is not set up yet'
			temp_page = await asyncio.wait_for(self.browser_context.new_page(), timeout=5.0)
			await asyncio.wait_for(temp_page.goto('about:blank'), timeout=2.0)

			# Create CDP session from the clean page
			cdp_session = await asyncio.wait_for(self.browser_context.new_cdp_session(temp_page), timeout=5.0)  # type: ignore

			try:
				# Get all browser targets
				targets = await asyncio.wait_for(cdp_session.send('Target.getTargets'), timeout=2.0)

				# Find the crashed page target
				blocked_target_id = None
				for target in targets.get('targetInfos', []):
					if target.get('type') == 'page' and target.get('url') == page_url:
						blocked_target_id = target.get('targetId')
						# self.logger.debug(f'Found target to close: {page_url}')
						break

				if blocked_target_id:
					# Force close the target
					self.logger.warning(
						f'🪓 Force-closing crashed page target_id={blocked_target_id} via CDP: {_log_pretty_url(page_url)}...'
					)
					await asyncio.wait_for(cdp_session.send('Target.closeTarget', {'targetId': blocked_target_id}), timeout=2.0)
					# self.logger.debug(f'☠️ Successfully force-closed crashed page target_id={blocked_target_id} via CDP: {_log_pretty_url(page_url)}')
					return True
				else:
					self.logger.debug(
						f'❌ Could not find CDP page target_id to force-close: {_log_pretty_url(page_url)} (concurrency issues?)'
					)
					return False

			finally:
				# Clean up
				try:
					await asyncio.wait_for(cdp_session.detach(), timeout=1.0)
				except Exception:
					pass
				await temp_page.close()

		except Exception as e:
			self.logger.error(f'❌ Using raw CDP to force-close crashed page failed: {type(e).__name__}: {e}')
			return False

	async def _try_reopen_url(self, url: str, timeout_ms: int | None = None) -> bool:
		"""Try to reopen a URL in a new page and check if it's responsive."""
		if not url or is_new_tab_page(url):
			return False

		timeout_ms = int(timeout_ms or self.browser_profile.default_navigation_timeout or 6000)

		try:
			self.logger.debug(f'🔄 Attempting to reload URL that crashed: {_log_pretty_url(url)}')

			# Create new page directly to avoid circular dependency
			assert self.browser_context is not None, 'Browser context is not set'
			new_page = await self.browser_context.new_page()
			self.agent_current_page = new_page

			# Update human tab reference if there is no human tab yet
			if (not self.human_current_page) or self.human_current_page.is_closed():
				self.human_current_page = new_page

			# Set viewport for new tab
			if self.browser_profile.viewport:
				await new_page.set_viewport_size(self.browser_profile.viewport)

			# Navigate with timeout using asyncio.wait
			nav_task = asyncio.create_task(new_page.goto(url, wait_until='load', timeout=timeout_ms))
			done, pending = await asyncio.wait([nav_task], timeout=(timeout_ms + 500) / 1000)

			if nav_task in pending:
				# Navigation timed out
				self.logger.debug(
					f'⚠️ Attempting to reload previously crashed URL {_log_pretty_url(url)} failed again, timed out again after {timeout_ms / 1000}s'
				)
				nav_task.cancel()
				try:
					await nav_task
				except asyncio.CancelledError:
					pass
			elif nav_task in done:
				try:
					await nav_task  # This will raise if navigation failed
				except Exception as e:
					self.logger.debug(
						f'⚠️ Attempting to reload previously crashed URL {_log_pretty_url(url)} failed again: {type(e).__name__}'
					)

			# Wait a bit for any transient blocking to resolve
			await asyncio.sleep(1.0)

			# Check if the reopened page is responsive
			# self.logger.debug('Checking if reopened page is responsive...')
			is_responsive = await self._is_page_responsive(new_page, timeout=2.0)

			if is_responsive:
				self.logger.info(f'✅ Page recovered and is now responsive after reopening on: {_log_pretty_url(url)}')
				return True
			else:
				self.logger.warning(f'⚠️ Reopened page {_log_pretty_url(url)} is still unresponsive')
				# Close the unresponsive page before returning
				# This is critical to prevent the recovery flow from hanging
				try:
					await self._force_close_page_via_cdp(new_page.url)
				except Exception as e:
					self.logger.error(
						f'❌ Failed to close crashed page {_log_pretty_url(url)} via CDP: {type(e).__name__}: {e} (something is very wrong or system is extremely overloaded)'
					)
				self.agent_current_page = None  # Clear reference to closed page
				return False

		except Exception as e:
			self.logger.error(f'❌ Retrying crashed page {_log_pretty_url(url)} failed: {type(e).__name__}: {e}')
			return False

	async def _create_blank_fallback_page(self, url: str) -> None:
		"""Create a new blank page as a fallback when recovery fails."""
		self.logger.warning(
			f'⚠️ Resetting to about:blank as fallback because browser is unable to load the original URL without crashing: {_log_pretty_url(url)}'
		)
		# self.logger.debug(f'Current agent_current_page: {self.agent_current_page}')

		# Close any existing broken page
		if self.agent_current_page and not self.agent_current_page.is_closed():
			try:
				await self.agent_current_page.close()
			except Exception:
				pass

		# Create fresh page directly (avoid decorated methods to prevent circular dependency)
		assert self.browser_context is not None, 'Browser context is not set'
		new_page = await self.browser_context.new_page()
		self.agent_current_page = new_page

		# Update human tab reference if there is no human tab yet
		if (not self.human_current_page) or self.human_current_page.is_closed():
			self.human_current_page = new_page

		# Set viewport for new tab
		if self.browser_profile.viewport:
			await new_page.set_viewport_size(self.browser_profile.viewport)

		# Navigate to blank
		try:
			await new_page.goto('about:blank', wait_until='load', timeout=5000)
		except Exception as e:
			self.logger.error(
				f'❌ Failed to navigate to about:blank: {type(e).__name__}: {e} (something is very wrong or system is extremely overloaded)'
			)
			raise

		# Verify it's responsive
		if not await self._is_page_responsive(new_page, timeout=1.0):
			raise BrowserError(
				'Browser is unable to load any new about:blank pages (something is very wrong or browser is extremely overloaded)'
			)

	@observe_debug(ignore_input=True, name='recover_unresponsive_page')
	async def _recover_unresponsive_page(self, calling_method: str, timeout_ms: int | None = None) -> None:
		"""Recover from an unresponsive page by closing and reopening it."""
		self.logger.warning(f'⚠️ Page JS engine became unresponsive in {calling_method}(), attempting recovery...')
		timeout_ms = min(3000, int(timeout_ms or self.browser_profile.default_navigation_timeout or 5000))

		# Check if browser process is still alive before attempting recovery
		if self.browser_pid:
			try:
				import psutil

				proc = psutil.Process(self.browser_pid)
				if proc.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
					self.logger.error(f'❌ Browser process {self.browser_pid} has crashed and cannot be recovered')
					raise RuntimeError('Browser process has crashed - cannot recover unresponsive page')
			except psutil.NoSuchProcess:
				self.logger.error(f'❌ Browser process {self.browser_pid} no longer exists')
				raise RuntimeError('Browser process has crashed - cannot recover unresponsive page')

		# Check if browser connection is still alive
		if self.browser and not self.browser.is_connected():
			self.logger.error('❌ Browser connection lost - browser process may have crashed')
			raise RuntimeError('Browser connection lost - cannot recover unresponsive page')

		# Prevent re-entrance
		self._in_recovery = True
		try:
			# Get current URL before recovery
			assert self.agent_current_page, 'Agent current page is not set'
			current_url = self.agent_current_page.url
			# self.logger.debug(f'Current URL: {current_url}')

			# Clear page references
			blocked_page = self.agent_current_page
			self.agent_current_page = None
			if blocked_page == self.human_current_page:
				self.human_current_page = None

			# Force-close the crashed page via CDP
			self.logger.debug('🪓 Page Recovery Step 1/3: Force-closing crashed page via CDP...')
			await self._force_close_page_via_cdp(current_url)

			# Remove the closed page from browser_context.pages by forcing a refresh
			# This prevents TargetClosedError when iterating through pages later
			if self.browser_context and self.browser_context.pages:
				# Additional cleanup: close any page objects that have the same url as the crashed page
				# (could close too many pages by accident if we have a few different tabs on the same URL)
				# Sometimes playwright doesn't immediately remove force-closed pages from the list
				for page in self.browser_context.pages[:]:  # Use slice to avoid modifying list during iteration
					if page.url == current_url and not page.is_closed() and not is_new_tab_page(page.url):
						try:
							# Try to close it via playwright as well
							await page.close()
							self.logger.debug(
								f'🪓 Closed 🅟 {str(id(page))[-2:]} because it has a known crash-causing URL: {_log_pretty_url(page.url)}'
							)
						except Exception:
							pass  # Page might already be closed via CDP

			# Try to reopen the URL (in case blocking was transient)
			self.logger.debug('🍼 Page Recovery Step 2/3: Trying to reopen the URL again...')
			if await self._try_reopen_url(current_url, timeout_ms=timeout_ms):
				self.logger.debug('✅ Page Recovery Step 3/3: Page loading succeeded after 2nd attempt!')
				return  # Success!

			# If that failed, fall back to blank page
			self.logger.debug(
				'❌ Page Recovery Step 3/3: Loading the page a 2nd time failed as well, browser seems unable to load this URL without getting stuck, retreating to a safe page...'
			)
			await self._create_blank_fallback_page(current_url)

		finally:
			# Always clear recovery flag
			self._in_recovery = False

	# region - Browser Actions
	@observe_debug(name='take_screenshot', ignore_output=True)
	@retry(
		retries=1,  # try up to 1 time to take the screenshot (2 total attempts)
		timeout=30,  # allow up to 30s for each attempt (includes recovery time)
		wait=1,  # wait 1s between each attempt
		# semaphore_limit=2,  # Allow 2 screenshots at a time to better utilize resources
		# semaphore_name='screenshot_global',
		# semaphore_scope='multiprocess',
		# semaphore_lax=True,  # Continue without semaphore if it can't be acquired
		# semaphore_timeout=15,  # Wait up to 15s for semaphore acquisition
	)
	@require_healthy_browser(usable_page=True, reopen_page=True)
	@time_execution_async('--take_screenshot')
	async def take_screenshot(self, full_page: bool = False) -> str | None:
		"""
		Returns a base64 encoded screenshot of the current page using CDP.

		The decorator order ensures:
		1. @retry runs first (outer decorator)
		2. @require_healthy_browser runs on each retry attempt
		3. Page responsiveness is checked before each screenshot attempt
		4. If page is unresponsive, it's recovered and the method is retried
		"""
		assert self.agent_current_page is not None, 'Agent current page is not set'
		assert self.browser_context, 'Browser context is not set'

		page = self.agent_current_page

		if is_new_tab_page(page.url):
			self.logger.warning(
				f'▫️ Sending LLM 4px placeholder instead of real screenshot of: {_log_pretty_url(page.url)} (page empty)'
			)
			# not an exception because there's no point in retrying if we hit this, its always pointless to screenshot about:blank
			# raise ValueError('Refusing to take unneeded screenshot of empty new tab page')
			# return a 4px*4px white png to avoid wasting tokens - instead of 1px*1px white png that was
			return PLACEHOLDER_4PX_SCREENSHOT

		# Always bring page to front before rendering, otherwise it crashes in some cases, not sure why
		try:
			await page.bring_to_front()
		except Exception:
			pass

		# Take screenshot using CDP to get around playwright's unnecessary slowness and weird behavior
		cdp_session = None
		try:
			# Create CDP session for the screenshot
			self.logger.debug(
				f'📸 Taking viewport-only PNG screenshot of page via fresh CDP session: {_log_pretty_url(page.url)}'
			)
			cdp_session = await self.browser_context.new_cdp_session(page)  # type: ignore

			# Capture screenshot via CDP
			screenshot_response = await cdp_session.send(
				'Page.captureScreenshot',
				{
					'captureBeyondViewport': False,
					'fromSurface': True,
					'format': 'png',
				},
			)

			screenshot_b64 = screenshot_response.get('data')
			if not screenshot_b64:
				raise Exception(
					f'CDP returned empty screenshot data for page {_log_pretty_url(page.url)}? (expected png base64)'
				)  # have never seen this happen in practice

			return screenshot_b64

		except Exception as err:
			error_str = f'{type(err).__name__}: {err}'
			if 'timeout' in error_str.lower():
				self.logger.warning(f'⏱️ Screenshot timed out on page {_log_pretty_url(page.url)} (possibly crashed): {error_str}')
			else:
				self.logger.error(f'❌ Screenshot failed on page {_log_pretty_url(page.url)} (possibly crashed): {error_str}')
			raise
		finally:
			if cdp_session:
				try:
					await asyncio.wait_for(cdp_session.detach(), timeout=1.0)
				except Exception:
					pass

	# region - User Actions

	@staticmethod
	async def _get_unique_filename(directory: str | Path, filename: str) -> str:
		"""Generate a unique filename for downloads by appending (1), (2), etc., if a file already exists."""
		base, ext = os.path.splitext(filename)
		counter = 1
		new_filename = filename
		while os.path.exists(os.path.join(directory, new_filename)):
			new_filename = f'{base} ({counter}){ext}'
			counter += 1
		return new_filename

	async def _start_context_tracing(self):
		"""Start tracing on browser context if trace_path is configured."""
		# TEMPORARILY DISABLED: Trace recording causing test timeouts
		# if self.browser_profile.traces_dir and self.browser_context:
		# 	try:
		# 		self.logger.debug(f'📽️ Starting tracing (will save to: {self.browser_profile.traces_dir})')
		# 		# Don't pass any path to start() - let Playwright handle internal temp files
		# 		await self.browser_context.tracing.start(
		# 			screenshots=True,
		# 			snapshots=True,
		# 			sources=False,  # Reduce trace size
		# 		)
		# 	except Exception as e:
		# 		self.logger.warning(f'Failed to start tracing: {e}')

	@staticmethod
	def _convert_simple_xpath_to_css_selector(xpath: str) -> str:
		"""Converts simple XPath expressions to CSS selectors."""
		if not xpath:
			return ''

		# Remove leading slash if present
		xpath = xpath.lstrip('/')

		# Split into parts
		parts = xpath.split('/')
		css_parts = []

		for part in parts:
			if not part:
				continue

			# Handle custom elements with colons by escaping them
			if ':' in part and '[' not in part:
				base_part = part.replace(':', r'\:')
				css_parts.append(base_part)
				continue

			# Handle index notation [n]
			if '[' in part:
				base_part = part[: part.find('[')]
				# Handle custom elements with colons in the base part
				if ':' in base_part:
					base_part = base_part.replace(':', r'\:')
				index_part = part[part.find('[') :]

				# Handle multiple indices
				indices = [i.strip('[]') for i in index_part.split(']')[:-1]]

				for idx in indices:
					try:
						# Handle numeric indices
						if idx.isdigit():
							index = int(idx) - 1
							base_part += f':nth-of-type({index + 1})'
						# Handle last() function
						elif idx == 'last()':
							base_part += ':last-of-type'
						# Handle position() functions
						elif 'position()' in idx:
							if '>1' in idx:
								base_part += ':nth-of-type(n+2)'
					except ValueError:
						continue

				css_parts.append(base_part)
			else:
				css_parts.append(part)

		base_selector = ' > '.join(css_parts)
		return base_selector

	@classmethod
	@time_execution_sync('--enhanced_css_selector_for_element')
	def _enhanced_css_selector_for_element(cls, element: DOMElementNode, include_dynamic_attributes: bool = True) -> str:
		"""
		Creates a CSS selector for a DOM element, handling various edge cases and special characters.

		Args:
						element: The DOM element to create a selector for

		Returns:
						A valid CSS selector string
		"""
		try:
			# Get base selector from XPath
			css_selector = cls._convert_simple_xpath_to_css_selector(element.xpath)

			# Handle class attributes
			if 'class' in element.attributes and element.attributes['class'] and include_dynamic_attributes:
				# Define a regex pattern for valid class names in CSS
				valid_class_name_pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_-]*$')

				# Iterate through the class attribute values
				classes = element.attributes['class'].split()
				for class_name in classes:
					# Skip empty class names
					if not class_name.strip():
						continue

					# Check if the class name is valid
					if valid_class_name_pattern.match(class_name):
						# Append the valid class name to the CSS selector
						css_selector += f'.{class_name}'
					else:
						# Skip invalid class names
						continue

			# Expanded set of safe attributes that are stable and useful for selection
			SAFE_ATTRIBUTES = {
				# Data attributes (if they're stable in your application)
				'id',
				# Standard HTML attributes
				'name',
				'type',
				'placeholder',
				# Accessibility attributes
				'aria-label',
				'aria-labelledby',
				'aria-describedby',
				'role',
				# Common form attributes
				'for',
				'autocomplete',
				'required',
				'readonly',
				# Media attributes
				'alt',
				'title',
				'src',
				# Custom stable attributes (add any application-specific ones)
				'href',
				'target',
			}

			if include_dynamic_attributes:
				dynamic_attributes = {
					'data-id',
					'data-qa',
					'data-cy',
					'data-testid',
				}
				SAFE_ATTRIBUTES.update(dynamic_attributes)

			# Handle other attributes
			for attribute, value in element.attributes.items():
				if attribute == 'class':
					continue

				# Skip invalid attribute names
				if not attribute.strip():
					continue

				if attribute not in SAFE_ATTRIBUTES:
					continue

				# Escape special characters in attribute names
				safe_attribute = attribute.replace(':', r'\:')

				# Handle different value cases
				if value == '':
					css_selector += f'[{safe_attribute}]'
				elif any(char in value for char in '"\'<>`\n\r\t'):
					# Use contains for values with special characters
					# For newline-containing text, only use the part before the newline
					if '\n' in value:
						value = value.split('\n')[0]
					# Regex-substitute *any* whitespace with a single space, then strip.
					collapsed_value = re.sub(r'\s+', ' ', value).strip()
					# Escape embedded double-quotes.
					safe_value = collapsed_value.replace('"', '\\"')
					css_selector += f'[{safe_attribute}*="{safe_value}"]'
				else:
					css_selector += f'[{safe_attribute}="{value}"]'

			return css_selector

		except Exception:
			# Fallback to a more basic selector if something goes wrong
			tag_name = element.tag_name or '*'
			return f"{tag_name}[highlight_index='{element.highlight_index}']"

	@require_healthy_browser(usable_page=True, reopen_page=True)
	@time_execution_async('--is_visible')
	async def _is_visible(self, element: ElementHandle) -> bool:
		"""
		Checks if an element is visible on the page.
		We use our own implementation instead of relying solely on Playwright's is_visible() because
		of edge cases with CSS frameworks like Tailwind. When elements use Tailwind's 'hidden' class,
		the computed style may return display as '' (empty string) instead of 'none', causing Playwright
		to incorrectly consider hidden elements as visible. By additionally checking the bounding box
		dimensions, we catch elements that have zero width/height regardless of how they were hidden.
		"""
		is_hidden = await element.is_hidden()
		bbox = await element.bounding_box()

		return not is_hidden and bbox is not None and bbox['width'] > 0 and bbox['height'] > 0

	@require_healthy_browser(usable_page=True, reopen_page=True)
	@time_execution_async('--get_locate_element')
	@observe_debug(ignore_input=True, name='get_locate_element')
	async def get_locate_element(self, element: DOMElementNode) -> ElementHandle | None:
		page = await self.get_current_page()
		current_frame = page

		# Start with the target element and collect all parents
		parents: list[DOMElementNode] = []
		current = element
		while current.parent is not None:
			parent = current.parent
			parents.append(parent)
			current = parent

		# Reverse the parents list to process from top to bottom
		parents.reverse()

		# Process all iframe parents in sequence
		iframes = [item for item in parents if item.tag_name == 'iframe']
		for parent in iframes:
			css_selector = self._enhanced_css_selector_for_element(
				parent,
				include_dynamic_attributes=self.browser_profile.include_dynamic_attributes,
			)
			# Use CSS selector if available, otherwise fall back to XPath
			if css_selector:
				current_frame = current_frame.frame_locator(css_selector)
			else:
				self.logger.debug(f'Using XPath for iframe: {parent.xpath}')
				current_frame = current_frame.frame_locator(f'xpath={parent.xpath}')

		css_selector = self._enhanced_css_selector_for_element(
			element, include_dynamic_attributes=self.browser_profile.include_dynamic_attributes
		)

		try:
			if isinstance(current_frame, FrameLocator):
				if css_selector:
					element_handle = await current_frame.locator(css_selector).element_handle()
				else:
					# Fall back to XPath when CSS selector is empty
					self.logger.debug(f'CSS selector empty, falling back to XPath: {element.xpath}')
					element_handle = await current_frame.locator(f'xpath={element.xpath}').element_handle()
				return element_handle
			else:
				# Try CSS selector first if available
				if css_selector:
					element_handle = await current_frame.query_selector(css_selector)
				else:
					# Fall back to XPath
					self.logger.debug(f'CSS selector empty, falling back to XPath: {element.xpath}')
					element_handle = await current_frame.locator(f'xpath={element.xpath}').element_handle()
				if element_handle:
					is_visible = await self._is_visible(element_handle)
					if is_visible:
						await element_handle.scroll_into_view_if_needed(timeout=1_000)
					return element_handle
				return None
		except Exception as e:
			# If CSS selector failed, try XPath as fallback
			if css_selector and 'CSS.escape' not in str(e):
				try:
					self.logger.debug(f'CSS selector failed, trying XPath fallback: {element.xpath}')
					if isinstance(current_frame, FrameLocator):
						element_handle = await current_frame.locator(f'xpath={element.xpath}').element_handle()
					else:
						element_handle = await current_frame.locator(f'xpath={element.xpath}').element_handle()

					if element_handle:
						is_visible = await self._is_visible(element_handle)
						if is_visible:
							await element_handle.scroll_into_view_if_needed(timeout=1_000)
						return element_handle
				except Exception as xpath_e:
					self.logger.error(
						f'❌ Failed to locate element with both CSS ({css_selector}) and XPath ({element.xpath}): {type(xpath_e).__name__}: {xpath_e}'
					)
					return None
			else:
				self.logger.error(
					f'❌ Failed to locate element {css_selector or element.xpath} on page {_log_pretty_url(page.url)}: {type(e).__name__}: {e}'
				)
				return None

	@require_healthy_browser(usable_page=True, reopen_page=True)
	@time_execution_async('--get_locate_element_by_xpath')
	async def get_locate_element_by_xpath(self, xpath: str) -> ElementHandle | None:
		"""
		Locates an element on the page using the provided XPath.
		"""
		page = await self.get_current_page()

		try:
			# Use XPath to locate the element
			element_handle = await page.query_selector(f'xpath={xpath}')
			if element_handle:
				is_visible = await self._is_visible(element_handle)
				if is_visible:
					await element_handle.scroll_into_view_if_needed(timeout=1_000)
				return element_handle
			return None
		except Exception as e:
			self.logger.error(f'❌ Failed to locate xpath {xpath} on page {_log_pretty_url(page.url)}: {type(e).__name__}: {e}')
			return None

	@require_healthy_browser(usable_page=True, reopen_page=True)
	@time_execution_async('--get_locate_element_by_css_selector')
	async def get_locate_element_by_css_selector(self, css_selector: str) -> ElementHandle | None:
		"""
		Locates an element on the page using the provided CSS selector.
		"""
		page = await self.get_current_page()

		try:
			# Use CSS selector to locate the element
			element_handle = await page.query_selector(css_selector)
			if element_handle:
				is_visible = await self._is_visible(element_handle)
				if is_visible:
					await element_handle.scroll_into_view_if_needed(timeout=1_000)
				return element_handle
			return None
		except Exception as e:
			self.logger.error(
				f'❌ Failed to locate element {css_selector} on page {_log_pretty_url(page.url)}: {type(e).__name__}: {e}'
			)
			return None

	@require_healthy_browser(usable_page=True, reopen_page=True)
	@time_execution_async('--get_locate_element_by_text')
	async def get_locate_element_by_text(
		self, text: str, nth: int | None = 0, element_type: str | None = None
	) -> ElementHandle | None:
		"""
		Locates an element on the page using the provided text.
		If `nth` is provided, it returns the nth matching element (0-based).
		If `element_type` is provided, filters by tag name (e.g., 'button', 'span').
		"""
		page = await self.get_current_page()
		try:
			# handle also specific element type or use any type.
			selector = f'{element_type or "*"}:text("{text}")'
			elements = await page.query_selector_all(selector)
			# considering only visible elements
			elements = [el for el in elements if await self._is_visible(el)]

			if not elements:
				self.logger.error(f"❌ No visible element with text '{text}' found on page {_log_pretty_url(page.url)}.")
				return None

			if nth is not None:
				if 0 <= nth < len(elements):
					element_handle = elements[nth]
				else:
					self.logger.error(
						f"❌ Visible element with text '{text}' not found at index #{nth} on page {_log_pretty_url(page.url)}."
					)
					return None
			else:
				element_handle = elements[0]

			is_visible = await self._is_visible(element_handle)
			if is_visible:
				await element_handle.scroll_into_view_if_needed(timeout=1_000)
			return element_handle
		except Exception as e:
			self.logger.error(
				f"❌ Failed to locate element by text '{text}' on page {_log_pretty_url(page.url)}: {type(e).__name__}: {e}"
			)
			return None

	@require_healthy_browser(usable_page=True, reopen_page=True)
	@time_execution_async('--input_text_element_node')
	@observe_debug(ignore_input=True, name='input_text_element_node')
	async def _input_text_element_node(self, element_node: DOMElementNode, text: str):
		"""
		Input text into an element with proper error handling and state management.
		Handles different types of input fields and ensures proper element state before input.
		"""
		try:
			element_handle = await self.get_locate_element(element_node)

			if element_handle is None:
				raise BrowserError(f'Element: {repr(element_node)} not found')

			# Ensure element is ready for input
			try:
				await element_handle.wait_for_element_state('stable', timeout=1_000)
				is_visible = await self._is_visible(element_handle)
				if is_visible:
					await element_handle.scroll_into_view_if_needed(timeout=1_000)
			except Exception:
				pass

			# let's first try to click and type
			try:
				await element_handle.evaluate('el => {el.textContent = ""; el.value = "";}')
				await element_handle.click(timeout=2_000)  # Add 2 second timeout
				await asyncio.sleep(0.1)  # Increased sleep time
				page = await self.get_current_page()
				await page.keyboard.type(text)
				return
			except Exception as e:
				self.logger.debug(f'Input text with click and type failed, trying element handle method: {e}')
				pass

			# Get element properties to determine input method
			tag_handle = await element_handle.get_property('tagName')
			tag_name = (await tag_handle.json_value()).lower()
			is_contenteditable = await element_handle.get_property('isContentEditable')
			readonly_handle = await element_handle.get_property('readOnly')
			disabled_handle = await element_handle.get_property('disabled')

			readonly = await readonly_handle.json_value() if readonly_handle else False
			disabled = await disabled_handle.json_value() if disabled_handle else False

			try:
				if (await is_contenteditable.json_value() or tag_name == 'input') and not (readonly or disabled):
					await element_handle.evaluate('el => {el.textContent = ""; el.value = "";}')
					await element_handle.type(text, delay=5, timeout=15_000)  # Add 5 second timeout
				else:
					# Try fill() first for supported elements
					try:
						await element_handle.fill(text, timeout=3_000)  # Add 3 second timeout
					except Exception as fill_error:
						# If fill() fails because element doesn't support it, try type() instead
						if 'not an <input>, <textarea>, <select>' in str(fill_error):
							self.logger.debug(f'Element does not support fill(), using type() instead: {fill_error}')
							await element_handle.evaluate('el => {el.textContent = ""; el.value = "";}')
							await element_handle.type(text, delay=5, timeout=15_000)
						else:
							raise
			except Exception as e:
				self.logger.error(f'Error during input text into element: {type(e).__name__}: {e}')
				raise BrowserError(f'Failed to input text into element: {repr(element_node)}')

		except Exception as e:
			# Get current page URL safely for error message
			try:
				page = await self.get_current_page()
				page_url = _log_pretty_url(page.url)
			except Exception:
				page_url = 'unknown page'

			self.logger.debug(
				f'❌ Failed to input text into element: {repr(element_node)} on page {page_url}: {type(e).__name__}: {e}'
			)
			raise BrowserError(f'Failed to input text into index {element_node.highlight_index}')

	@require_healthy_browser(usable_page=True, reopen_page=True)
	@time_execution_async('--switch_to_tab')
	async def switch_to_tab(self, page_id: int) -> Page:
		"""Switch to a specific tab by its page_id (aka tab index exposed to LLM)"""
		assert self.browser_context is not None, 'Browser context is not set'
		pages = self.browser_context.pages

		if page_id >= len(pages):
			raise BrowserError(f'No tab found with page_id: {page_id}')

		page = pages[page_id]

		# Check if the tab's URL is allowed before switching
		if not self._is_url_allowed(page.url):
			raise BrowserError(f'Cannot switch to tab with non-allowed URL: {page.url}')

		# Update both tab references - agent wants this tab, and it's now in the foreground
		self.agent_current_page = page
		await self.agent_current_page.bring_to_front()  # crucial for screenshot to work

		# in order for a human watching to be able to follow along with what the agent is doing
		# update the human's active tab to match the agent's
		if self.human_current_page != page:
			# TODO: figure out how to do this without bringing the entire window to the foreground and stealing foreground app focus
			# might require browser-use extension loaded into the browser so we can use chrome.tabs extension APIs
			# await page.bring_to_front()
			pass

		self.human_current_page = page

		# Invalidate cached state since we've switched to a different tab
		# The cached state contains DOM elements and selector map from the previous tab
		self._cached_browser_state_summary = None
		self._cached_clickable_element_hashes = None

		try:
			await page.wait_for_load_state()
		except Exception as e:
			self.logger.warning(f'⚠️ New page failed to fully load: {type(e).__name__}: {e}')

		# Set the viewport size for the tab
		if self.browser_profile.viewport:
			await page.set_viewport_size(self.browser_profile.viewport)

		return page

	# region - Helper methods for easier access to the DOM
	@observe_debug(name='get_selector_map', ignore_output=True, ignore_input=True)
	@require_healthy_browser(usable_page=True, reopen_page=True)
	async def get_selector_map(self) -> SelectorMap:
		if self._cached_browser_state_summary is None:
			return {}
		return self._cached_browser_state_summary.selector_map

	@observe_debug(ignore_input=True, ignore_output=True, name='get_element_by_index')
	@require_healthy_browser(usable_page=True, reopen_page=True)
	async def get_element_by_index(self, index: int) -> ElementHandle | None:
		selector_map = await self.get_selector_map()
		element_handle = await self.get_locate_element(selector_map[index])
		return element_handle

	@observe_debug(ignore_input=True, ignore_output=True, name='is_file_input_by_index')
	async def is_file_input_by_index(self, index: int) -> bool:
		try:
			selector_map = await self.get_selector_map()
			node = selector_map[index]
			return self.is_file_input(node)
		except Exception as e:
			self.logger.debug(f'❌ Error in is_file_input(index={index}): {type(e).__name__}: {e}')
			return False

	@staticmethod
	def is_file_input(node: DOMElementNode) -> bool:
		return (
			isinstance(node, DOMElementNode)
			and getattr(node, 'tag_name', '').lower() == 'input'
			and node.attributes.get('type', '').lower() == 'file'
		)

	@require_healthy_browser(usable_page=True, reopen_page=True)
	async def find_file_upload_element_by_index(
		self, index: int, max_height: int = 3, max_descendant_depth: int = 3
	) -> DOMElementNode | None:
		"""
		Find the closest file input to the selected element by traversing the DOM bottom-up.
		At each level (up to max_height ancestors):
		- Check the current node itself
		- Check all its children/descendants up to max_descendant_depth
		- Check all siblings (and their descendants up to max_descendant_depth)
		Returns the first file input found, or None if not found.
		"""
		try:
			selector_map = await self.get_selector_map()
			if index not in selector_map:
				return None

			candidate_element = selector_map[index]

			def find_file_input_in_descendants(node: DOMElementNode, depth: int) -> DOMElementNode | None:
				if depth < 0 or not isinstance(node, DOMElementNode):
					return None
				if self.is_file_input(node):
					return node
				for child in getattr(node, 'children', []):
					result = find_file_input_in_descendants(child, depth - 1)
					if result:
						return result
				return None

			current = candidate_element
			for _ in range(max_height + 1):  # include the candidate itself
				# 1. Check the current node itself
				if self.is_file_input(current):
					return current
				# 2. Check all descendants of the current node
				result = find_file_input_in_descendants(current, max_descendant_depth)
				if result:
					return result
				# 3. Check all siblings and their descendants
				parent = getattr(current, 'parent', None)
				if parent:
					for sibling in getattr(parent, 'children', []):
						if sibling is current:
							continue
						if self.is_file_input(sibling):
							return sibling
						result = find_file_input_in_descendants(sibling, max_descendant_depth)
						if result:
							return result
				current = parent
				if not current:
					break
			return None
		except Exception as e:
			page = await self.get_current_page()
			self.logger.debug(
				f'❌ Error in find_file_upload_element_by_index(index={index}) on page {_log_pretty_url(page.url)}: {type(e).__name__}: {e}'
			)
			return None

	@require_healthy_browser(usable_page=True, reopen_page=True)
	async def get_scroll_info(self, page: Page) -> tuple[int, int]:
		"""Get scroll position information for the current page."""
		scroll_y = await page.evaluate('window.scrollY')
		viewport_height = await page.evaluate('window.innerHeight')
		total_height = await page.evaluate('document.documentElement.scrollHeight')
		# Convert to int to handle fractional pixels
		pixels_above = int(scroll_y)
		pixels_below = int(max(0, total_height - (scroll_y + viewport_height)))
		return pixels_above, pixels_below

	@require_healthy_browser(usable_page=True, reopen_page=True)
	async def get_page_info(self, page: Page) -> PageInfo:
		"""Get comprehensive page size and scroll information."""
		# Get all page dimensions and scroll info in one JavaScript call for efficiency
		page_data = await page.evaluate("""() => {
			return {
				// Current viewport dimensions
				viewport_width: window.innerWidth,
				viewport_height: window.innerHeight,
				
				// Total page dimensions
				page_width: Math.max(
					document.documentElement.scrollWidth,
					document.body.scrollWidth || 0
				),
				page_height: Math.max(
					document.documentElement.scrollHeight,
					document.body.scrollHeight || 0
				),
				
				// Current scroll position
				scroll_x: window.scrollX || window.pageXOffset || document.documentElement.scrollLeft || 0,
				scroll_y: window.scrollY || window.pageYOffset || document.documentElement.scrollTop || 0
			};
		}""")

		# Calculate derived values (convert to int to handle fractional pixels)
		viewport_width = int(page_data['viewport_width'])
		viewport_height = int(page_data['viewport_height'])
		page_width = int(page_data['page_width'])
		page_height = int(page_data['page_height'])
		scroll_x = int(page_data['scroll_x'])
		scroll_y = int(page_data['scroll_y'])

		# Calculate scroll information
		pixels_above = scroll_y
		pixels_below = max(0, page_height - (scroll_y + viewport_height))
		pixels_left = scroll_x
		pixels_right = max(0, page_width - (scroll_x + viewport_width))

		# Create PageInfo object with comprehensive information
		page_info = PageInfo(
			viewport_width=viewport_width,
			viewport_height=viewport_height,
			page_width=page_width,
			page_height=page_height,
			scroll_x=scroll_x,
			scroll_y=scroll_y,
			pixels_above=pixels_above,
			pixels_below=pixels_below,
			pixels_left=pixels_left,
			pixels_right=pixels_right,
		)

		return page_info

	async def _scroll_with_cdp_gesture(self, page: Page, pixels: int) -> bool:
		"""
		Scroll using CDP Input.synthesizeScrollGesture for universal compatibility.

		Args:
			page: The page to scroll
			pixels: Number of pixels to scroll (positive = up, negative = down)

		Returns:
			True if successful, False if failed
		"""
		try:
			# Use CDP to synthesize scroll gesture - works in all contexts including PDFs
			cdp_session = await page.context.new_cdp_session(page)  # type: ignore

			# Get viewport center for scroll origin
			viewport = await page.evaluate("""
				() => ({
					width: window.innerWidth,
					height: window.innerHeight
				})
			""")

			center_x = viewport['width'] // 2
			center_y = viewport['height'] // 2

			await cdp_session.send(
				'Input.synthesizeScrollGesture',
				{
					'x': center_x,
					'y': center_y,
					'xDistance': 0,
					'yDistance': -pixels,  # Negative = scroll down, Positive = scroll up
					'gestureSourceType': 'mouse',  # Use mouse gestures for better compatibility
					'speed': 3000,  # Pixels per second
				},
			)

			try:
				await asyncio.wait_for(cdp_session.detach(), timeout=1.0)
			except (TimeoutError, Exception):
				pass
			self.logger.debug(f'📄 Scrolled via CDP Input.synthesizeScrollGesture: {pixels}px')
			return True

		except Exception as e:
			self.logger.warning(f'❌ Scrolling via CDP Input.synthesizeScrollGesture failed: {type(e).__name__}: {e}')
			return False

	@require_healthy_browser(usable_page=True, reopen_page=True)
	async def _scroll_container(self, pixels: int) -> None:
		"""Scroll using CDP gesture synthesis with JavaScript fallback."""

		page = await self.get_current_page()

		# Try CDP scroll gesture first (works universally including PDFs)
		if await self._scroll_with_cdp_gesture(page, pixels):
			return

		# Fallback to JavaScript for older browsers or when CDP fails
		self.logger.debug('Falling back to JavaScript scrolling')
		SMART_SCROLL_JS = """(dy) => {
			const bigEnough = el => el.clientHeight >= window.innerHeight * 0.5;
			const canScroll = el =>
				el &&
				/(auto|scroll|overlay)/.test(getComputedStyle(el).overflowY) &&
				el.scrollHeight > el.clientHeight &&
				bigEnough(el);

			let el = document.activeElement;
			while (el && !canScroll(el) && el !== document.body) el = el.parentElement;

			el = canScroll(el)
					? el
					: [...document.querySelectorAll('*')].find(canScroll)
					|| document.scrollingElement
					|| document.documentElement;

			if (el === document.scrollingElement ||
				el === document.documentElement ||
				el === document.body) {
				window.scrollBy(0, dy);
			} else {
				el.scrollBy({ top: dy, behavior: 'auto' });
			}
		}"""
		await page.evaluate(SMART_SCROLL_JS, pixels)

	# --- DVD Screensaver Loading Animation Helper ---
	async def _show_dvd_screensaver_loading_animation(self, page: Page) -> None:
		"""
		Injects a DVD screensaver-style bouncing logo loading animation overlay into the given Playwright Page.
		This is used to visually indicate that the browser is setting up or waiting.
		"""
		if CONFIG.IS_IN_EVALS:
			# dont bother wasting CPU showing animations during evals
			return

		# we could enforce this, but maybe it's useful to be able to show it on other tabs?
		# assert is_new_tab_page(page.url), 'DVD screensaver loading animation should only be shown on new tab pages'

		# all in one JS function for speed, we want as few roundtrip CDP calls as possible
		# between opening the tab and showing the animation
		try:
			await page.evaluate(
				"""(browser_session_label) => {
				// Ensure document.body exists before proceeding
				if (!document.body) {
					// Try again after DOM is ready
					if (document.readyState === 'loading') {
						document.addEventListener('DOMContentLoaded', () => arguments.callee(browser_session_label));
					}
					return;
				}
				
				const animated_title = `Starting agent ${browser_session_label}...`;
				if (document.title === animated_title) {
					return;      // already run on this tab, dont run again
				}
				document.title = animated_title;

				// Create the main overlay
				const loadingOverlay = document.createElement('div');
				loadingOverlay.id = 'pretty-loading-animation';
				loadingOverlay.style.position = 'fixed';
				loadingOverlay.style.top = '0';
				loadingOverlay.style.left = '0';
				loadingOverlay.style.width = '100vw';
				loadingOverlay.style.height = '100vh';
				loadingOverlay.style.background = '#000';
				loadingOverlay.style.zIndex = '99999';
				loadingOverlay.style.overflow = 'hidden';

				// Create the image element
				const img = document.createElement('img');
				img.src = 'https://cf.browser-use.com/logo.svg';
				img.alt = 'Browser-Use';
				img.style.width = '200px';
				img.style.height = 'auto';
				img.style.position = 'absolute';
				img.style.left = '0px';
				img.style.top = '0px';
				img.style.zIndex = '2';
				img.style.opacity = '0.8';

				loadingOverlay.appendChild(img);
				document.body.appendChild(loadingOverlay);

				// DVD screensaver bounce logic
				let x = Math.random() * (window.innerWidth - 300);
				let y = Math.random() * (window.innerHeight - 300);
				let dx = 1.2 + Math.random() * 0.4; // px per frame
				let dy = 1.2 + Math.random() * 0.4;
				// Randomize direction
				if (Math.random() > 0.5) dx = -dx;
				if (Math.random() > 0.5) dy = -dy;

				function animate() {
					const imgWidth = img.offsetWidth || 300;
					const imgHeight = img.offsetHeight || 300;
					x += dx;
					y += dy;

					if (x <= 0) {
						x = 0;
						dx = Math.abs(dx);
					} else if (x + imgWidth >= window.innerWidth) {
						x = window.innerWidth - imgWidth;
						dx = -Math.abs(dx);
					}
					if (y <= 0) {
						y = 0;
						dy = Math.abs(dy);
					} else if (y + imgHeight >= window.innerHeight) {
						y = window.innerHeight - imgHeight;
						dy = -Math.abs(dy);
					}

					img.style.left = `${x}px`;
					img.style.top = `${y}px`;

					requestAnimationFrame(animate);
				}
				animate();

				// Responsive: update bounds on resize
				window.addEventListener('resize', () => {
					x = Math.min(x, window.innerWidth - img.offsetWidth);
					y = Math.min(y, window.innerHeight - img.offsetHeight);
				});

				// Add a little CSS for smoothness
				const style = document.createElement('style');
				style.textContent = `
					#pretty-loading-animation {
						/*backdrop-filter: blur(2px) brightness(0.9);*/
					}
					#pretty-loading-animation img {
						user-select: none;
						pointer-events: none;
					}
				`;
				document.head.appendChild(style);
			}""",
				str(self.id)[-4:],
			)
		except Exception as e:
			self.logger.debug(f'❌ Failed to show 📀 DVD loading animation: {type(e).__name__}: {e}')

	@observe_debug(ignore_input=True, ignore_output=True, name='get_browser_state_with_recovery')
	async def get_browser_state_with_recovery(
		self, cache_clickable_elements_hashes: bool = True, include_screenshot: bool = True
	) -> BrowserStateSummary:
		"""Get browser state with multiple fallback strategies for error recovery

		Parameters:
		-----------
		cache_clickable_elements_hashes: bool
			If True, cache the clickable elements hashes for the current state.
		include_screenshot: bool
			If True, include screenshot in the state summary. Set to False to improve performance
			when screenshots are not needed (e.g., in multi_act element validation).
		"""

		# Try 1: Full state summary (current implementation) - like main branch
		try:
			await self._wait_for_page_and_frames_load()
			return await self.get_state_summary(cache_clickable_elements_hashes, include_screenshot=include_screenshot)
		except Exception as e:
			self.logger.warning(f'Full state retrieval failed: {type(e).__name__}: {e}')

		self.logger.warning('🔄 Falling back to minimal state summary')
		return await self.get_minimal_state_summary()

	async def _is_pdf_viewer(self, page: Page) -> bool:
		"""
		Check if the current page is displaying a PDF in Chrome's PDF viewer.
		Returns True if PDF is detected, False otherwise.
		"""
		try:
			is_pdf_viewer = await page.evaluate("""
				() => {
					// Check for Chrome's built-in PDF viewer (updated selector)
					const pdfEmbed = document.querySelector('embed[type="application/x-google-chrome-pdf"]') ||
									 document.querySelector('embed[type="application/pdf"]');
					const isPdfViewer = !!pdfEmbed;
					
					// Also check if the URL ends with .pdf or has PDF content-type
					const url = window.location.href;
					const isPdfUrl = url.toLowerCase().includes('.pdf') || 
									document.contentType === 'application/pdf';
					
					return isPdfViewer || isPdfUrl;
				}
			""")
			return is_pdf_viewer
		except Exception as e:
			self.logger.debug(f'Error checking PDF viewer: {type(e).__name__}: {e}')
			return False

	async def _auto_download_pdf_if_needed(self, page: Page) -> str | None:
		"""
		Check if the current page is a PDF viewer and automatically download the PDF if so.
		Returns the download path if a PDF was downloaded, None otherwise.
		"""
		if not self.browser_profile.downloads_path or not self._auto_download_pdfs:
			return None

		try:
			# Check if we're in a PDF viewer
			is_pdf_viewer = await self._is_pdf_viewer(page)
			self.logger.debug(f'is_pdf_viewer: {is_pdf_viewer}')

			if not is_pdf_viewer:
				return None

			# Get the PDF URL
			pdf_url = page.url

			# Check if we've already downloaded this PDF
			pdf_filename = os.path.basename(pdf_url.split('?')[0])  # Remove query params
			if not pdf_filename or not pdf_filename.endswith('.pdf'):
				# Generate filename from URL
				from urllib.parse import urlparse

				parsed = urlparse(pdf_url)
				pdf_filename = os.path.basename(parsed.path) or 'document.pdf'
				if not pdf_filename.endswith('.pdf'):
					pdf_filename += '.pdf'

			# Check if already downloaded
			expected_path = os.path.join(self.browser_profile.downloads_path, pdf_filename)
			if any(os.path.basename(downloaded) == pdf_filename for downloaded in self._downloaded_files):
				self.logger.debug(f'📄 PDF already downloaded: {pdf_filename}')
				return None

			self.logger.info(f'📄 Auto-downloading PDF from: {pdf_url}')

			# Download the actual PDF file using JavaScript fetch
			# Note: This should hit the browser cache since Chrome already downloaded the PDF to display it
			try:
				self.logger.debug(f'Downloading PDF from URL: {pdf_url}')

				# Properly escape the URL to prevent JavaScript injection
				escaped_pdf_url = json.dumps(pdf_url)

				download_result = await page.evaluate(f"""
					async () => {{
						try {{
							// Use fetch with cache: 'force-cache' to prioritize cached version
							const response = await fetch({escaped_pdf_url}, {{
								cache: 'force-cache'
							}});
							if (!response.ok) {{
								throw new Error(`HTTP error! status: ${{response.status}}`);
							}}
							const blob = await response.blob();
							const arrayBuffer = await blob.arrayBuffer();
							const uint8Array = new Uint8Array(arrayBuffer);
							
							// Log whether this was served from cache
							const fromCache = response.headers.has('age') || 
											 !response.headers.has('date') ||
											 performance.getEntriesByName({escaped_pdf_url}).some(entry => 
												 entry.transferSize === 0 || entry.transferSize < entry.encodedBodySize
											 );
											 
							return {{ 
								data: Array.from(uint8Array),
								fromCache: fromCache,
								responseSize: uint8Array.length,
								transferSize: response.headers.get('content-length') || 'unknown'
							}};
						}} catch (error) {{
							throw new Error(`Fetch failed: ${{error.message}}`);
						}}
					}}
				""")

				if download_result and download_result.get('data') and len(download_result['data']) > 0:
					# Ensure unique filename
					unique_filename = await self._get_unique_filename(self.browser_profile.downloads_path, pdf_filename)
					download_path = os.path.join(self.browser_profile.downloads_path, unique_filename)

					# Save the PDF asynchronously
					async with await anyio.open_file(download_path, 'wb') as f:
						await f.write(bytes(download_result['data']))

					# Track the downloaded file
					self._downloaded_files.append(download_path)

					# Log cache information
					cache_status = 'from cache' if download_result.get('fromCache') else 'from network'
					response_size = download_result.get('responseSize', 0)
					self.logger.info(f'📄 Auto-downloaded PDF ({cache_status}, {response_size:,} bytes): {download_path}')

					return download_path
				else:
					self.logger.warning(f'⚠️ No data received when downloading PDF from {pdf_url}')
					return None

			except Exception as e:
				self.logger.warning(f'⚠️ Failed to auto-download PDF from {pdf_url}: {type(e).__name__}: {e}')
				return None

		except Exception as e:
			self.logger.warning(f'⚠️ Error in PDF auto-download check: {type(e).__name__}: {e}')
			return None

# From browser/session.py
def require_healthy_browser(usable_page=True, reopen_page=True):
	"""Decorator for BrowserSession methods to ensure browser/page is healthy before execution.

	This ridiculous overengineered logic is necessary to work around playwright's completely broken handling of crashed pages.
	- When a page is loading, playwright calls will hang indefinitely.
	- When a page is blocked by a JS while(true){}, playwright calls will hang indefinitely.
	- When a page is unresponsive because the system is out of CPU or Memory, playwright calls will hang indefinitely.
	asyncio.wait(...) is the most extreme method available to try and terminate asyncio tasks in python, but even this does not work
	because it's likely the underlying playwright node.js process that's crashing and synchronously blocking the python side.
	This is why we must use CDP directly and skip playwright eventually.

	Args:
		usable_page: If True, check that the agent_current_page is valid and responsive before executing the method, if invalid log it but continue anyway
		reopen_page: If True, attempt to reopen the page if it's crashed, invalid, or unresponsive (only applies if usable_page=True)
	"""

	def decorator(func):
		assert asyncio.iscoroutinefunction(func), '@require_healthy_browser only supports async methods'

		@wraps(func)
		async def wrapper(self: BrowserSession, *args, **kwargs):
			try:
				if not self.initialized or not self.browser_context:
					# raise RuntimeError('BrowserSession(...).start() must be called first to launch or connect to the browser')
					await self.start()  # just start it automatically if not already started

				if not self.agent_current_page or self.agent_current_page.is_closed():
					self.agent_current_page = (
						self.browser_context.pages[0] if (self.browser_context and len(self.browser_context.pages) > 0) else None
					)

				# always require at least one tab to be open for the context to be considered usable, dont check responsiveness unless usable_page=True
				if not self.agent_current_page or self.agent_current_page.is_closed():
					# Create new page directly to avoid circular dependency
					assert self.browser_context is not None, 'Browser context is not set'
					self.logger.debug(
						f'@require_healthy_browser: Creating new page in {func.__name__} because agent_current_page is closed/missing'
					)
					new_page = await self.browser_context.new_page()
					self.agent_current_page = new_page
					if (not self.human_current_page) or self.human_current_page.is_closed():
						self.human_current_page = new_page
					if self.browser_profile.viewport:
						await new_page.set_viewport_size(self.browser_profile.viewport)

				assert self.agent_current_page and not self.agent_current_page.is_closed()

				if not hasattr(self, '_cached_browser_state_summary'):
					raise RuntimeError('BrowserSession(...).start() must be called first to initialize the browser session')

				# Check page responsiveness if usable_page=True
				if usable_page:
					# Skip if already in recovery to prevent infinite recursion
					if hasattr(self, '_in_recovery') and self._in_recovery:
						# self.logger.debug('Already in recovery, skipping responsiveness check')
						return await func(self, *args, **kwargs)

					# Skip responsiveness check for about:blank pages - they're always responsive (I hope, otherwise something is very wrong)
					if self.agent_current_page and is_new_tab_page(self.agent_current_page.url):
						# self.logger.debug('Skipping responsiveness check for about:blank page')
						return await func(self, *args, **kwargs)

					# Check if page is responsive
					# self.logger.debug(f'Checking page responsiveness for {func.__name__}...')
					if await self._is_page_responsive(self.agent_current_page):
						# self.logger.debug('✅ Confirmed page is responsive')
						pass
					else:
						# Page is unresponsive - handle recovery
						if not reopen_page:
							self.logger.warning(
								'⚠️ Page unresponsive but @require_healthy_browser(reopen_page=False), attempting to continue anyway...'
							)
						else:
							try:
								await self._recover_unresponsive_page(
									func.__name__, timeout_ms=int(self.browser_profile.default_navigation_timeout or 5000) + 5_000
								)
								page_url = self.agent_current_page.url if self.agent_current_page else 'unknown page'
								self.logger.debug(
									f'🤕 Crashed page recovery finished, attempting to continue with {func.__name__}() on {_log_pretty_url(page_url)}...'
								)
							except Exception as e:
								page_url = self.agent_current_page.url if self.agent_current_page else 'unknown page'
								self.logger.warning(
									f'❌ Crashed page recovery failed, could not run {func.__name__}(), page is stuck unresponsive on {_log_pretty_url(page_url)}...'
								)
								raise  # Re-raise to let retry decorator / callsite handle it

				return await func(self, *args, **kwargs)

			except Exception as e:
				# Check if this is a TargetClosedError or similar connection error
				if 'TargetClosedError' in str(type(e)) or 'browser has been closed' in str(e):
					self.logger.warning(
						f'✂️ Browser {self._connection_str} disconnected before BrowserSession.{func.__name__}() could run... (error: {type(e).__name__}: {e})'
					)
					self._reset_connection_state()
				# Re-raise all hard errors so the caller can handle them appropriately
				raise

		return wrapper

	return decorator

# From browser/session.py
def apply_session_overrides_to_profile(self) -> Self:
		"""Apply any extra **kwargs passed to BrowserSession(...) as session-specific config overrides on top of browser_profile"""
		session_own_fields = type(self).model_fields.keys()

		# get all the extra kwarg overrides passed to BrowserSession(...) that are actually
		# config Fields tracked by BrowserProfile, instead of BrowserSession's own args
		profile_overrides = self.model_dump(exclude=set(session_own_fields))

		# FOR REPL DEBUGGING ONLY, NEVER ALLOW CIRCULAR REFERENCES IN REAL CODE:
		# self.browser_profile._in_use_by_session = self

		self.browser_profile = self.browser_profile.model_copy(update=profile_overrides)

		# FOR REPL DEBUGGING ONLY, NEVER ALLOW CIRCULAR REFERENCES IN REAL CODE:
		# self.browser_profile._in_use_by_session = self

		return self

# From browser/session.py
def set_browser_ownership(self) -> Self:
		"""Set _owns_browser_resources based on whether we're connecting to an external browser"""
		# If user provided CDP URL, WSS URL, or existing browser/context, we don't own the browser
		if self.cdp_url or self.wss_url or self.browser or self.browser_context:
			self._owns_browser_resources = False
		return self

# From browser/session.py
def model_copy(self, **kwargs) -> Self:
		"""Create a copy of this BrowserSession that shares the browser resources but doesn't own them.

		This method creates a copy that:
		- Shares the same browser, browser_context, and playwright objects
		- Doesn't own the browser resources (won't close them when garbage collected)
		- Keeps a reference to the original to prevent premature garbage collection
		"""
		# Create the copy using the parent class method
		copy = super().model_copy(**kwargs)

		# The copy doesn't own the browser resources
		copy._owns_browser_resources = False

		# Keep a reference to the original to prevent garbage collection
		copy._original_browser_session = self

		# Manually copy over the excluded fields that are needed for browser connection
		# These fields are excluded in the model config but need to be shared
		copy.playwright = self.playwright
		copy.browser = self.browser
		copy.browser_context = self.browser_context
		copy.agent_current_page = self.agent_current_page
		copy.human_current_page = self.human_current_page
		copy.browser_pid = self.browser_pid

		return copy

# From browser/session.py
def prepare_user_data_dir(self, check_conflicts: bool = True) -> None:
		"""Create and prepare the user data dir, handling conflicts if needed.

		Args:
			check_conflicts: Whether to check for and handle singleton lock conflicts
		"""
		if self.browser_profile.user_data_dir:
			try:
				self.browser_profile.user_data_dir = Path(self.browser_profile.user_data_dir).expanduser().resolve()
				self.browser_profile.user_data_dir.mkdir(parents=True, exist_ok=True)
				(self.browser_profile.user_data_dir / '.browseruse_profile_id').write_text(self.browser_profile.id)
			except Exception as e:
				raise ValueError(
					f'Unusable path provided for user_data_dir= {_log_pretty_path(self.browser_profile.user_data_dir)} (check for typos/permissions issues)'
				) from e

			# Remove stale singleton lock file ONLY if no process is using this profile
			# This must happen BEFORE checking for conflicts to avoid false positives
			singleton_lock = self.browser_profile.user_data_dir / 'SingletonLock'
			if singleton_lock.exists():
				# Check if any process is actually using this user_data_dir
				has_active_process = False
				target_dir = str(self.browser_profile.user_data_dir)
				for proc in psutil.process_iter(['pid', 'cmdline']):
					# Skip our own browser process
					if hasattr(self, 'browser_pid') and self.browser_pid and proc.info['pid'] == self.browser_pid:
						continue

					cmdline = proc.info['cmdline'] or []
					# Check both formats: --user-data-dir=/path and --user-data-dir /path
					for i, arg in enumerate(cmdline):
						if arg.startswith('--user-data-dir='):
							try:
								if str(Path(arg.split('=', 1)[1]).expanduser().resolve()) == target_dir:
									has_active_process = True
									break
							except Exception:
								if arg.split('=', 1)[1] == str(self.browser_profile.user_data_dir):
									has_active_process = True
									break
						elif arg == '--user-data-dir' and i + 1 < len(cmdline):
							try:
								if str(Path(cmdline[i + 1]).expanduser().resolve()) == target_dir:
									has_active_process = True
									break
							except Exception:
								if cmdline[i + 1] == str(self.browser_profile.user_data_dir):
									has_active_process = True
									break
					if has_active_process:
						break

				if not has_active_process:
					# No active process, safe to remove stale lock
					try:
						# Handle both regular files and symlinks
						if singleton_lock.is_symlink() or singleton_lock.exists():
							singleton_lock.unlink()
							self.logger.debug(
								f'🧹 Removed stale SingletonLock file from {_log_pretty_path(self.browser_profile.user_data_dir)} (no active Chrome process found)'
							)
					except Exception:
						pass  # Ignore errors removing lock file

			# Check for conflicts and fallback if needed (AFTER cleaning stale locks)
			if check_conflicts and self._check_for_singleton_lock_conflict():
				self._fallback_to_temp_profile()
				# Recursive call without conflict checking to prepare the new temp dir
				return self.prepare_user_data_dir(check_conflicts=False)

		# Create directories for all paths that need them
		dir_paths = {
			'downloads_path': self.browser_profile.downloads_path,
			'record_video_dir': self.browser_profile.record_video_dir,
			'traces_dir': self.browser_profile.traces_dir,
		}

		file_paths = {
			'record_har_path': self.browser_profile.record_har_path,
		}

		# Handle directory creation
		for path_name, path_value in dir_paths.items():
			if path_value:
				try:
					path_obj = Path(path_value).expanduser().resolve()
					path_obj.mkdir(parents=True, exist_ok=True)
					setattr(self.browser_profile, path_name, str(path_obj) if path_name == 'traces_dir' else path_obj)
				except Exception as e:
					self.logger.error(f'❌ Failed to create {path_name} directory {path_value}: {e}')

		# Handle file path parent directory creation
		for path_name, path_value in file_paths.items():
			if path_value:
				try:
					path_obj = Path(path_value).expanduser().resolve()
					path_obj.parent.mkdir(parents=True, exist_ok=True)
				except Exception as e:
					self.logger.error(f'❌ Failed to create parent directory for {path_name} {path_value}: {e}')

# From browser/session.py
def tabs(self) -> list[Page]:
		if not self.browser_context:
			return []
		return list(self.browser_context.pages)

# From browser/session.py
def downloaded_files(self) -> list[str]:
		"""
		Get list of all files downloaded during this browser session.

		Returns:
		    list[str]: List of absolute file paths to downloaded files
		"""
		self.logger.debug(f'📁 Retrieved {len(self._downloaded_files)} downloaded files from session tracking')
		return self._downloaded_files.copy()

# From browser/session.py
def set_auto_download_pdfs(self, enabled: bool) -> None:
		"""
		Enable or disable automatic PDF downloading when PDFs are encountered.

		Args:
		    enabled: Whether to automatically download PDFs
		"""
		self._auto_download_pdfs = enabled
		self.logger.info(f'📄 PDF auto-download {"enabled" if enabled else "disabled"}')

# From browser/session.py
def auto_download_pdfs(self) -> bool:
		"""Get current PDF auto-download setting."""
		return self._auto_download_pdfs

# From browser/session.py
def is_file_input(node: DOMElementNode) -> bool:
		return (
			isinstance(node, DOMElementNode)
			and getattr(node, 'tag_name', '').lower() == 'input'
			and node.attributes.get('type', '').lower() == 'file'
		)

# From browser/session.py
def shudown_playwright():
			if not self.playwright:
				return
			try:
				loop = asyncio.get_running_loop()
				self.logger.debug('🛑 Shutting down shared global playwright node.js client')
				task = loop.create_task(self.playwright.stop())
				if hasattr(task, '_log_destroy_pending'):
					task._log_destroy_pending = False  # type: ignore
			except Exception:
				pass
			self.playwright = None

# From browser/session.py
def find_file_input_in_descendants(node: DOMElementNode, depth: int) -> DOMElementNode | None:
				if depth < 0 or not isinstance(node, DOMElementNode):
					return None
				if self.is_file_input(node):
					return node
				for child in getattr(node, 'children', []):
					result = find_file_input_in_descendants(child, depth - 1)
					if result:
						return result
				return None

from dataclasses import field

# From browser/views.py
class TabInfo(BaseModel):
	"""Represents information about a browser tab"""

	page_id: int
	url: str
	title: str
	parent_page_id: int | None = None

# From browser/views.py
class PageInfo(BaseModel):
	"""Comprehensive page size and scroll information"""

	# Current viewport dimensions
	viewport_width: int
	viewport_height: int

	# Total page dimensions
	page_width: int
	page_height: int

	# Current scroll position
	scroll_x: int
	scroll_y: int

	# Calculated scroll information
	pixels_above: int
	pixels_below: int
	pixels_left: int
	pixels_right: int

# From browser/views.py
class BrowserStateSummary(DOMState):
	"""The summary of the browser's current state designed for an LLM to process"""

	# provided by DOMState:
	# element_tree: DOMElementNode
	# selector_map: SelectorMap

	url: str
	title: str
	tabs: list[TabInfo]
	screenshot: str | None = field(default=None, repr=False)
	page_info: PageInfo | None = None  # Enhanced page information

	# Keep legacy fields for backward compatibility
	pixels_above: int = 0
	pixels_below: int = 0
	browser_errors: list[str] = field(default_factory=list)
	is_pdf_viewer: bool = False  # Whether the current page is a PDF viewer
	loading_status: str | None = None

# From browser/views.py
class BrowserStateHistory:
	"""The summary of the browser's state at a past point in time to usse in LLM message history"""

	url: str
	title: str
	tabs: list[TabInfo]
	interacted_element: list[DOMHistoryElement | None] | list[None]
	screenshot_path: str | None = None

	def get_screenshot(self) -> str | None:
		"""Load screenshot from disk and return as base64 string"""
		if not self.screenshot_path:
			return None

		import base64
		from pathlib import Path

		path_obj = Path(self.screenshot_path)
		if not path_obj.exists():
			return None

		try:
			with open(path_obj, 'rb') as f:
				screenshot_data = f.read()
			return base64.b64encode(screenshot_data).decode('utf-8')
		except Exception:
			return None

	def to_dict(self) -> dict[str, Any]:
		data = {}
		data['tabs'] = [tab.model_dump() for tab in self.tabs]
		data['screenshot_path'] = self.screenshot_path
		data['interacted_element'] = [el.to_dict() if el else None for el in self.interacted_element]
		data['url'] = self.url
		data['title'] = self.title
		return data

# From browser/views.py
class BrowserError(Exception):
	"""Base class for all browser errors"""

# From browser/views.py
class URLNotAllowedError(BrowserError):
	"""Error raised when a URL is not allowed"""

# From browser/views.py
def get_screenshot(self) -> str | None:
		"""Load screenshot from disk and return as base64 string"""
		if not self.screenshot_path:
			return None

		import base64
		from pathlib import Path

		path_obj = Path(self.screenshot_path)
		if not path_obj.exists():
			return None

		try:
			with open(path_obj, 'rb') as f:
				screenshot_data = f.read()
			return base64.b64encode(screenshot_data).decode('utf-8')
		except Exception:
			return None

# From browser/views.py
def to_dict(self) -> dict[str, Any]:
		data = {}
		data['tabs'] = [tab.model_dump() for tab in self.tabs]
		data['screenshot_path'] = self.screenshot_path
		data['interacted_element'] = [el.to_dict() if el else None for el in self.interacted_element]
		data['url'] = self.url
		data['title'] = self.title
		return data

from typing import Union
from openai import BaseModel

# From llm/messages.py
class ContentPartTextParam(BaseModel):
	text: str
	type: Literal['text'] = 'text'

	def __str__(self) -> str:
		return f'Text: {_truncate(self.text)}'

	def __repr__(self) -> str:
		return f'ContentPartTextParam(text={_truncate(self.text)})'

# From llm/messages.py
class ContentPartRefusalParam(BaseModel):
	refusal: str
	type: Literal['refusal'] = 'refusal'

	def __str__(self) -> str:
		return f'Refusal: {_truncate(self.refusal)}'

	def __repr__(self) -> str:
		return f'ContentPartRefusalParam(refusal={_truncate(repr(self.refusal), 50)})'

# From llm/messages.py
class ImageURL(BaseModel):
	url: str
	"""Either a URL of the image or the base64 encoded image data."""
	detail: Literal['auto', 'low', 'high'] = 'auto'
	"""Specifies the detail level of the image.

    Learn more in the
    [Vision guide](https://platform.openai.com/docs/guides/vision#low-or-high-fidelity-image-understanding).
    """
	# needed for Anthropic
	media_type: SupportedImageMediaType = 'image/png'

	def __str__(self) -> str:
		url_display = _format_image_url(self.url)
		return f'🖼️  Image[{self.media_type}, detail={self.detail}]: {url_display}'

	def __repr__(self) -> str:
		url_repr = _format_image_url(self.url, 30)
		return f'ImageURL(url={repr(url_repr)}, detail={repr(self.detail)}, media_type={repr(self.media_type)})'

# From llm/messages.py
class ContentPartImageParam(BaseModel):
	image_url: ImageURL
	type: Literal['image_url'] = 'image_url'

	def __str__(self) -> str:
		return str(self.image_url)

	def __repr__(self) -> str:
		return f'ContentPartImageParam(image_url={repr(self.image_url)})'

# From llm/messages.py
class Function(BaseModel):
	arguments: str
	"""
    The arguments to call the function with, as generated by the model in JSON
    format. Note that the model does not always generate valid JSON, and may
    hallucinate parameters not defined by your function schema. Validate the
    arguments in your code before calling your function.
    """
	name: str
	"""The name of the function to call."""

	def __str__(self) -> str:
		args_preview = _truncate(self.arguments, 80)
		return f'{self.name}({args_preview})'

	def __repr__(self) -> str:
		args_repr = _truncate(repr(self.arguments), 50)
		return f'Function(name={repr(self.name)}, arguments={args_repr})'

# From llm/messages.py
class ToolCall(BaseModel):
	id: str
	"""The ID of the tool call."""
	function: Function
	"""The function that the model called."""
	type: Literal['function'] = 'function'
	"""The type of the tool. Currently, only `function` is supported."""

	def __str__(self) -> str:
		return f'ToolCall[{self.id}]: {self.function}'

	def __repr__(self) -> str:
		return f'ToolCall(id={repr(self.id)}, function={repr(self.function)})'

# From llm/messages.py
class _MessageBase(BaseModel):
	"""Base class for all message types"""

	role: Literal['user', 'system', 'assistant']

	cache: bool = False
	"""Whether to cache this message. This is only applicable when using Anthropic models.
	"""

# From llm/messages.py
class UserMessage(_MessageBase):
	role: Literal['user'] = 'user'
	"""The role of the messages author, in this case `user`."""

	content: str | list[ContentPartTextParam | ContentPartImageParam]
	"""The contents of the user message."""

	name: str | None = None
	"""An optional name for the participant.

    Provides the model information to differentiate between participants of the same
    role.
    """

	@property
	def text(self) -> str:
		"""
		Automatically parse the text inside content, whether it's a string or a list of content parts.
		"""
		if isinstance(self.content, str):
			return self.content
		elif isinstance(self.content, list):
			return '\n'.join([part.text for part in self.content if part.type == 'text'])
		else:
			return ''

	def __str__(self) -> str:
		return f'UserMessage(content={self.text})'

	def __repr__(self) -> str:
		return f'UserMessage(content={repr(self.text)})'

# From llm/messages.py
class SystemMessage(_MessageBase):
	role: Literal['system'] = 'system'
	"""The role of the messages author, in this case `system`."""

	content: str | list[ContentPartTextParam]
	"""The contents of the system message."""

	name: str | None = None

	@property
	def text(self) -> str:
		"""
		Automatically parse the text inside content, whether it's a string or a list of content parts.
		"""
		if isinstance(self.content, str):
			return self.content
		elif isinstance(self.content, list):
			return '\n'.join([part.text for part in self.content if part.type == 'text'])
		else:
			return ''

	def __str__(self) -> str:
		return f'SystemMessage(content={self.text})'

	def __repr__(self) -> str:
		return f'SystemMessage(content={repr(self.text)})'

# From llm/messages.py
class AssistantMessage(_MessageBase):
	role: Literal['assistant'] = 'assistant'
	"""The role of the messages author, in this case `assistant`."""

	content: str | list[ContentPartTextParam | ContentPartRefusalParam] | None
	"""The contents of the assistant message."""

	name: str | None = None

	refusal: str | None = None
	"""The refusal message by the assistant."""

	tool_calls: list[ToolCall] = []
	"""The tool calls generated by the model, such as function calls."""

	@property
	def text(self) -> str:
		"""
		Automatically parse the text inside content, whether it's a string or a list of content parts.
		"""
		if isinstance(self.content, str):
			return self.content
		elif isinstance(self.content, list):
			text = ''
			for part in self.content:
				if part.type == 'text':
					text += part.text
				elif part.type == 'refusal':
					text += f'[Refusal] {part.refusal}'
			return text
		else:
			return ''

	def __str__(self) -> str:
		return f'AssistantMessage(content={self.text})'

	def __repr__(self) -> str:
		return f'AssistantMessage(content={repr(self.text)})'

# From llm/messages.py
def text(self) -> str:
		"""
		Automatically parse the text inside content, whether it's a string or a list of content parts.
		"""
		if isinstance(self.content, str):
			return self.content
		elif isinstance(self.content, list):
			return '\n'.join([part.text for part in self.content if part.type == 'text'])
		else:
			return ''

from typing import Protocol
from typing import overload
from typing import runtime_checkable
from browser_use.llm.views import ChatInvokeCompletion
from pydantic_core import core_schema

# From llm/base.py
class BaseChatModel(Protocol):
	_verified_api_keys: bool = False

	model: str

	@property
	def provider(self) -> str: ...

	@property
	def name(self) -> str: ...

	@property
	def model_name(self) -> str:
		# for legacy support
		return self.model

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]: ...

	@classmethod
	def __get_pydantic_core_schema__(
		cls,
		source_type: type,
		handler: Any,
	) -> Any:
		"""
		Allow this Protocol to be used in Pydantic models -> very useful to typesafe the agent settings for example.
		Returns a schema that allows any object (since this is a Protocol).
		"""
		from pydantic_core import core_schema

		# Return a schema that accepts any object for Protocol types
		return core_schema.any_schema()

# From llm/base.py
def provider(self) -> str: ...

# From llm/base.py
def model_name(self) -> str:
		# for legacy support
		return self.model


# From llm/exceptions.py
class ModelError(Exception):
	pass

# From llm/exceptions.py
class ModelProviderError(ModelError):
	"""Exception raised when a model provider returns an error."""

	def __init__(
		self,
		message: str,
		status_code: int = 502,
		model: str | None = None,
	):
		super().__init__(message, status_code)
		self.model = model

# From llm/exceptions.py
class ModelRateLimitError(ModelProviderError):
	"""Exception raised when a model provider returns a rate limit error."""

	def __init__(
		self,
		message: str,
		status_code: int = 429,
		model: str | None = None,
	):
		super().__init__(message, status_code, model)


# From llm/schema.py
class SchemaOptimizer:
	@staticmethod
	def create_optimized_json_schema(model: type[BaseModel]) -> dict[str, Any]:
		"""
		Create the most optimized schema by flattening all $ref/$defs while preserving
		FULL descriptions and ALL action definitions. Also ensures OpenAI strict mode compatibility.

		Args:
			model: The Pydantic model to optimize

		Returns:
			Optimized schema with all $refs resolved and strict mode compatibility
		"""
		# Generate original schema
		original_schema = model.model_json_schema()

		# Extract $defs for reference resolution, then flatten everything
		defs_lookup = original_schema.get('$defs', {})

		def optimize_schema(
			obj: Any,
			defs_lookup: dict[str, Any] | None = None,
			*,
			in_properties: bool = False,  # NEW: track context
		) -> Any:
			"""Apply all optimization techniques including flattening all $ref/$defs"""
			if isinstance(obj, dict):
				optimized: dict[str, Any] = {}
				flattened_ref: dict[str, Any] | None = None

				# Skip unnecessary fields AND $defs (we'll inline everything)
				skip_fields = ['additionalProperties', '$defs']

				for key, value in obj.items():
					if key in skip_fields:
						continue

					# Skip metadata "title" unless we're iterating inside an actual `properties` map
					if key == 'title' and not in_properties:
						continue

					# Preserve FULL descriptions without truncation
					elif key == 'description':
						optimized[key] = value

					# Handle type field
					elif key == 'type':
						optimized[key] = value

					# FLATTEN: Resolve $ref by inlining the actual definition
					elif key == '$ref' and defs_lookup:
						ref_path = value.split('/')[-1]  # Get the definition name from "#/$defs/SomeName"
						if ref_path in defs_lookup:
							# Get the referenced definition and flatten it
							referenced_def = defs_lookup[ref_path]
							flattened_ref = optimize_schema(referenced_def, defs_lookup)

					# Keep all anyOf structures (action unions) and resolve any $refs within
					elif key == 'anyOf' and isinstance(value, list):
						optimized[key] = [optimize_schema(item, defs_lookup) for item in value]

					# Recursively optimize nested structures
					elif key in ['properties', 'items']:
						optimized[key] = optimize_schema(
							value,
							defs_lookup,
							in_properties=(key == 'properties'),
						)

					# Keep essential validation fields
					elif key in ['type', 'required', 'minimum', 'maximum', 'minItems', 'maxItems', 'pattern', 'default']:
						optimized[key] = value if not isinstance(value, (dict, list)) else optimize_schema(value, defs_lookup)

					# Recursively process all other fields
					else:
						optimized[key] = optimize_schema(value, defs_lookup) if isinstance(value, (dict, list)) else value

				# If we have a flattened reference, merge it with the optimized properties
				if flattened_ref is not None and isinstance(flattened_ref, dict):
					# Start with the flattened reference as the base
					result = flattened_ref.copy()

					# Merge in any sibling properties that were processed
					for key, value in optimized.items():
						# Preserve descriptions from the original object if they exist
						if key == 'description' and 'description' not in result:
							result[key] = value
						elif key != 'description':  # Don't overwrite description from flattened ref
							result[key] = value

					return result
				else:
					# No $ref, just return the optimized object
					# CRITICAL: Add additionalProperties: false to ALL objects for OpenAI strict mode
					if optimized.get('type') == 'object':
						optimized['additionalProperties'] = False

					return optimized

			elif isinstance(obj, list):
				return [optimize_schema(item, defs_lookup, in_properties=in_properties) for item in obj]
			return obj

		# Create optimized schema with flattening
		optimized_result = optimize_schema(original_schema, defs_lookup)

		# Ensure we have a dictionary (should always be the case for schema root)
		if not isinstance(optimized_result, dict):
			raise ValueError('Optimized schema result is not a dictionary')

		optimized_schema: dict[str, Any] = optimized_result

		# Additional pass to ensure ALL objects have additionalProperties: false
		def ensure_additional_properties_false(obj: Any) -> None:
			"""Ensure all objects have additionalProperties: false"""
			if isinstance(obj, dict):
				# If it's an object type, ensure additionalProperties is false
				if obj.get('type') == 'object':
					obj['additionalProperties'] = False

				# Recursively apply to all values
				for value in obj.values():
					if isinstance(value, (dict, list)):
						ensure_additional_properties_false(value)
			elif isinstance(obj, list):
				for item in obj:
					if isinstance(item, (dict, list)):
						ensure_additional_properties_false(item)

		ensure_additional_properties_false(optimized_schema)
		SchemaOptimizer._make_strict_compatible(optimized_schema)

		return optimized_schema

	@staticmethod
	def _make_strict_compatible(schema: dict[str, Any] | list[Any]) -> None:
		"""Ensure all properties are required for OpenAI strict mode"""
		if isinstance(schema, dict):
			# First recursively apply to nested objects
			for key, value in schema.items():
				if isinstance(value, (dict, list)) and key != 'required':
					SchemaOptimizer._make_strict_compatible(value)

			# Then update required for this level
			if 'properties' in schema and 'type' in schema and schema['type'] == 'object':
				# Add all properties to required array
				all_props = list(schema['properties'].keys())
				schema['required'] = all_props  # Set all properties as required

		elif isinstance(schema, list):
			for item in schema:
				SchemaOptimizer._make_strict_compatible(item)

# From llm/schema.py
def create_optimized_json_schema(model: type[BaseModel]) -> dict[str, Any]:
		"""
		Create the most optimized schema by flattening all $ref/$defs while preserving
		FULL descriptions and ALL action definitions. Also ensures OpenAI strict mode compatibility.

		Args:
			model: The Pydantic model to optimize

		Returns:
			Optimized schema with all $refs resolved and strict mode compatibility
		"""
		# Generate original schema
		original_schema = model.model_json_schema()

		# Extract $defs for reference resolution, then flatten everything
		defs_lookup = original_schema.get('$defs', {})

		def optimize_schema(
			obj: Any,
			defs_lookup: dict[str, Any] | None = None,
			*,
			in_properties: bool = False,  # NEW: track context
		) -> Any:
			"""Apply all optimization techniques including flattening all $ref/$defs"""
			if isinstance(obj, dict):
				optimized: dict[str, Any] = {}
				flattened_ref: dict[str, Any] | None = None

				# Skip unnecessary fields AND $defs (we'll inline everything)
				skip_fields = ['additionalProperties', '$defs']

				for key, value in obj.items():
					if key in skip_fields:
						continue

					# Skip metadata "title" unless we're iterating inside an actual `properties` map
					if key == 'title' and not in_properties:
						continue

					# Preserve FULL descriptions without truncation
					elif key == 'description':
						optimized[key] = value

					# Handle type field
					elif key == 'type':
						optimized[key] = value

					# FLATTEN: Resolve $ref by inlining the actual definition
					elif key == '$ref' and defs_lookup:
						ref_path = value.split('/')[-1]  # Get the definition name from "#/$defs/SomeName"
						if ref_path in defs_lookup:
							# Get the referenced definition and flatten it
							referenced_def = defs_lookup[ref_path]
							flattened_ref = optimize_schema(referenced_def, defs_lookup)

					# Keep all anyOf structures (action unions) and resolve any $refs within
					elif key == 'anyOf' and isinstance(value, list):
						optimized[key] = [optimize_schema(item, defs_lookup) for item in value]

					# Recursively optimize nested structures
					elif key in ['properties', 'items']:
						optimized[key] = optimize_schema(
							value,
							defs_lookup,
							in_properties=(key == 'properties'),
						)

					# Keep essential validation fields
					elif key in ['type', 'required', 'minimum', 'maximum', 'minItems', 'maxItems', 'pattern', 'default']:
						optimized[key] = value if not isinstance(value, (dict, list)) else optimize_schema(value, defs_lookup)

					# Recursively process all other fields
					else:
						optimized[key] = optimize_schema(value, defs_lookup) if isinstance(value, (dict, list)) else value

				# If we have a flattened reference, merge it with the optimized properties
				if flattened_ref is not None and isinstance(flattened_ref, dict):
					# Start with the flattened reference as the base
					result = flattened_ref.copy()

					# Merge in any sibling properties that were processed
					for key, value in optimized.items():
						# Preserve descriptions from the original object if they exist
						if key == 'description' and 'description' not in result:
							result[key] = value
						elif key != 'description':  # Don't overwrite description from flattened ref
							result[key] = value

					return result
				else:
					# No $ref, just return the optimized object
					# CRITICAL: Add additionalProperties: false to ALL objects for OpenAI strict mode
					if optimized.get('type') == 'object':
						optimized['additionalProperties'] = False

					return optimized

			elif isinstance(obj, list):
				return [optimize_schema(item, defs_lookup, in_properties=in_properties) for item in obj]
			return obj

		# Create optimized schema with flattening
		optimized_result = optimize_schema(original_schema, defs_lookup)

		# Ensure we have a dictionary (should always be the case for schema root)
		if not isinstance(optimized_result, dict):
			raise ValueError('Optimized schema result is not a dictionary')

		optimized_schema: dict[str, Any] = optimized_result

		# Additional pass to ensure ALL objects have additionalProperties: false
		def ensure_additional_properties_false(obj: Any) -> None:
			"""Ensure all objects have additionalProperties: false"""
			if isinstance(obj, dict):
				# If it's an object type, ensure additionalProperties is false
				if obj.get('type') == 'object':
					obj['additionalProperties'] = False

				# Recursively apply to all values
				for value in obj.values():
					if isinstance(value, (dict, list)):
						ensure_additional_properties_false(value)
			elif isinstance(obj, list):
				for item in obj:
					if isinstance(item, (dict, list)):
						ensure_additional_properties_false(item)

		ensure_additional_properties_false(optimized_schema)
		SchemaOptimizer._make_strict_compatible(optimized_schema)

		return optimized_schema

# From llm/schema.py
def optimize_schema(
			obj: Any,
			defs_lookup: dict[str, Any] | None = None,
			*,
			in_properties: bool = False,  # NEW: track context
		) -> Any:
			"""Apply all optimization techniques including flattening all $ref/$defs"""
			if isinstance(obj, dict):
				optimized: dict[str, Any] = {}
				flattened_ref: dict[str, Any] | None = None

				# Skip unnecessary fields AND $defs (we'll inline everything)
				skip_fields = ['additionalProperties', '$defs']

				for key, value in obj.items():
					if key in skip_fields:
						continue

					# Skip metadata "title" unless we're iterating inside an actual `properties` map
					if key == 'title' and not in_properties:
						continue

					# Preserve FULL descriptions without truncation
					elif key == 'description':
						optimized[key] = value

					# Handle type field
					elif key == 'type':
						optimized[key] = value

					# FLATTEN: Resolve $ref by inlining the actual definition
					elif key == '$ref' and defs_lookup:
						ref_path = value.split('/')[-1]  # Get the definition name from "#/$defs/SomeName"
						if ref_path in defs_lookup:
							# Get the referenced definition and flatten it
							referenced_def = defs_lookup[ref_path]
							flattened_ref = optimize_schema(referenced_def, defs_lookup)

					# Keep all anyOf structures (action unions) and resolve any $refs within
					elif key == 'anyOf' and isinstance(value, list):
						optimized[key] = [optimize_schema(item, defs_lookup) for item in value]

					# Recursively optimize nested structures
					elif key in ['properties', 'items']:
						optimized[key] = optimize_schema(
							value,
							defs_lookup,
							in_properties=(key == 'properties'),
						)

					# Keep essential validation fields
					elif key in ['type', 'required', 'minimum', 'maximum', 'minItems', 'maxItems', 'pattern', 'default']:
						optimized[key] = value if not isinstance(value, (dict, list)) else optimize_schema(value, defs_lookup)

					# Recursively process all other fields
					else:
						optimized[key] = optimize_schema(value, defs_lookup) if isinstance(value, (dict, list)) else value

				# If we have a flattened reference, merge it with the optimized properties
				if flattened_ref is not None and isinstance(flattened_ref, dict):
					# Start with the flattened reference as the base
					result = flattened_ref.copy()

					# Merge in any sibling properties that were processed
					for key, value in optimized.items():
						# Preserve descriptions from the original object if they exist
						if key == 'description' and 'description' not in result:
							result[key] = value
						elif key != 'description':  # Don't overwrite description from flattened ref
							result[key] = value

					return result
				else:
					# No $ref, just return the optimized object
					# CRITICAL: Add additionalProperties: false to ALL objects for OpenAI strict mode
					if optimized.get('type') == 'object':
						optimized['additionalProperties'] = False

					return optimized

			elif isinstance(obj, list):
				return [optimize_schema(item, defs_lookup, in_properties=in_properties) for item in obj]
			return obj

# From llm/schema.py
def ensure_additional_properties_false(obj: Any) -> None:
			"""Ensure all objects have additionalProperties: false"""
			if isinstance(obj, dict):
				# If it's an object type, ensure additionalProperties is false
				if obj.get('type') == 'object':
					obj['additionalProperties'] = False

				# Recursively apply to all values
				for value in obj.values():
					if isinstance(value, (dict, list)):
						ensure_additional_properties_false(value)
			elif isinstance(obj, list):
				for item in obj:
					if isinstance(item, (dict, list)):
						ensure_additional_properties_false(item)


# From llm/views.py
class ChatInvokeUsage(BaseModel):
	"""
	Usage information for a chat model invocation.
	"""

	prompt_tokens: int
	"""The number of tokens in the prompt (this includes the cached tokens as well. When calculating the cost, subtract the cached tokens from the prompt tokens)"""

	prompt_cached_tokens: int | None
	"""The number of cached tokens."""

	prompt_cache_creation_tokens: int | None
	"""Anthropic only: The number of tokens used to create the cache."""

	prompt_image_tokens: int | None
	"""Google only: The number of tokens in the image (prompt tokens is the text tokens + image tokens in that case)"""

	completion_tokens: int
	"""The number of tokens in the completion."""

	total_tokens: int
	"""The total number of tokens in the response."""

# From llm/views.py
class ChatInvokeCompletion(BaseModel, Generic[T]):
	"""
	Response from a chat model invocation.
	"""

	completion: T
	"""The completion of the response."""

	# Thinking stuff
	thinking: str | None = None
	redacted_thinking: str | None = None

	usage: ChatInvokeUsage | None
	"""The usage of the response."""

import pyperclip
import tiktoken
from browser_use.agent.prompts import AgentMessagePrompt
from browser_use.browser.types import ViewportSize



# From history_tree_processor/view.py
class HashedDomElement:
	"""
	Hash of the dom element to be used as a unique identifier
	"""

	branch_path_hash: str
	attributes_hash: str
	xpath_hash: str

# From history_tree_processor/view.py
class Coordinates(BaseModel):
	x: int
	y: int

# From history_tree_processor/view.py
class CoordinateSet(BaseModel):
	top_left: Coordinates
	top_right: Coordinates
	bottom_left: Coordinates
	bottom_right: Coordinates
	center: Coordinates
	width: int
	height: int

# From history_tree_processor/view.py
class ViewportInfo(BaseModel):
	scroll_x: int | None = None
	scroll_y: int | None = None
	width: int
	height: int

# From history_tree_processor/view.py
class DOMHistoryElement:
	tag_name: str
	xpath: str
	highlight_index: int | None
	entire_parent_branch_path: list[str]
	attributes: dict[str, str]
	shadow_root: bool = False
	css_selector: str | None = None
	page_coordinates: CoordinateSet | None = None
	viewport_coordinates: CoordinateSet | None = None
	viewport_info: ViewportInfo | None = None

	def to_dict(self) -> dict:
		page_coordinates = self.page_coordinates.model_dump() if self.page_coordinates else None
		viewport_coordinates = self.viewport_coordinates.model_dump() if self.viewport_coordinates else None
		viewport_info = self.viewport_info.model_dump() if self.viewport_info else None

		return {
			'tag_name': self.tag_name,
			'xpath': self.xpath,
			'highlight_index': self.highlight_index,
			'entire_parent_branch_path': self.entire_parent_branch_path,
			'attributes': self.attributes,
			'shadow_root': self.shadow_root,
			'css_selector': self.css_selector,
			'page_coordinates': page_coordinates,
			'viewport_coordinates': viewport_coordinates,
			'viewport_info': viewport_info,
		}

import hashlib
from browser_use.dom.history_tree_processor.view import DOMHistoryElement
from browser_use.browser.context import BrowserContext

# From history_tree_processor/service.py
class HistoryTreeProcessor:
	""" "
	Operations on the DOM elements

	@dev be careful - text nodes can change even if elements stay the same
	"""

	@staticmethod
	def convert_dom_element_to_history_element(dom_element: DOMElementNode) -> DOMHistoryElement:
		from browser_use.browser.context import BrowserContext

		parent_branch_path = HistoryTreeProcessor._get_parent_branch_path(dom_element)
		css_selector = BrowserContext._enhanced_css_selector_for_element(dom_element)
		return DOMHistoryElement(
			dom_element.tag_name,
			dom_element.xpath,
			dom_element.highlight_index,
			parent_branch_path,
			dom_element.attributes,
			dom_element.shadow_root,
			css_selector=css_selector,
			page_coordinates=dom_element.page_coordinates,
			viewport_coordinates=dom_element.viewport_coordinates,
			viewport_info=dom_element.viewport_info,
		)

	@staticmethod
	def find_history_element_in_tree(dom_history_element: DOMHistoryElement, tree: DOMElementNode) -> DOMElementNode | None:
		hashed_dom_history_element = HistoryTreeProcessor._hash_dom_history_element(dom_history_element)

		def process_node(node: DOMElementNode):
			if node.highlight_index is not None:
				hashed_node = HistoryTreeProcessor._hash_dom_element(node)
				if hashed_node == hashed_dom_history_element:
					return node
			for child in node.children:
				if isinstance(child, DOMElementNode):
					result = process_node(child)
					if result is not None:
						return result
			return None

		return process_node(tree)

	@staticmethod
	def compare_history_element_and_dom_element(dom_history_element: DOMHistoryElement, dom_element: DOMElementNode) -> bool:
		hashed_dom_history_element = HistoryTreeProcessor._hash_dom_history_element(dom_history_element)
		hashed_dom_element = HistoryTreeProcessor._hash_dom_element(dom_element)

		return hashed_dom_history_element == hashed_dom_element

	@staticmethod
	def _hash_dom_history_element(dom_history_element: DOMHistoryElement) -> HashedDomElement:
		branch_path_hash = HistoryTreeProcessor._parent_branch_path_hash(dom_history_element.entire_parent_branch_path)
		attributes_hash = HistoryTreeProcessor._attributes_hash(dom_history_element.attributes)
		xpath_hash = HistoryTreeProcessor._xpath_hash(dom_history_element.xpath)

		return HashedDomElement(branch_path_hash, attributes_hash, xpath_hash)

	@staticmethod
	def _hash_dom_element(dom_element: DOMElementNode) -> HashedDomElement:
		parent_branch_path = HistoryTreeProcessor._get_parent_branch_path(dom_element)
		branch_path_hash = HistoryTreeProcessor._parent_branch_path_hash(parent_branch_path)
		attributes_hash = HistoryTreeProcessor._attributes_hash(dom_element.attributes)
		xpath_hash = HistoryTreeProcessor._xpath_hash(dom_element.xpath)
		# text_hash = DomTreeProcessor._text_hash(dom_element)

		return HashedDomElement(branch_path_hash, attributes_hash, xpath_hash)

	@staticmethod
	def _get_parent_branch_path(dom_element: DOMElementNode) -> list[str]:
		parents: list[DOMElementNode] = []
		current_element: DOMElementNode = dom_element
		while current_element.parent is not None:
			parents.append(current_element)
			current_element = current_element.parent

		parents.reverse()

		return [parent.tag_name for parent in parents]

	@staticmethod
	def _parent_branch_path_hash(parent_branch_path: list[str]) -> str:
		parent_branch_path_string = '/'.join(parent_branch_path)
		return hashlib.sha256(parent_branch_path_string.encode()).hexdigest()

	@staticmethod
	def _attributes_hash(attributes: dict[str, str]) -> str:
		attributes_string = ''.join(f'{key}={value}' for key, value in attributes.items())
		return hashlib.sha256(attributes_string.encode()).hexdigest()

	@staticmethod
	def _xpath_hash(xpath: str) -> str:
		return hashlib.sha256(xpath.encode()).hexdigest()

	@staticmethod
	def _text_hash(dom_element: DOMElementNode) -> str:
		""" """
		text_string = dom_element.get_all_text_till_next_clickable_element()
		return hashlib.sha256(text_string.encode()).hexdigest()

# From history_tree_processor/service.py
def convert_dom_element_to_history_element(dom_element: DOMElementNode) -> DOMHistoryElement:
		from browser_use.browser.context import BrowserContext

		parent_branch_path = HistoryTreeProcessor._get_parent_branch_path(dom_element)
		css_selector = BrowserContext._enhanced_css_selector_for_element(dom_element)
		return DOMHistoryElement(
			dom_element.tag_name,
			dom_element.xpath,
			dom_element.highlight_index,
			parent_branch_path,
			dom_element.attributes,
			dom_element.shadow_root,
			css_selector=css_selector,
			page_coordinates=dom_element.page_coordinates,
			viewport_coordinates=dom_element.viewport_coordinates,
			viewport_info=dom_element.viewport_info,
		)

# From history_tree_processor/service.py
def find_history_element_in_tree(dom_history_element: DOMHistoryElement, tree: DOMElementNode) -> DOMElementNode | None:
		hashed_dom_history_element = HistoryTreeProcessor._hash_dom_history_element(dom_history_element)

		def process_node(node: DOMElementNode):
			if node.highlight_index is not None:
				hashed_node = HistoryTreeProcessor._hash_dom_element(node)
				if hashed_node == hashed_dom_history_element:
					return node
			for child in node.children:
				if isinstance(child, DOMElementNode):
					result = process_node(child)
					if result is not None:
						return result
			return None

		return process_node(tree)

# From history_tree_processor/service.py
def compare_history_element_and_dom_element(dom_history_element: DOMHistoryElement, dom_element: DOMElementNode) -> bool:
		hashed_dom_history_element = HistoryTreeProcessor._hash_dom_history_element(dom_history_element)
		hashed_dom_element = HistoryTreeProcessor._hash_dom_element(dom_element)

		return hashed_dom_history_element == hashed_dom_element


# From clickable_element_processor/service.py
class ClickableElementProcessor:
	@staticmethod
	def get_clickable_elements_hashes(dom_element: DOMElementNode) -> set[str]:
		"""Get all clickable elements in the DOM tree"""
		clickable_elements = ClickableElementProcessor.get_clickable_elements(dom_element)
		return {ClickableElementProcessor.hash_dom_element(element) for element in clickable_elements}

	@staticmethod
	def get_clickable_elements(dom_element: DOMElementNode) -> list[DOMElementNode]:
		"""Get all clickable elements in the DOM tree"""
		clickable_elements = list()
		for child in dom_element.children:
			if isinstance(child, DOMElementNode):
				if child.highlight_index:
					clickable_elements.append(child)

				clickable_elements.extend(ClickableElementProcessor.get_clickable_elements(child))

		return list(clickable_elements)

	@staticmethod
	def hash_dom_element(dom_element: DOMElementNode) -> str:
		parent_branch_path = ClickableElementProcessor._get_parent_branch_path(dom_element)
		branch_path_hash = ClickableElementProcessor._parent_branch_path_hash(parent_branch_path)
		attributes_hash = ClickableElementProcessor._attributes_hash(dom_element.attributes)
		xpath_hash = ClickableElementProcessor._xpath_hash(dom_element.xpath)
		# text_hash = DomTreeProcessor._text_hash(dom_element)

		return ClickableElementProcessor._hash_string(f'{branch_path_hash}-{attributes_hash}-{xpath_hash}')

	@staticmethod
	def _get_parent_branch_path(dom_element: DOMElementNode) -> list[str]:
		parents: list[DOMElementNode] = []
		current_element: DOMElementNode = dom_element
		while current_element.parent is not None:
			parents.append(current_element)
			current_element = current_element.parent

		parents.reverse()

		return [parent.tag_name for parent in parents]

	@staticmethod
	def _parent_branch_path_hash(parent_branch_path: list[str]) -> str:
		parent_branch_path_string = '/'.join(parent_branch_path)
		return hashlib.sha256(parent_branch_path_string.encode()).hexdigest()

	@staticmethod
	def _attributes_hash(attributes: dict[str, str]) -> str:
		attributes_string = ''.join(f'{key}={value}' for key, value in attributes.items())
		return ClickableElementProcessor._hash_string(attributes_string)

	@staticmethod
	def _xpath_hash(xpath: str) -> str:
		return ClickableElementProcessor._hash_string(xpath)

	@staticmethod
	def _text_hash(dom_element: DOMElementNode) -> str:
		""" """
		text_string = dom_element.get_all_text_till_next_clickable_element()
		return ClickableElementProcessor._hash_string(text_string)

	@staticmethod
	def _hash_string(string: str) -> str:
		return hashlib.sha256(string.encode()).hexdigest()

# From clickable_element_processor/service.py
def get_clickable_elements_hashes(dom_element: DOMElementNode) -> set[str]:
		"""Get all clickable elements in the DOM tree"""
		clickable_elements = ClickableElementProcessor.get_clickable_elements(dom_element)
		return {ClickableElementProcessor.hash_dom_element(element) for element in clickable_elements}

# From clickable_element_processor/service.py
def get_clickable_elements(dom_element: DOMElementNode) -> list[DOMElementNode]:
		"""Get all clickable elements in the DOM tree"""
		clickable_elements = list()
		for child in dom_element.children:
			if isinstance(child, DOMElementNode):
				if child.highlight_index:
					clickable_elements.append(child)

				clickable_elements.extend(ClickableElementProcessor.get_clickable_elements(child))

		return list(clickable_elements)

# From clickable_element_processor/service.py
def hash_dom_element(dom_element: DOMElementNode) -> str:
		parent_branch_path = ClickableElementProcessor._get_parent_branch_path(dom_element)
		branch_path_hash = ClickableElementProcessor._parent_branch_path_hash(parent_branch_path)
		attributes_hash = ClickableElementProcessor._attributes_hash(dom_element.attributes)
		xpath_hash = ClickableElementProcessor._xpath_hash(dom_element.xpath)
		# text_hash = DomTreeProcessor._text_hash(dom_element)

		return ClickableElementProcessor._hash_string(f'{branch_path_hash}-{attributes_hash}-{xpath_hash}')

import functools
from inspect import Parameter
from inspect import iscoroutinefunction
from inspect import signature
from types import UnionType
from typing import get_args
from typing import get_origin
from pydantic import RootModel
from browser_use.controller.registry.views import ActionRegistry
from browser_use.controller.registry.views import RegisteredAction
from browser_use.controller.registry.views import SpecialActionParameters

# From registry/service.py
class Registry(Generic[Context]):
	"""Service for registering and managing actions"""

	def __init__(self, exclude_actions: list[str] | None = None):
		self.registry = ActionRegistry()
		self.telemetry = ProductTelemetry()
		self.exclude_actions = exclude_actions if exclude_actions is not None else []

	def _get_special_param_types(self) -> dict[str, type | UnionType | None]:
		"""Get the expected types for special parameters from SpecialActionParameters"""
		# Manually define the expected types to avoid issues with Optional handling.
		# we should try to reduce this list to 0 if possible, give as few standardized objects to all the actions
		# but each driver should decide what is relevant to expose the action methods,
		# e.g. playwright page, 2fa code getters, sensitive_data wrappers, other context, etc.
		return {
			'context': None,  # Context is a TypeVar, so we can't validate type
			'browser_session': BrowserSession,
			'browser': BrowserSession,  # legacy name
			'browser_context': BrowserSession,  # legacy name
			'page': Page,
			'page_extraction_llm': BaseChatModel,
			'available_file_paths': list,
			'has_sensitive_data': bool,
			'file_system': FileSystem,
		}

	def _normalize_action_function_signature(
		self,
		func: Callable,
		description: str,
		param_model: type[BaseModel] | None = None,
	) -> tuple[Callable, type[BaseModel]]:
		"""
		Normalize action function to accept only kwargs.

		Returns:
			- Normalized function that accepts (*_, params: ParamModel, **special_params)
			- The param model to use for registration
		"""
		sig = signature(func)
		parameters = list(sig.parameters.values())
		special_param_types = self._get_special_param_types()
		special_param_names = set(special_param_types.keys())

		# Step 1: Validate no **kwargs in original function signature
		# if it needs default values it must use a dedicated param_model: BaseModel instead
		for param in parameters:
			if param.kind == Parameter.VAR_KEYWORD:
				raise ValueError(
					f"Action '{func.__name__}' has **{param.name} which is not allowed. "
					f'Actions must have explicit positional parameters only.'
				)

		# Step 2: Separate special and action parameters
		action_params = []
		special_params = []
		param_model_provided = param_model is not None

		for i, param in enumerate(parameters):
			# Check if this is a Type 1 pattern (first param is BaseModel)
			if i == 0 and param_model_provided and param.name not in special_param_names:
				# This is Type 1 pattern - skip the params argument
				continue

			if param.name in special_param_names:
				# Validate special parameter type
				expected_type = special_param_types.get(param.name)
				if param.annotation != Parameter.empty and expected_type is not None:
					# Handle Optional types - normalize both sides
					param_type = param.annotation
					origin = get_origin(param_type)
					if origin is Union:
						args = get_args(param_type)
						# Find non-None type
						param_type = next((arg for arg in args if arg is not type(None)), param_type)

					# Check if types are compatible (exact match, subclass, or generic list)
					types_compatible = (
						param_type == expected_type
						or (
							inspect.isclass(param_type)
							and inspect.isclass(expected_type)
							and issubclass(param_type, expected_type)
						)
						or
						# Handle list[T] vs list comparison
						(expected_type is list and (param_type is list or get_origin(param_type) is list))
					)

					if not types_compatible:
						expected_type_name = getattr(expected_type, '__name__', str(expected_type))
						param_type_name = getattr(param_type, '__name__', str(param_type))
						raise ValueError(
							f"Action '{func.__name__}' parameter '{param.name}: {param_type_name}' "
							f"conflicts with special argument injected by controller: '{param.name}: {expected_type_name}'"
						)
				special_params.append(param)
			else:
				action_params.append(param)

		# Step 3: Create or validate param model
		if not param_model_provided:
			# Type 2: Generate param model from action params
			if action_params:
				params_dict = {}
				for param in action_params:
					annotation = param.annotation if param.annotation != Parameter.empty else str
					default = ... if param.default == Parameter.empty else param.default
					params_dict[param.name] = (annotation, default)

				param_model = create_model(f'{func.__name__}_Params', __base__=ActionModel, **params_dict)
			else:
				# No action params, create empty model
				param_model = create_model(
					f'{func.__name__}_Params',
					__base__=ActionModel,
				)
		assert param_model is not None, f'param_model is None for {func.__name__}'

		# Step 4: Create normalized wrapper function
		@functools.wraps(func)
		async def normalized_wrapper(*args, params: BaseModel | None = None, **kwargs):
			"""Normalized action that only accepts kwargs"""
			# Validate no positional args
			if args:
				raise TypeError(f'{func.__name__}() does not accept positional arguments, only keyword arguments are allowed')

			# Prepare arguments for original function
			call_args = []
			call_kwargs = {}

			# Handle Type 1 pattern (first arg is the param model)
			if param_model_provided and parameters and parameters[0].name not in special_param_names:
				if params is None:
					raise ValueError(f"{func.__name__}() missing required 'params' argument")
				# For Type 1, we'll use the params object as first argument
				pass
			else:
				# Type 2 pattern - need to unpack params
				# If params is None, try to create it from kwargs
				if params is None and action_params:
					# Extract action params from kwargs
					action_kwargs = {}
					for param in action_params:
						if param.name in kwargs:
							action_kwargs[param.name] = kwargs[param.name]
					if action_kwargs:
						# Use the param_model which has the correct types defined
						params = param_model(**action_kwargs)

			# Build call_args by iterating through original function parameters in order
			params_dict = params.model_dump() if params is not None else {}

			for i, param in enumerate(parameters):
				# Skip first param for Type 1 pattern (it's the model itself)
				if param_model_provided and i == 0 and param.name not in special_param_names:
					call_args.append(params)
				elif param.name in special_param_names:
					# This is a special parameter
					if param.name in kwargs:
						value = kwargs[param.name]
						# Check if required special param is None
						if value is None and param.default == Parameter.empty:
							if param.name == 'browser_session':
								raise ValueError(f'Action {func.__name__} requires browser_session but none provided.')
							elif param.name == 'page_extraction_llm':
								raise ValueError(f'Action {func.__name__} requires page_extraction_llm but none provided.')
							elif param.name == 'file_system':
								raise ValueError(f'Action {func.__name__} requires file_system but none provided.')
							elif param.name == 'page':
								raise ValueError(f'Action {func.__name__} requires page but none provided.')
							elif param.name == 'available_file_paths':
								raise ValueError(f'Action {func.__name__} requires available_file_paths but none provided.')
							elif param.name == 'file_system':
								raise ValueError(f'Action {func.__name__} requires file_system but none provided.')
							else:
								raise ValueError(f"{func.__name__}() missing required special parameter '{param.name}'")
						call_args.append(value)
					elif param.default != Parameter.empty:
						call_args.append(param.default)
					else:
						# Special param is required but not provided
						if param.name == 'browser_session':
							raise ValueError(f'Action {func.__name__} requires browser_session but none provided.')
						elif param.name == 'page_extraction_llm':
							raise ValueError(f'Action {func.__name__} requires page_extraction_llm but none provided.')
						elif param.name == 'file_system':
							raise ValueError(f'Action {func.__name__} requires file_system but none provided.')
						elif param.name == 'page':
							raise ValueError(f'Action {func.__name__} requires page but none provided.')
						elif param.name == 'available_file_paths':
							raise ValueError(f'Action {func.__name__} requires available_file_paths but none provided.')
						elif param.name == 'file_system':
							raise ValueError(f'Action {func.__name__} requires file_system but none provided.')
						else:
							raise ValueError(f"{func.__name__}() missing required special parameter '{param.name}'")
				else:
					# This is an action parameter
					if param.name in params_dict:
						call_args.append(params_dict[param.name])
					elif param.default != Parameter.empty:
						call_args.append(param.default)
					else:
						raise ValueError(f"{func.__name__}() missing required parameter '{param.name}'")

			# Call original function with positional args
			if iscoroutinefunction(func):
				return await func(*call_args)
			else:
				return await asyncio.to_thread(func, *call_args)

		# Update wrapper signature to be kwargs-only
		new_params = [Parameter('params', Parameter.KEYWORD_ONLY, default=None, annotation=Optional[param_model])]

		# Add special params as keyword-only
		for sp in special_params:
			new_params.append(Parameter(sp.name, Parameter.KEYWORD_ONLY, default=sp.default, annotation=sp.annotation))

		# Add **kwargs to accept and ignore extra params
		new_params.append(Parameter('kwargs', Parameter.VAR_KEYWORD))

		normalized_wrapper.__signature__ = sig.replace(parameters=new_params)  # type: ignore[attr-defined]

		return normalized_wrapper, param_model

	# @time_execution_sync('--create_param_model')
	def _create_param_model(self, function: Callable) -> type[BaseModel]:
		"""Creates a Pydantic model from function signature"""
		sig = signature(function)
		special_param_names = set(SpecialActionParameters.model_fields.keys())
		params = {
			name: (param.annotation, ... if param.default == param.empty else param.default)
			for name, param in sig.parameters.items()
			if name not in special_param_names
		}
		# TODO: make the types here work
		return create_model(
			f'{function.__name__}_parameters',
			__base__=ActionModel,
			**params,  # type: ignore
		)

	def action(
		self,
		description: str,
		param_model: type[BaseModel] | None = None,
		domains: list[str] | None = None,
		allowed_domains: list[str] | None = None,
		page_filter: Callable[[Any], bool] | None = None,
	):
		"""Decorator for registering actions"""
		# Handle aliases: domains and allowed_domains are the same parameter
		if allowed_domains is not None and domains is not None:
			raise ValueError("Cannot specify both 'domains' and 'allowed_domains' - they are aliases for the same parameter")

		final_domains = allowed_domains if allowed_domains is not None else domains

		def decorator(func: Callable):
			# Skip registration if action is in exclude_actions
			if func.__name__ in self.exclude_actions:
				return func

			# Normalize the function signature
			normalized_func, actual_param_model = self._normalize_action_function_signature(func, description, param_model)

			action = RegisteredAction(
				name=func.__name__,
				description=description,
				function=normalized_func,
				param_model=actual_param_model,
				domains=final_domains,
				page_filter=page_filter,
			)
			self.registry.actions[func.__name__] = action

			# Return the normalized function so it can be called with kwargs
			return normalized_func

		return decorator

	@observe_debug(ignore_input=True, ignore_output=True, name='execute_action')
	@time_execution_async('--execute_action')
	async def execute_action(
		self,
		action_name: str,
		params: dict,
		browser_session: BrowserSession | None = None,
		page_extraction_llm: BaseChatModel | None = None,
		file_system: FileSystem | None = None,
		sensitive_data: dict[str, str | dict[str, str]] | None = None,
		available_file_paths: list[str] | None = None,
		#
		context: Context | None = None,
	) -> Any:
		"""Execute a registered action with simplified parameter handling"""
		if action_name not in self.registry.actions:
			raise ValueError(f'Action {action_name} not found')

		action = self.registry.actions[action_name]
		try:
			# Create the validated Pydantic model
			try:
				validated_params = action.param_model(**params)
			except Exception as e:
				raise ValueError(f'Invalid parameters {params} for action {action_name}: {type(e)}: {e}') from e

			if sensitive_data:
				# Get current URL if browser_session is provided
				current_url = None
				if browser_session:
					if browser_session.agent_current_page:
						current_url = browser_session.agent_current_page.url
					else:
						current_page = await browser_session.get_current_page()
						current_url = current_page.url if current_page else None
				validated_params = self._replace_sensitive_data(validated_params, sensitive_data, current_url)

			# Build special context dict
			special_context = {
				'context': context,
				'browser_session': browser_session,
				'browser': browser_session,  # legacy support
				'browser_context': browser_session,  # legacy support
				'page_extraction_llm': page_extraction_llm,
				'available_file_paths': available_file_paths,
				'has_sensitive_data': action_name == 'input_text' and bool(sensitive_data),
				'file_system': file_system,
			}

			# Handle async page parameter if needed
			if browser_session:
				# Check if function signature includes 'page' parameter
				sig = signature(action.function)
				if 'page' in sig.parameters:
					special_context['page'] = await browser_session.get_current_page()

			# All functions are now normalized to accept kwargs only
			# Call with params and unpacked special context
			try:
				return await action.function(params=validated_params, **special_context)
			except Exception as e:
				# Retry once if it's a page error
				# logger.warning(f'⚠️ Action {action_name}() failed: {type(e).__name__}: {e}, trying one more time...')
				# special_context['page'] = browser_session and await browser_session.get_current_page()
				# try:
				# 	return await action.function(params=validated_params, **special_context)
				# except Exception as retry_error:
				# 	raise RuntimeError(
				# 		f'Action {action_name}() failed: {type(e).__name__}: {e} (page may have closed or navigated away mid-action)'
				# 	) from retry_error
				raise

		except ValueError as e:
			# Preserve ValueError messages from validation
			if 'requires browser_session but none provided' in str(e) or 'requires page_extraction_llm but none provided' in str(
				e
			):
				raise RuntimeError(str(e)) from e
			else:
				raise RuntimeError(f'Error executing action {action_name}: {str(e)}') from e
		except Exception as e:
			raise RuntimeError(f'Error executing action {action_name}: {str(e)}') from e

	def _log_sensitive_data_usage(self, placeholders_used: set[str], current_url: str | None) -> None:
		"""Log when sensitive data is being used on a page"""
		if placeholders_used:
			url_info = f' on {current_url}' if current_url and not is_new_tab_page(current_url) else ''
			logger.info(f'🔒 Using sensitive data placeholders: {", ".join(sorted(placeholders_used))}{url_info}')

	def _replace_sensitive_data(
		self, params: BaseModel, sensitive_data: dict[str, Any], current_url: str | None = None
	) -> BaseModel:
		"""
		Replaces sensitive data placeholders in params with actual values.

		Args:
			params: The parameter object containing <secret>placeholder</secret> tags
			sensitive_data: Dictionary of sensitive data, either in old format {key: value}
						   or new format {domain_pattern: {key: value}}
			current_url: Optional current URL for domain matching

		Returns:
			BaseModel: The parameter object with placeholders replaced by actual values
		"""
		secret_pattern = re.compile(r'<secret>(.*?)</secret>')

		# Set to track all missing placeholders across the full object
		all_missing_placeholders = set()
		# Set to track successfully replaced placeholders
		replaced_placeholders = set()

		# Process sensitive data based on format and current URL
		applicable_secrets = {}

		for domain_or_key, content in sensitive_data.items():
			if isinstance(content, dict):
				# New format: {domain_pattern: {key: value}}
				# Only include secrets for domains that match the current URL
				if current_url and not is_new_tab_page(current_url):
					# it's a real url, check it using our custom allowed_domains scheme://*.example.com glob matching
					if match_url_with_domain_pattern(current_url, domain_or_key):
						applicable_secrets.update(content)
			else:
				# Old format: {key: value}, expose to all domains (only allowed for legacy reasons)
				applicable_secrets[domain_or_key] = content

		# Filter out empty values
		applicable_secrets = {k: v for k, v in applicable_secrets.items() if v}

		def recursively_replace_secrets(value: str | dict | list) -> str | dict | list:
			if isinstance(value, str):
				matches = secret_pattern.findall(value)

				for placeholder in matches:
					if placeholder in applicable_secrets:
						value = value.replace(f'<secret>{placeholder}</secret>', applicable_secrets[placeholder])
						replaced_placeholders.add(placeholder)
					else:
						# Keep track of missing placeholders
						all_missing_placeholders.add(placeholder)
						# Don't replace the tag, keep it as is

				return value
			elif isinstance(value, dict):
				return {k: recursively_replace_secrets(v) for k, v in value.items()}
			elif isinstance(value, list):
				return [recursively_replace_secrets(v) for v in value]
			return value

		params_dump = params.model_dump()
		processed_params = recursively_replace_secrets(params_dump)

		# Log sensitive data usage
		self._log_sensitive_data_usage(replaced_placeholders, current_url)

		# Log a warning if any placeholders are missing
		if all_missing_placeholders:
			logger.warning(f'Missing or empty keys in sensitive_data dictionary: {", ".join(all_missing_placeholders)}')

		return type(params).model_validate(processed_params)

	# @time_execution_sync('--create_action_model')
	def create_action_model(self, include_actions: list[str] | None = None, page=None) -> type[ActionModel]:
		"""Creates a Union of individual action models from registered actions,
		used by LLM APIs that support tool calling & enforce a schema.

		Each action model contains only the specific action being used,
		rather than all actions with most set to None.
		"""
		from typing import Union

		# Filter actions based on page if provided:
		#   if page is None, only include actions with no filters
		#   if page is provided, only include actions that match the page

		available_actions: dict[str, RegisteredAction] = {}
		for name, action in self.registry.actions.items():
			if include_actions is not None and name not in include_actions:
				continue

			# If no page provided, only include actions with no filters
			if page is None:
				if action.page_filter is None and action.domains is None:
					available_actions[name] = action
				continue

			# Check page_filter if present
			domain_is_allowed = self.registry._match_domains(action.domains, page.url)
			page_is_allowed = self.registry._match_page_filter(action.page_filter, page)

			# Include action if both filters match (or if either is not present)
			if domain_is_allowed and page_is_allowed:
				available_actions[name] = action

		# Create individual action models for each action
		individual_action_models: list[type[BaseModel]] = []

		for name, action in available_actions.items():
			# Create an individual model for each action that contains only one field
			individual_model = create_model(
				f'{name.title().replace("_", "")}ActionModel',
				__base__=ActionModel,
				**{
					name: (
						action.param_model,
						Field(description=action.description),
					)  # type: ignore
				},
			)
			individual_action_models.append(individual_model)

		# If no actions available, return empty ActionModel
		if not individual_action_models:
			return create_model('EmptyActionModel', __base__=ActionModel)

		# Create proper Union type that maintains ActionModel interface
		if len(individual_action_models) == 1:
			# If only one action, return it directly (no Union needed)
			result_model = individual_action_models[0]

		# Meaning the length is more than 1
		else:
			# Create a Union type using RootModel that properly delegates ActionModel methods
			union_type = Union[tuple(individual_action_models)]  # type: ignore : Typing doesn't understand that the length is >= 2 (by design)

			class ActionModelUnion(RootModel[union_type]):  # type: ignore
				"""Union of all available action models that maintains ActionModel interface"""

				def get_index(self) -> int | None:
					"""Delegate get_index to the underlying action model"""
					if hasattr(self.root, 'get_index'):
						return self.root.get_index()  # type: ignore
					return None

				def set_index(self, index: int):
					"""Delegate set_index to the underlying action model"""
					if hasattr(self.root, 'set_index'):
						self.root.set_index(index)  # type: ignore

				def model_dump(self, **kwargs):
					"""Delegate model_dump to the underlying action model"""
					if hasattr(self.root, 'model_dump'):
						return self.root.model_dump(**kwargs)  # type: ignore
					return super().model_dump(**kwargs)

			# Set the name for better debugging
			ActionModelUnion.__name__ = 'ActionModel'
			ActionModelUnion.__qualname__ = 'ActionModel'

			result_model = ActionModelUnion

		return result_model  # type:ignore

	def get_prompt_description(self, page=None) -> str:
		"""Get a description of all actions for the prompt

		If page is provided, only include actions that are available for that page
		based on their filter_func
		"""
		return self.registry.get_prompt_description(page=page)

# From registry/service.py
class ActionModelUnion(RootModel[union_type]):  # type: ignore
				"""Union of all available action models that maintains ActionModel interface"""

				def get_index(self) -> int | None:
					"""Delegate get_index to the underlying action model"""
					if hasattr(self.root, 'get_index'):
						return self.root.get_index()  # type: ignore
					return None

				def set_index(self, index: int):
					"""Delegate set_index to the underlying action model"""
					if hasattr(self.root, 'set_index'):
						self.root.set_index(index)  # type: ignore

				def model_dump(self, **kwargs):
					"""Delegate model_dump to the underlying action model"""
					if hasattr(self.root, 'model_dump'):
						return self.root.model_dump(**kwargs)  # type: ignore
					return super().model_dump(**kwargs)

# From registry/service.py
def create_action_model(self, include_actions: list[str] | None = None, page=None) -> type[ActionModel]:
		"""Creates a Union of individual action models from registered actions,
		used by LLM APIs that support tool calling & enforce a schema.

		Each action model contains only the specific action being used,
		rather than all actions with most set to None.
		"""
		from typing import Union

		# Filter actions based on page if provided:
		#   if page is None, only include actions with no filters
		#   if page is provided, only include actions that match the page

		available_actions: dict[str, RegisteredAction] = {}
		for name, action in self.registry.actions.items():
			if include_actions is not None and name not in include_actions:
				continue

			# If no page provided, only include actions with no filters
			if page is None:
				if action.page_filter is None and action.domains is None:
					available_actions[name] = action
				continue

			# Check page_filter if present
			domain_is_allowed = self.registry._match_domains(action.domains, page.url)
			page_is_allowed = self.registry._match_page_filter(action.page_filter, page)

			# Include action if both filters match (or if either is not present)
			if domain_is_allowed and page_is_allowed:
				available_actions[name] = action

		# Create individual action models for each action
		individual_action_models: list[type[BaseModel]] = []

		for name, action in available_actions.items():
			# Create an individual model for each action that contains only one field
			individual_model = create_model(
				f'{name.title().replace("_", "")}ActionModel',
				__base__=ActionModel,
				**{
					name: (
						action.param_model,
						Field(description=action.description),
					)  # type: ignore
				},
			)
			individual_action_models.append(individual_model)

		# If no actions available, return empty ActionModel
		if not individual_action_models:
			return create_model('EmptyActionModel', __base__=ActionModel)

		# Create proper Union type that maintains ActionModel interface
		if len(individual_action_models) == 1:
			# If only one action, return it directly (no Union needed)
			result_model = individual_action_models[0]

		# Meaning the length is more than 1
		else:
			# Create a Union type using RootModel that properly delegates ActionModel methods
			union_type = Union[tuple(individual_action_models)]  # type: ignore : Typing doesn't understand that the length is >= 2 (by design)

			class ActionModelUnion(RootModel[union_type]):  # type: ignore
				"""Union of all available action models that maintains ActionModel interface"""

				def get_index(self) -> int | None:
					"""Delegate get_index to the underlying action model"""
					if hasattr(self.root, 'get_index'):
						return self.root.get_index()  # type: ignore
					return None

				def set_index(self, index: int):
					"""Delegate set_index to the underlying action model"""
					if hasattr(self.root, 'set_index'):
						self.root.set_index(index)  # type: ignore

				def model_dump(self, **kwargs):
					"""Delegate model_dump to the underlying action model"""
					if hasattr(self.root, 'model_dump'):
						return self.root.model_dump(**kwargs)  # type: ignore
					return super().model_dump(**kwargs)

			# Set the name for better debugging
			ActionModelUnion.__name__ = 'ActionModel'
			ActionModelUnion.__qualname__ = 'ActionModel'

			result_model = ActionModelUnion

		return result_model

# From registry/service.py
def get_prompt_description(self, page=None) -> str:
		"""Get a description of all actions for the prompt

		If page is provided, only include actions that are available for that page
		based on their filter_func
		"""
		return self.registry.get_prompt_description(page=page)

# From registry/service.py
def recursively_replace_secrets(value: str | dict | list) -> str | dict | list:
			if isinstance(value, str):
				matches = secret_pattern.findall(value)

				for placeholder in matches:
					if placeholder in applicable_secrets:
						value = value.replace(f'<secret>{placeholder}</secret>', applicable_secrets[placeholder])
						replaced_placeholders.add(placeholder)
					else:
						# Keep track of missing placeholders
						all_missing_placeholders.add(placeholder)
						# Don't replace the tag, keep it as is

				return value
			elif isinstance(value, dict):
				return {k: recursively_replace_secrets(v) for k, v in value.items()}
			elif isinstance(value, list):
				return [recursively_replace_secrets(v) for v in value]
			return value

# From registry/service.py
def get_index(self) -> int | None:
					"""Delegate get_index to the underlying action model"""
					if hasattr(self.root, 'get_index'):
						return self.root.get_index()  # type: ignore
					return None

# From registry/service.py
def set_index(self, index: int):
					"""Delegate set_index to the underlying action model"""
					if hasattr(self.root, 'set_index'):
						self.root.set_index(index)


# From registry/views.py
class RegisteredAction(BaseModel):
	"""Model for a registered action"""

	name: str
	description: str
	function: Callable
	param_model: type[BaseModel]

	# filters: provide specific domains or a function to determine whether the action should be available on the given page or not
	domains: list[str] | None = None  # e.g. ['*.google.com', 'www.bing.com', 'yahoo.*]
	page_filter: Callable[[Page], bool] | None = None

	model_config = ConfigDict(arbitrary_types_allowed=True)

	def prompt_description(self) -> str:
		"""Get a description of the action for the prompt"""
		skip_keys = ['title']
		s = f'{self.description}: \n'
		s += '{' + str(self.name) + ': '
		s += str(
			{
				k: {sub_k: sub_v for sub_k, sub_v in v.items() if sub_k not in skip_keys}
				for k, v in self.param_model.model_json_schema()['properties'].items()
			}
		)
		s += '}'
		return s

# From registry/views.py
class ActionModel(BaseModel):
	"""Base model for dynamically created action models"""

	# this will have all the registered actions, e.g.
	# click_element = param_model = ClickElementParams
	# done = param_model = None
	#
	model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

	def get_index(self) -> int | None:
		"""Get the index of the action"""
		# {'clicked_element': {'index':5}}
		params = self.model_dump(exclude_unset=True).values()
		if not params:
			return None
		for param in params:
			if param is not None and 'index' in param:
				return param['index']
		return None

	def set_index(self, index: int):
		"""Overwrite the index of the action"""
		# Get the action name and params
		action_data = self.model_dump(exclude_unset=True)
		action_name = next(iter(action_data.keys()))
		action_params = getattr(self, action_name)

		# Update the index directly on the model
		if hasattr(action_params, 'index'):
			action_params.index = index

# From registry/views.py
class ActionRegistry(BaseModel):
	"""Model representing the action registry"""

	actions: dict[str, RegisteredAction] = {}

	@staticmethod
	def _match_domains(domains: list[str] | None, url: str) -> bool:
		"""
		Match a list of domain glob patterns against a URL.

		Args:
			domains: A list of domain patterns that can include glob patterns (* wildcard)
			url: The URL to match against

		Returns:
			True if the URL's domain matches the pattern, False otherwise
		"""

		if domains is None or not url:
			return True

		# Use the centralized URL matching logic from utils
		from browser_use.utils import match_url_with_domain_pattern

		for domain_pattern in domains:
			if match_url_with_domain_pattern(url, domain_pattern):
				return True
		return False

	@staticmethod
	def _match_page_filter(page_filter: Callable[[Page], bool] | None, page: Page) -> bool:
		"""Match a page filter against a page"""
		if page_filter is None:
			return True
		return page_filter(page)

	def get_prompt_description(self, page: Page | None = None) -> str:
		"""Get a description of all actions for the prompt

		Args:
			page: If provided, filter actions by page using page_filter and domains.

		Returns:
			A string description of available actions.
			- If page is None: return only actions with no page_filter and no domains (for system prompt)
			- If page is provided: return only filtered actions that match the current page (excluding unfiltered actions)
		"""
		if page is None:
			# For system prompt (no page provided), include only actions with no filters
			return '\n'.join(
				action.prompt_description()
				for action in self.actions.values()
				if action.page_filter is None and action.domains is None
			)

		# only include filtered actions for the current page
		filtered_actions = []
		for action in self.actions.values():
			if not (action.domains or action.page_filter):
				# skip actions with no filters, they are already included in the system prompt
				continue

			domain_is_allowed = self._match_domains(action.domains, page.url)
			page_is_allowed = self._match_page_filter(action.page_filter, page)

			if domain_is_allowed and page_is_allowed:
				filtered_actions.append(action)

		return '\n'.join(action.prompt_description() for action in filtered_actions)

# From registry/views.py
class SpecialActionParameters(BaseModel):
	"""Model defining all special parameters that can be injected into actions"""

	model_config = ConfigDict(arbitrary_types_allowed=True)

	# optional user-provided context object passed down from Agent(context=...)
	# e.g. can contain anything, external db connections, file handles, queues, runtime config objects, etc.
	# that you might want to be able to access quickly from within many of your actions
	# browser-use code doesn't use this at all, we just pass it down to your actions for convenience
	context: Any | None = None

	# browser-use session object, can be used to create new tabs, navigate, access playwright objects, etc.
	browser_session: BrowserSession | None = None

	# legacy support for actions that ask for the old model names
	browser: BrowserSession | None = None
	browser_context: BrowserSession | None = (
		None  # extra confusing, this is actually not referring to a playwright BrowserContext,
		# but rather the name for BrowserUse's own old BrowserContext object from <v0.2.0
		# should be deprecated then removed after v0.3.0 to avoid ambiguity
	)  # we can't change it too fast because many people's custom actions out in the wild expect this argument

	# actions can get the playwright Page, shortcut for page = await browser_session.get_current_page()
	page: Page | None = None

	# extra injected config if the action asks for these arg names
	page_extraction_llm: BaseChatModel | None = None
	file_system: FileSystem | None = None
	available_file_paths: list[str] | None = None
	has_sensitive_data: bool = False

	@classmethod
	def get_browser_requiring_params(cls) -> set[str]:
		"""Get parameter names that require browser_session"""
		return {'browser_session', 'browser', 'browser_context', 'page'}

# From registry/views.py
def prompt_description(self) -> str:
		"""Get a description of the action for the prompt"""
		skip_keys = ['title']
		s = f'{self.description}: \n'
		s += '{' + str(self.name) + ': '
		s += str(
			{
				k: {sub_k: sub_v for sub_k, sub_v in v.items() if sub_k not in skip_keys}
				for k, v in self.param_model.model_json_schema()['properties'].items()
			}
		)
		s += '}'
		return s

# From registry/views.py
def get_browser_requiring_params(cls) -> set[str]:
		"""Get parameter names that require browser_session"""
		return {'browser_session', 'browser', 'browser_context', 'page'}


from browser_use.agent.message_manager.views import HistoryItem
from browser_use.agent.views import MessageManagerState

# From message_manager/service.py
class MessageManager:
	vision_detail_level: Literal['auto', 'low', 'high']

	def __init__(
		self,
		task: str,
		system_message: SystemMessage,
		file_system: FileSystem,
		state: MessageManagerState = MessageManagerState(),
		use_thinking: bool = True,
		include_attributes: list[str] | None = None,
		sensitive_data: dict[str, str | dict[str, str]] | None = None,
		max_history_items: int | None = None,
		vision_detail_level: Literal['auto', 'low', 'high'] = 'auto',
		include_tool_call_examples: bool = False,
	):
		self.task = task
		self.state = state
		self.system_prompt = system_message
		self.file_system = file_system
		self.sensitive_data_description = ''
		self.use_thinking = use_thinking
		self.max_history_items = max_history_items
		self.vision_detail_level = vision_detail_level
		self.include_tool_call_examples = include_tool_call_examples

		assert max_history_items is None or max_history_items > 5, 'max_history_items must be None or greater than 5'

		# Store settings as direct attributes instead of in a settings object
		self.include_attributes = include_attributes or []
		self.sensitive_data = sensitive_data
		self.last_input_messages = []
		# Only initialize messages if state is empty
		if len(self.state.history.get_messages()) == 0:
			self._set_message_with_type(self.system_prompt, 'system')

	@property
	def agent_history_description(self) -> str:
		"""Build agent history description from list of items, respecting max_history_items limit"""
		if self.max_history_items is None:
			# Include all items
			return '\n'.join(item.to_string() for item in self.state.agent_history_items)

		total_items = len(self.state.agent_history_items)

		# If we have fewer items than the limit, just return all items
		if total_items <= self.max_history_items:
			return '\n'.join(item.to_string() for item in self.state.agent_history_items)

		# We have more items than the limit, so we need to omit some
		omitted_count = total_items - self.max_history_items

		# Show first item + omitted message + most recent (max_history_items - 1) items
		# The omitted message doesn't count against the limit, only real history items do
		recent_items_count = self.max_history_items - 1  # -1 for first item

		items_to_include = [
			self.state.agent_history_items[0].to_string(),  # Keep first item (initialization)
			f'<sys>[... {omitted_count} previous steps omitted...]</sys>',
		]
		# Add most recent items
		items_to_include.extend([item.to_string() for item in self.state.agent_history_items[-recent_items_count:]])

		return '\n'.join(items_to_include)

	def add_new_task(self, new_task: str) -> None:
		self.task = new_task
		task_update_item = HistoryItem(system_message=f'User updated <user_request> to: {new_task}')
		self.state.agent_history_items.append(task_update_item)

	def _update_agent_history_description(
		self,
		model_output: AgentOutput | None = None,
		result: list[ActionResult] | None = None,
		step_info: AgentStepInfo | None = None,
	) -> None:
		"""Update the agent history description"""

		if result is None:
			result = []
		step_number = step_info.step_number if step_info else None

		self.state.read_state_description = ''

		action_results = ''
		result_len = len(result)
		for idx, action_result in enumerate(result):
			if action_result.include_extracted_content_only_once and action_result.extracted_content:
				self.state.read_state_description += action_result.extracted_content + '\n'
				logger.debug(f'Added extracted_content to read_state_description: {action_result.extracted_content}')

			if action_result.long_term_memory:
				action_results += f'Action {idx + 1}/{result_len}: {action_result.long_term_memory}\n'
				logger.debug(f'Added long_term_memory to action_results: {action_result.long_term_memory}')
			elif action_result.extracted_content and not action_result.include_extracted_content_only_once:
				action_results += f'Action {idx + 1}/{result_len}: {action_result.extracted_content}\n'
				logger.debug(f'Added extracted_content to action_results: {action_result.extracted_content}')

			if action_result.error:
				if len(action_result.error) > 200:
					error_text = action_result.error[:100] + '......' + action_result.error[-100:]
				else:
					error_text = action_result.error
				action_results += f'Action {idx + 1}/{result_len}: {error_text}\n'
				logger.debug(f'Added error to action_results: {error_text}')

		if action_results:
			action_results = f'Action Results:\n{action_results}'
		action_results = action_results.strip('\n') if action_results else None

		# Build the history item
		if model_output is None:
			# Only add error history item if we have a valid step number
			if step_number is not None and step_number > 0:
				history_item = HistoryItem(step_number=step_number, error='Agent failed to output in the right format.')
				self.state.agent_history_items.append(history_item)
		else:
			history_item = HistoryItem(
				step_number=step_number,
				evaluation_previous_goal=model_output.current_state.evaluation_previous_goal,
				memory=model_output.current_state.memory,
				next_goal=model_output.current_state.next_goal,
				action_results=action_results,
			)
			self.state.agent_history_items.append(history_item)

	def _get_sensitive_data_description(self, current_page_url) -> str:
		sensitive_data = self.sensitive_data
		if not sensitive_data:
			return ''

		# Collect placeholders for sensitive data
		placeholders: set[str] = set()

		for key, value in sensitive_data.items():
			if isinstance(value, dict):
				# New format: {domain: {key: value}}
				if match_url_with_domain_pattern(current_page_url, key, True):
					placeholders.update(value.keys())
			else:
				# Old format: {key: value}
				placeholders.add(key)

		if placeholders:
			placeholder_list = sorted(list(placeholders))
			info = f'Here are placeholders for sensitive data:\n{placeholder_list}\n'
			info += 'To use them, write <secret>the placeholder name</secret>'
			return info

		return ''

	@observe_debug(ignore_input=True, ignore_output=True, name='create_state_messages')
	@time_execution_sync('--create_state_messages')
	def create_state_messages(
		self,
		browser_state_summary: BrowserStateSummary,
		model_output: AgentOutput | None = None,
		result: list[ActionResult] | None = None,
		step_info: AgentStepInfo | None = None,
		use_vision=True,
		page_filtered_actions: str | None = None,
		sensitive_data=None,
		available_file_paths: list[str] | None = None,  # Always pass current available_file_paths
	) -> None:
		"""Create single state message with all content"""

		# Clear contextual messages from previous steps to prevent accumulation
		self.state.history.context_messages.clear()

		# First, update the agent history items with the latest step results
		self._update_agent_history_description(model_output, result, step_info)
		if sensitive_data:
			self.sensitive_data_description = self._get_sensitive_data_description(browser_state_summary.url)

		# Use only the current screenshot
		screenshots = []
		if browser_state_summary.screenshot:
			screenshots.append(browser_state_summary.screenshot)

		# Create single state message with all content
		assert browser_state_summary
		state_message = AgentMessagePrompt(
			browser_state_summary=browser_state_summary,
			file_system=self.file_system,
			agent_history_description=self.agent_history_description,
			read_state_description=self.state.read_state_description,
			task=self.task,
			include_attributes=self.include_attributes,
			step_info=step_info,
			page_filtered_actions=page_filtered_actions,
			sensitive_data=self.sensitive_data_description,
			available_file_paths=available_file_paths,
			screenshots=screenshots,
			vision_detail_level=self.vision_detail_level,
		).get_user_message(use_vision)

		# Set the state message with caching enabled
		self._set_message_with_type(state_message, 'state')

	def _log_history_lines(self) -> str:
		"""Generate a formatted log string of message history for debugging / printing to terminal"""
		# TODO: fix logging

		# try:
		# 	total_input_tokens = 0
		# 	message_lines = []
		# 	terminal_width = shutil.get_terminal_size((80, 20)).columns

		# 	for i, m in enumerate(self.state.history.messages):
		# 		try:
		# 			total_input_tokens += m.metadata.tokens
		# 			is_last_message = i == len(self.state.history.messages) - 1

		# 			# Extract content for logging
		# 			content = _log_extract_message_content(m.message, is_last_message, m.metadata)

		# 			# Format the message line(s)
		# 			lines = _log_format_message_line(m, content, is_last_message, terminal_width)
		# 			message_lines.extend(lines)
		# 		except Exception as e:
		# 			logger.warning(f'Failed to format message {i} for logging: {e}')
		# 			# Add a fallback line for this message
		# 			message_lines.append('❓[   ?]: [Error formatting this message]')

		# 	# Build final log message
		# 	return (
		# 		f'📜 LLM Message history ({len(self.state.history.messages)} messages, {total_input_tokens} tokens):\n'
		# 		+ '\n'.join(message_lines)
		# 	)
		# except Exception as e:
		# 	logger.warning(f'Failed to generate history log: {e}')
		# 	# Return a minimal fallback message
		# 	return f'📜 LLM Message history (error generating log: {e})'

		return ''

	@time_execution_sync('--get_messages')
	def get_messages(self) -> list[BaseMessage]:
		"""Get current message list, potentially trimmed to max tokens"""

		# Log message history for debugging
		logger.debug(self._log_history_lines())
		self.last_input_messages = self.state.history.get_messages()
		return self.last_input_messages

	def _set_message_with_type(self, message: BaseMessage, message_type: Literal['system', 'state']) -> None:
		"""Replace a specific state message slot with a new message"""
		# filter out sensitive data from the message
		if self.sensitive_data:
			message = self._filter_sensitive_data(message)

		if message_type == 'system':
			self.state.history.system_message = message
		elif message_type == 'state':
			self.state.history.state_message = message
		else:
			raise ValueError(f'Invalid state message type: {message_type}')

	def _add_context_message(self, message: BaseMessage) -> None:
		"""Add a contextual message specific to this step (e.g., validation errors, retry instructions, timeout warnings)"""
		# filter out sensitive data from the message
		if self.sensitive_data:
			message = self._filter_sensitive_data(message)

		self.state.history.context_messages.append(message)

	@time_execution_sync('--filter_sensitive_data')
	def _filter_sensitive_data(self, message: BaseMessage) -> BaseMessage:
		"""Filter out sensitive data from the message"""

		def replace_sensitive(value: str) -> str:
			if not self.sensitive_data:
				return value

			# Collect all sensitive values, immediately converting old format to new format
			sensitive_values: dict[str, str] = {}

			# Process all sensitive data entries
			for key_or_domain, content in self.sensitive_data.items():
				if isinstance(content, dict):
					# Already in new format: {domain: {key: value}}
					for key, val in content.items():
						if val:  # Skip empty values
							sensitive_values[key] = val
				elif content:  # Old format: {key: value} - convert to new format internally
					# We treat this as if it was {'http*://*': {key_or_domain: content}}
					sensitive_values[key_or_domain] = content

			# If there are no valid sensitive data entries, just return the original value
			if not sensitive_values:
				logger.warning('No valid entries found in sensitive_data dictionary')
				return value

			# Replace all valid sensitive data values with their placeholder tags
			for key, val in sensitive_values.items():
				value = value.replace(val, f'<secret>{key}</secret>')

			return value

		if isinstance(message.content, str):
			message.content = replace_sensitive(message.content)
		elif isinstance(message.content, list):
			for i, item in enumerate(message.content):
				if isinstance(item, ContentPartTextParam):
					item.text = replace_sensitive(item.text)
					message.content[i] = item
		return message

# From message_manager/service.py
def agent_history_description(self) -> str:
		"""Build agent history description from list of items, respecting max_history_items limit"""
		if self.max_history_items is None:
			# Include all items
			return '\n'.join(item.to_string() for item in self.state.agent_history_items)

		total_items = len(self.state.agent_history_items)

		# If we have fewer items than the limit, just return all items
		if total_items <= self.max_history_items:
			return '\n'.join(item.to_string() for item in self.state.agent_history_items)

		# We have more items than the limit, so we need to omit some
		omitted_count = total_items - self.max_history_items

		# Show first item + omitted message + most recent (max_history_items - 1) items
		# The omitted message doesn't count against the limit, only real history items do
		recent_items_count = self.max_history_items - 1  # -1 for first item

		items_to_include = [
			self.state.agent_history_items[0].to_string(),  # Keep first item (initialization)
			f'<sys>[... {omitted_count} previous steps omitted...]</sys>',
		]
		# Add most recent items
		items_to_include.extend([item.to_string() for item in self.state.agent_history_items[-recent_items_count:]])

		return '\n'.join(items_to_include)

# From message_manager/service.py
def create_state_messages(
		self,
		browser_state_summary: BrowserStateSummary,
		model_output: AgentOutput | None = None,
		result: list[ActionResult] | None = None,
		step_info: AgentStepInfo | None = None,
		use_vision=True,
		page_filtered_actions: str | None = None,
		sensitive_data=None,
		available_file_paths: list[str] | None = None,  # Always pass current available_file_paths
	) -> None:
		"""Create single state message with all content"""

		# Clear contextual messages from previous steps to prevent accumulation
		self.state.history.context_messages.clear()

		# First, update the agent history items with the latest step results
		self._update_agent_history_description(model_output, result, step_info)
		if sensitive_data:
			self.sensitive_data_description = self._get_sensitive_data_description(browser_state_summary.url)

		# Use only the current screenshot
		screenshots = []
		if browser_state_summary.screenshot:
			screenshots.append(browser_state_summary.screenshot)

		# Create single state message with all content
		assert browser_state_summary
		state_message = AgentMessagePrompt(
			browser_state_summary=browser_state_summary,
			file_system=self.file_system,
			agent_history_description=self.agent_history_description,
			read_state_description=self.state.read_state_description,
			task=self.task,
			include_attributes=self.include_attributes,
			step_info=step_info,
			page_filtered_actions=page_filtered_actions,
			sensitive_data=self.sensitive_data_description,
			available_file_paths=available_file_paths,
			screenshots=screenshots,
			vision_detail_level=self.vision_detail_level,
		).get_user_message(use_vision)

		# Set the state message with caching enabled
		self._set_message_with_type(state_message, 'state')

# From message_manager/service.py
def get_messages(self) -> list[BaseMessage]:
		"""Get current message list, potentially trimmed to max tokens"""

		# Log message history for debugging
		logger.debug(self._log_history_lines())
		self.last_input_messages = self.state.history.get_messages()
		return self.last_input_messages

# From message_manager/service.py
def replace_sensitive(value: str) -> str:
			if not self.sensitive_data:
				return value

			# Collect all sensitive values, immediately converting old format to new format
			sensitive_values: dict[str, str] = {}

			# Process all sensitive data entries
			for key_or_domain, content in self.sensitive_data.items():
				if isinstance(content, dict):
					# Already in new format: {domain: {key: value}}
					for key, val in content.items():
						if val:  # Skip empty values
							sensitive_values[key] = val
				elif content:  # Old format: {key: value} - convert to new format internally
					# We treat this as if it was {'http*://*': {key_or_domain: content}}
					sensitive_values[key_or_domain] = content

			# If there are no valid sensitive data entries, just return the original value
			if not sensitive_values:
				logger.warning('No valid entries found in sensitive_data dictionary')
				return value

			# Replace all valid sensitive data values with their placeholder tags
			for key, val in sensitive_values.items():
				value = value.replace(val, f'<secret>{key}</secret>')

			return value


# From message_manager/views.py
class HistoryItem(BaseModel):
	"""Represents a single agent history item with its data and string representation"""

	step_number: int | None = None
	evaluation_previous_goal: str | None = None
	memory: str | None = None
	next_goal: str | None = None
	action_results: str | None = None
	error: str | None = None
	system_message: str | None = None

	model_config = ConfigDict(arbitrary_types_allowed=True)

	def model_post_init(self, __context) -> None:
		"""Validate that error and system_message are not both provided"""
		if self.error is not None and self.system_message is not None:
			raise ValueError('Cannot have both error and system_message at the same time')

	def to_string(self) -> str:
		"""Get string representation of the history item"""
		step_str = f'step_{self.step_number}' if self.step_number is not None else 'step_unknown'

		if self.error:
			return f"""<{step_str}>
{self.error}
</{step_str}>"""
		elif self.system_message:
			return f"""<sys>
{self.system_message}
</sys>"""
		else:
			content_parts = []

			# Only include evaluation_previous_goal if it's not None/empty
			if self.evaluation_previous_goal:
				content_parts.append(f'Evaluation of Previous Step: {self.evaluation_previous_goal}')

			# Always include memory
			if self.memory:
				content_parts.append(f'Memory: {self.memory}')

			# Only include next_goal if it's not None/empty
			if self.next_goal:
				content_parts.append(f'Next Goal: {self.next_goal}')

			if self.action_results:
				content_parts.append(self.action_results)

			content = '\n'.join(content_parts)

			return f"""<{step_str}>
{content}
</{step_str}>"""

# From message_manager/views.py
class MessageHistory(BaseModel):
	"""History of messages"""

	system_message: BaseMessage | None = None
	state_message: BaseMessage | None = None
	context_messages: list[BaseMessage] = Field(default_factory=list)
	model_config = ConfigDict(arbitrary_types_allowed=True)

	def get_messages(self) -> list[BaseMessage]:
		"""Get all messages in the correct order: system -> state -> contextual"""
		messages = []
		if self.system_message:
			messages.append(self.system_message)
		if self.state_message:
			messages.append(self.state_message)
		messages.extend(self.context_messages)

		return messages

# From message_manager/views.py
class MessageManagerState(BaseModel):
	"""Holds the state for MessageManager"""

	history: MessageHistory = Field(default_factory=MessageHistory)
	tool_id: int = 1
	agent_history_items: list[HistoryItem] = Field(
		default_factory=lambda: [HistoryItem(step_number=0, system_message='Agent initialized')]
	)
	read_state_description: str = ''

	model_config = ConfigDict(arbitrary_types_allowed=True)

# From message_manager/views.py
def model_post_init(self, __context) -> None:
		"""Validate that error and system_message are not both provided"""
		if self.error is not None and self.system_message is not None:
			raise ValueError('Cannot have both error and system_message at the same time')

# From message_manager/views.py
def to_string(self) -> str:
		"""Get string representation of the history item"""
		step_str = f'step_{self.step_number}' if self.step_number is not None else 'step_unknown'

		if self.error:
			return f"""<{step_str}>
{self.error}
</{step_str}>"""
		elif self.system_message:
			return f"""<sys>
{self.system_message}
</sys>"""
		else:
			content_parts = []

			# Only include evaluation_previous_goal if it's not None/empty
			if self.evaluation_previous_goal:
				content_parts.append(f'Evaluation of Previous Step: {self.evaluation_previous_goal}')

			# Always include memory
			if self.memory:
				content_parts.append(f'Memory: {self.memory}')

			# Only include next_goal if it's not None/empty
			if self.next_goal:
				content_parts.append(f'Next Goal: {self.next_goal}')

			if self.action_results:
				content_parts.append(self.action_results)

			content = '\n'.join(content_parts)

			return f"""<{step_str}>
{content}
</{step_str}>"""

from service import GmailService

# From gmail/actions.py
class GetRecentEmailsParams(BaseModel):
	"""Parameters for getting recent emails"""

	keyword: str = Field(default='', description='A single keyword for search, e.g. github, airbnb, etc.')
	max_results: int = Field(default=3, ge=1, le=50, description='Maximum number of emails to retrieve (1-50, default: 3)')

# From gmail/actions.py
def register_gmail_actions(
	controller: Controller, gmail_service: GmailService | None = None, access_token: str | None = None
) -> Controller:
	"""
	Register Gmail actions with the provided controller
	Args:
	    controller: The browser-use controller to register actions with
	    gmail_service: Optional pre-configured Gmail service instance
	    access_token: Optional direct access token (alternative to file-based auth)
	"""
	global _gmail_service

	# Use provided service or create a new one with access token if provided
	if gmail_service:
		_gmail_service = gmail_service
	elif access_token:
		_gmail_service = GmailService(access_token=access_token)
	else:
		_gmail_service = GmailService()

	@controller.registry.action(
		description='Get recent emails from the mailbox with a keyword to retrieve verification codes, OTP, 2FA tokens, magic links, or any recent email content. Keep your query a single keyword.',
		param_model=GetRecentEmailsParams,
	)
	async def get_recent_emails(params: GetRecentEmailsParams) -> ActionResult:
		"""Get recent emails from the last 5 minutes with full content"""
		try:
			if _gmail_service is None:
				raise RuntimeError('Gmail service not initialized')

			# Ensure authentication
			if not _gmail_service.is_authenticated():
				logger.info('📧 Gmail not authenticated, attempting authentication...')
				authenticated = await _gmail_service.authenticate()
				if not authenticated:
					return ActionResult(
						extracted_content='Failed to authenticate with Gmail. Please ensure Gmail credentials are set up properly.',
						long_term_memory='Gmail authentication failed',
					)

			# Use specified max_results (1-50, default 10), last 5 minutes
			max_results = params.max_results
			time_filter = '5m'

			# Build query with time filter and optional user query
			query_parts = [f'newer_than:{time_filter}']
			if params.keyword.strip():
				query_parts.append(params.keyword.strip())

			query = ' '.join(query_parts)
			logger.info(f'🔍 Gmail search query: {query}')

			# Get emails
			emails = await _gmail_service.get_recent_emails(max_results=max_results, query=query, time_filter=time_filter)

			if not emails:
				query_info = f" matching '{params.keyword}'" if params.keyword.strip() else ''
				memory = f'No recent emails found from last {time_filter}{query_info}'
				return ActionResult(
					extracted_content=memory,
					long_term_memory=memory,
				)

			# Format with full email content for large display
			content = f'Found {len(emails)} recent email{"s" if len(emails) > 1 else ""} from the last {time_filter}:\n\n'

			for i, email in enumerate(emails, 1):
				content += f'Email {i}:\n'
				content += f'From: {email["from"]}\n'
				content += f'Subject: {email["subject"]}\n'
				content += f'Date: {email["date"]}\n'
				content += f'Content:\n{email["body"]}\n'
				content += '-' * 50 + '\n\n'

			logger.info(f'📧 Retrieved {len(emails)} recent emails')
			return ActionResult(
				extracted_content=content,
				include_extracted_content_only_once=True,
				long_term_memory=f'Retrieved {len(emails)} recent emails from last {time_filter} for query {query}.',
			)

		except Exception as e:
			logger.error(f'Error getting recent emails: {e}')
			return ActionResult(
				error=f'Error getting recent emails: {str(e)}',
				long_term_memory='Failed to get recent emails due to error',
			)

	return controller

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# From gmail/service.py
class GmailService:
	"""
	Gmail API service for email reading.
	Provides functionality to:
	- Authenticate with Gmail API using OAuth2
	- Read recent emails with filtering
	- Return full email content for agent analysis
	"""

	# Gmail API scopes
	SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

	def __init__(
		self,
		credentials_file: str | None = None,
		token_file: str | None = None,
		config_dir: str | None = None,
		access_token: str | None = None,
	):
		"""
		Initialize Gmail Service
		Args:
		    credentials_file: Path to OAuth credentials JSON from Google Cloud Console
		    token_file: Path to store/load access tokens
		    config_dir: Directory to store config files (defaults to browser-use config directory)
		    access_token: Direct access token (skips file-based auth if provided)
		"""
		# Set up configuration directory using browser-use's config system
		if config_dir is None:
			self.config_dir = CONFIG.BROWSER_USE_CONFIG_DIR
		else:
			self.config_dir = Path(config_dir).expanduser().resolve()

		# Ensure config directory exists (only if not using direct token)
		if access_token is None:
			self.config_dir.mkdir(parents=True, exist_ok=True)

		# Set up credential paths
		self.credentials_file = credentials_file or self.config_dir / 'gmail_credentials.json'
		self.token_file = token_file or self.config_dir / 'gmail_token.json'

		# Direct access token support
		self.access_token = access_token

		self.service = None
		self.creds = None
		self._authenticated = False

	def is_authenticated(self) -> bool:
		"""Check if Gmail service is authenticated"""
		return self._authenticated and self.service is not None

	async def authenticate(self) -> bool:
		"""
		Handle OAuth authentication and token management
		Returns:
		    bool: True if authentication successful, False otherwise
		"""
		try:
			logger.info('🔐 Authenticating with Gmail API...')

			# Check if using direct access token
			if self.access_token:
				logger.info('🔑 Using provided access token')
				# Create credentials from access token
				self.creds = Credentials(token=self.access_token, scopes=self.SCOPES)
				# Test token validity by building service
				self.service = build('gmail', 'v1', credentials=self.creds)
				self._authenticated = True
				logger.info('✅ Gmail API ready with access token!')
				return True

			# Original file-based authentication flow
			# Try to load existing tokens
			if os.path.exists(self.token_file):
				self.creds = Credentials.from_authorized_user_file(str(self.token_file), self.SCOPES)
				logger.debug('📁 Loaded existing tokens')

			# If no valid credentials, run OAuth flow
			if not self.creds or not self.creds.valid:
				if self.creds and self.creds.expired and self.creds.refresh_token:
					logger.info('🔄 Refreshing expired tokens...')
					self.creds.refresh(Request())
				else:
					logger.info('🌐 Starting OAuth flow...')
					if not os.path.exists(self.credentials_file):
						logger.error(
							f'❌ Gmail credentials file not found: {self.credentials_file}\n'
							'Please download it from Google Cloud Console:\n'
							'1. Go to https://console.cloud.google.com/\n'
							'2. APIs & Services > Credentials\n'
							'3. Download OAuth 2.0 Client JSON\n'
							f"4. Save as 'gmail_credentials.json' in {self.config_dir}/"
						)
						return False

					flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_file), self.SCOPES)
					# Use specific redirect URI to match OAuth credentials
					self.creds = flow.run_local_server(port=8080, open_browser=True)

				# Save tokens for next time
				async with aiofiles.open(self.token_file, 'w') as token:
					await token.write(self.creds.to_json())
				logger.info(f'💾 Tokens saved to {self.token_file}')

			# Build Gmail service
			self.service = build('gmail', 'v1', credentials=self.creds)
			self._authenticated = True
			logger.info('✅ Gmail API ready!')
			return True

		except Exception as e:
			logger.error(f'❌ Gmail authentication failed: {e}')
			return False

	async def get_recent_emails(self, max_results: int = 10, query: str = '', time_filter: str = '1h') -> list[dict[str, Any]]:
		"""
		Get recent emails with optional query filter
		Args:
		    max_results: Maximum number of emails to fetch
		    query: Gmail search query (e.g., 'from:noreply@example.com')
		    time_filter: Time filter (e.g., '5m', '1h', '1d')
		Returns:
		    List of email dictionaries with parsed content
		"""
		if not self.is_authenticated():
			logger.error('❌ Gmail service not authenticated. Call authenticate() first.')
			return []

		try:
			# Add time filter to query if provided
			if time_filter and 'newer_than:' not in query:
				query = f'newer_than:{time_filter} {query}'.strip()

			logger.info(f'📧 Fetching {max_results} recent emails...')
			if query:
				logger.debug(f'🔍 Query: {query}')

			# Get message list
			assert self.service is not None
			results = self.service.users().messages().list(userId='me', maxResults=max_results, q=query).execute()

			messages = results.get('messages', [])
			if not messages:
				logger.info('📭 No messages found')
				return []

			logger.info(f'📨 Found {len(messages)} messages, fetching details...')

			# Get full message details
			emails = []
			for i, message in enumerate(messages, 1):
				logger.debug(f'📖 Reading email {i}/{len(messages)}...')

				full_message = self.service.users().messages().get(userId='me', id=message['id'], format='full').execute()

				email_data = self._parse_email(full_message)
				emails.append(email_data)

			return emails

		except HttpError as error:
			logger.error(f'❌ Gmail API error: {error}')
			return []
		except Exception as e:
			logger.error(f'❌ Unexpected error fetching emails: {e}')
			return []

	def _parse_email(self, message: dict[str, Any]) -> dict[str, Any]:
		"""Parse Gmail message into readable format"""
		headers = {h['name']: h['value'] for h in message['payload']['headers']}

		return {
			'id': message['id'],
			'thread_id': message['threadId'],
			'subject': headers.get('Subject', ''),
			'from': headers.get('From', ''),
			'to': headers.get('To', ''),
			'date': headers.get('Date', ''),
			'timestamp': int(message['internalDate']),
			'body': self._extract_body(message['payload']),
			'raw_message': message,
		}

	def _extract_body(self, payload: dict[str, Any]) -> str:
		"""Extract email body from payload"""
		body = ''

		if payload.get('body', {}).get('data'):
			# Simple email body
			body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
		elif payload.get('parts'):
			# Multi-part email
			for part in payload['parts']:
				if part['mimeType'] == 'text/plain' and part.get('body', {}).get('data'):
					part_body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
					body += part_body
				elif part['mimeType'] == 'text/html' and not body and part.get('body', {}).get('data'):
					# Fallback to HTML if no plain text
					body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')

		return body

from collections.abc import Iterable
from collections.abc import Mapping
from openai import APIConnectionError
from openai import APIStatusError
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionContentPartTextParam
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.shared.chat_model import ChatModel
from openai.types.shared_params.reasoning_effort import ReasoningEffort
from openai.types.shared_params.response_format_json_schema import JSONSchema
from openai.types.shared_params.response_format_json_schema import ResponseFormatJSONSchema
from browser_use.llm.exceptions import ModelProviderError
from browser_use.llm.openai.serializer import OpenAIMessageSerializer
from browser_use.llm.schema import SchemaOptimizer

# From openai/chat.py
class ChatOpenAI(BaseChatModel):
	"""
	A wrapper around AsyncOpenAI that implements the BaseLLM protocol.

	This class accepts all AsyncOpenAI parameters while adding model
	and temperature parameters for the LLM interface (if temperature it not `None`).
	"""

	# Model configuration
	model: ChatModel | str

	# Model params
	# set to 0.1 because browser-use aims to be more reliable and deterministic
	temperature: float | None = 0.2
	frequency_penalty: float | None = 0.1
	reasoning_effort: ReasoningEffort = 'low'
	seed: int | None = None
	service_tier: Literal['auto', 'default', 'flex', 'priority', 'scale'] | None = None
	top_p: float | None = None
	add_schema_to_system_prompt: bool = False  # Add JSON schema to system prompt instead of using response_format

	# Client initialization parameters
	api_key: str | None = None
	organization: str | None = None
	project: str | None = None
	base_url: str | httpx.URL | None = None
	websocket_base_url: str | httpx.URL | None = None
	timeout: float | httpx.Timeout | None = None
	max_retries: int = 10  # Increase default retries for automation reliability
	default_headers: Mapping[str, str] | None = None
	default_query: Mapping[str, object] | None = None
	http_client: httpx.AsyncClient | None = None
	_strict_response_validation: bool = False
	max_completion_tokens: int | None = 8000

	# Static
	@property
	def provider(self) -> str:
		return 'openai'

	def _get_client_params(self) -> dict[str, Any]:
		"""Prepare client parameters dictionary."""
		# Define base client params
		base_params = {
			'api_key': self.api_key,
			'organization': self.organization,
			'project': self.project,
			'base_url': self.base_url,
			'websocket_base_url': self.websocket_base_url,
			'timeout': self.timeout,
			'max_retries': self.max_retries,
			'default_headers': self.default_headers,
			'default_query': self.default_query,
			'_strict_response_validation': self._strict_response_validation,
		}

		# Create client_params dict with non-None values
		client_params = {k: v for k, v in base_params.items() if v is not None}

		# Add http_client if provided
		if self.http_client is not None:
			client_params['http_client'] = self.http_client

		return client_params

	def get_client(self) -> AsyncOpenAI:
		"""
		Returns an AsyncOpenAI client.

		Returns:
			AsyncOpenAI: An instance of the AsyncOpenAI client.
		"""
		client_params = self._get_client_params()
		return AsyncOpenAI(**client_params)

	@property
	def name(self) -> str:
		return str(self.model)

	def _get_usage(self, response: ChatCompletion) -> ChatInvokeUsage | None:
		if response.usage is not None:
			completion_tokens = response.usage.completion_tokens
			completion_token_details = response.usage.completion_tokens_details
			if completion_token_details is not None:
				reasoning_tokens = completion_token_details.reasoning_tokens
				if reasoning_tokens is not None:
					completion_tokens += reasoning_tokens

			usage = ChatInvokeUsage(
				prompt_tokens=response.usage.prompt_tokens,
				prompt_cached_tokens=response.usage.prompt_tokens_details.cached_tokens
				if response.usage.prompt_tokens_details is not None
				else None,
				prompt_cache_creation_tokens=None,
				prompt_image_tokens=None,
				# Completion
				completion_tokens=completion_tokens,
				total_tokens=response.usage.total_tokens,
			)
		else:
			usage = None

		return usage

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		"""
		Invoke the model with the given messages.

		Args:
			messages: List of chat messages
			output_format: Optional Pydantic model class for structured output

		Returns:
			Either a string response or an instance of output_format
		"""

		openai_messages = OpenAIMessageSerializer.serialize_messages(messages)

		try:
			model_params: dict[str, Any] = {}

			if self.temperature is not None:
				model_params['temperature'] = self.temperature

			if self.frequency_penalty is not None:
				model_params['frequency_penalty'] = self.frequency_penalty

			if self.max_completion_tokens is not None:
				model_params['max_completion_tokens'] = self.max_completion_tokens

			if self.top_p is not None:
				model_params['top_p'] = self.top_p

			if self.seed is not None:
				model_params['seed'] = self.seed

			if self.service_tier is not None:
				model_params['service_tier'] = self.service_tier

			if any(str(m).lower() in str(self.model).lower() for m in ReasoningModels):
				model_params['reasoning_effort'] = self.reasoning_effort
				del model_params['temperature']
				del model_params['frequency_penalty']

			if output_format is None:
				# Return string response
				response = await self.get_client().chat.completions.create(
					model=self.model,
					messages=openai_messages,
					**model_params,
				)

				usage = self._get_usage(response)
				return ChatInvokeCompletion(
					completion=response.choices[0].message.content or '',
					usage=usage,
				)

			else:
				response_format: JSONSchema = {
					'name': 'agent_output',
					'strict': True,
					'schema': SchemaOptimizer.create_optimized_json_schema(output_format),
				}

				# Add JSON schema to system prompt if requested
				if self.add_schema_to_system_prompt and openai_messages and openai_messages[0]['role'] == 'system':
					schema_text = f'\n<json_schema>\n{response_format}\n</json_schema>'
					if isinstance(openai_messages[0]['content'], str):
						openai_messages[0]['content'] += schema_text
					elif isinstance(openai_messages[0]['content'], Iterable):
						openai_messages[0]['content'] = list(openai_messages[0]['content']) + [
							ChatCompletionContentPartTextParam(text=schema_text, type='text')
						]

				# Return structured response
				response = await self.get_client().chat.completions.create(
					model=self.model,
					messages=openai_messages,
					response_format=ResponseFormatJSONSchema(json_schema=response_format, type='json_schema'),
					**model_params,
				)

				if response.choices[0].message.content is None:
					raise ModelProviderError(
						message='Failed to parse structured output from model response',
						status_code=500,
						model=self.name,
					)

				usage = self._get_usage(response)

				parsed = output_format.model_validate_json(response.choices[0].message.content)

				return ChatInvokeCompletion(
					completion=parsed,
					usage=usage,
				)

		except RateLimitError as e:
			error_message = e.response.json().get('error', {})
			error_message = (
				error_message.get('message', 'Unknown model error') if isinstance(error_message, dict) else error_message
			)
			raise ModelProviderError(
				message=error_message,
				status_code=e.response.status_code,
				model=self.name,
			) from e

		except APIConnectionError as e:
			raise ModelProviderError(message=str(e), model=self.name) from e

		except APIStatusError as e:
			try:
				error_message = e.response.json().get('error', {})
			except Exception:
				error_message = e.response.text
			error_message = (
				error_message.get('message', 'Unknown model error') if isinstance(error_message, dict) else error_message
			)
			raise ModelProviderError(
				message=error_message,
				status_code=e.response.status_code,
				model=self.name,
			) from e

		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e

# From openai/chat.py
def get_client(self) -> AsyncOpenAI:
		"""
		Returns an AsyncOpenAI client.

		Returns:
			AsyncOpenAI: An instance of the AsyncOpenAI client.
		"""
		client_params = self._get_client_params()
		return AsyncOpenAI(**client_params)


# From openai/like.py
class ChatOpenAILike(ChatOpenAI):
	"""
	A class for to interact with any provider using the OpenAI API schema.

	Args:
	    model (str): The name of the OpenAI model to use.
	"""

	model: str

from openai.types.chat import ChatCompletionAssistantMessageParam
from openai.types.chat import ChatCompletionContentPartImageParam
from openai.types.chat import ChatCompletionContentPartRefusalParam
from openai.types.chat import ChatCompletionMessageFunctionToolCallParam
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import ChatCompletionSystemMessageParam
from openai.types.chat import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.chat.chat_completion_message_function_tool_call_param import Function
from browser_use.llm.messages import AssistantMessage
from browser_use.llm.messages import ContentPartRefusalParam
from browser_use.llm.messages import ToolCall

# From openai/serializer.py
class OpenAIMessageSerializer:
	"""Serializer for converting between custom message types and OpenAI message param types."""

	@staticmethod
	def _serialize_content_part_text(part: ContentPartTextParam) -> ChatCompletionContentPartTextParam:
		return ChatCompletionContentPartTextParam(text=part.text, type='text')

	@staticmethod
	def _serialize_content_part_image(part: ContentPartImageParam) -> ChatCompletionContentPartImageParam:
		return ChatCompletionContentPartImageParam(
			image_url=ImageURL(url=part.image_url.url, detail=part.image_url.detail),
			type='image_url',
		)

	@staticmethod
	def _serialize_content_part_refusal(part: ContentPartRefusalParam) -> ChatCompletionContentPartRefusalParam:
		return ChatCompletionContentPartRefusalParam(refusal=part.refusal, type='refusal')

	@staticmethod
	def _serialize_user_content(
		content: str | list[ContentPartTextParam | ContentPartImageParam],
	) -> str | list[ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam]:
		"""Serialize content for user messages (text and images allowed)."""
		if isinstance(content, str):
			return content

		serialized_parts: list[ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam] = []
		for part in content:
			if part.type == 'text':
				serialized_parts.append(OpenAIMessageSerializer._serialize_content_part_text(part))
			elif part.type == 'image_url':
				serialized_parts.append(OpenAIMessageSerializer._serialize_content_part_image(part))
		return serialized_parts

	@staticmethod
	def _serialize_system_content(
		content: str | list[ContentPartTextParam],
	) -> str | list[ChatCompletionContentPartTextParam]:
		"""Serialize content for system messages (text only)."""
		if isinstance(content, str):
			return content

		serialized_parts: list[ChatCompletionContentPartTextParam] = []
		for part in content:
			if part.type == 'text':
				serialized_parts.append(OpenAIMessageSerializer._serialize_content_part_text(part))
		return serialized_parts

	@staticmethod
	def _serialize_assistant_content(
		content: str | list[ContentPartTextParam | ContentPartRefusalParam] | None,
	) -> str | list[ChatCompletionContentPartTextParam | ChatCompletionContentPartRefusalParam] | None:
		"""Serialize content for assistant messages (text and refusal allowed)."""
		if content is None:
			return None
		if isinstance(content, str):
			return content

		serialized_parts: list[ChatCompletionContentPartTextParam | ChatCompletionContentPartRefusalParam] = []
		for part in content:
			if part.type == 'text':
				serialized_parts.append(OpenAIMessageSerializer._serialize_content_part_text(part))
			elif part.type == 'refusal':
				serialized_parts.append(OpenAIMessageSerializer._serialize_content_part_refusal(part))
		return serialized_parts

	@staticmethod
	def _serialize_tool_call(tool_call: ToolCall) -> ChatCompletionMessageFunctionToolCallParam:
		return ChatCompletionMessageFunctionToolCallParam(
			id=tool_call.id,
			function=Function(name=tool_call.function.name, arguments=tool_call.function.arguments),
			type='function',
		)

	# endregion

	# region - Serialize overloads
	@overload
	@staticmethod
	def serialize(message: UserMessage) -> ChatCompletionUserMessageParam: ...

	@overload
	@staticmethod
	def serialize(message: SystemMessage) -> ChatCompletionSystemMessageParam: ...

	@overload
	@staticmethod
	def serialize(message: AssistantMessage) -> ChatCompletionAssistantMessageParam: ...

	@staticmethod
	def serialize(message: BaseMessage) -> ChatCompletionMessageParam:
		"""Serialize a custom message to an OpenAI message param."""

		if isinstance(message, UserMessage):
			user_result: ChatCompletionUserMessageParam = {
				'role': 'user',
				'content': OpenAIMessageSerializer._serialize_user_content(message.content),
			}
			if message.name is not None:
				user_result['name'] = message.name
			return user_result

		elif isinstance(message, SystemMessage):
			system_result: ChatCompletionSystemMessageParam = {
				'role': 'system',
				'content': OpenAIMessageSerializer._serialize_system_content(message.content),
			}
			if message.name is not None:
				system_result['name'] = message.name
			return system_result

		elif isinstance(message, AssistantMessage):
			# Handle content serialization
			content = None
			if message.content is not None:
				content = OpenAIMessageSerializer._serialize_assistant_content(message.content)

			assistant_result: ChatCompletionAssistantMessageParam = {'role': 'assistant'}

			# Only add content if it's not None
			if content is not None:
				assistant_result['content'] = content

			if message.name is not None:
				assistant_result['name'] = message.name
			if message.refusal is not None:
				assistant_result['refusal'] = message.refusal
			if message.tool_calls:
				assistant_result['tool_calls'] = [OpenAIMessageSerializer._serialize_tool_call(tc) for tc in message.tool_calls]

			return assistant_result

		else:
			raise ValueError(f'Unknown message type: {type(message)}')

	@staticmethod
	def serialize_messages(messages: list[BaseMessage]) -> list[ChatCompletionMessageParam]:
		return [OpenAIMessageSerializer.serialize(m) for m in messages]

# From openai/serializer.py
def serialize(message: UserMessage) -> ChatCompletionUserMessageParam: ...

# From openai/serializer.py
def serialize_messages(messages: list[BaseMessage]) -> list[ChatCompletionMessageParam]:
		return [OpenAIMessageSerializer.serialize(m) for m in messages]

from openai import AsyncAzureOpenAI
from openai.types.shared import ChatModel
from browser_use.llm.openai.like import ChatOpenAILike

# From azure/chat.py
class ChatAzureOpenAI(ChatOpenAILike):
	"""
	A class for to interact with any provider using the OpenAI API schema.

	Args:
	    model (str): The name of the OpenAI model to use. Defaults to "not-provided".
	    api_key (Optional[str]): The API key to use. Defaults to "not-provided".
	"""

	# Model configuration
	model: str | ChatModel

	# Client initialization parameters
	api_key: str | None = None
	api_version: str | None = '2024-10-21'
	azure_endpoint: str | None = None
	azure_deployment: str | None = None
	base_url: str | None = None
	azure_ad_token: str | None = None
	azure_ad_token_provider: Any | None = None

	default_headers: dict[str, str] | None = None
	default_query: dict[str, Any] | None = None

	client: AsyncAzureOpenAIClient | None = None

	@property
	def provider(self) -> str:
		return 'azure'

	def _get_client_params(self) -> dict[str, Any]:
		_client_params: dict[str, Any] = {}

		self.api_key = self.api_key or os.getenv('AZURE_OPENAI_API_KEY')
		self.azure_endpoint = self.azure_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
		self.azure_deployment = self.azure_deployment or os.getenv('AZURE_OPENAI_DEPLOYMENT')
		params_mapping = {
			'api_key': self.api_key,
			'api_version': self.api_version,
			'organization': self.organization,
			'azure_endpoint': self.azure_endpoint,
			'azure_deployment': self.azure_deployment,
			'base_url': self.base_url,
			'azure_ad_token': self.azure_ad_token,
			'azure_ad_token_provider': self.azure_ad_token_provider,
			'http_client': self.http_client,
		}
		if self.default_headers is not None:
			_client_params['default_headers'] = self.default_headers
		if self.default_query is not None:
			_client_params['default_query'] = self.default_query

		_client_params.update({k: v for k, v in params_mapping.items() if v is not None})

		return _client_params

	def get_client(self) -> AsyncAzureOpenAIClient:
		"""
		Returns an asynchronous OpenAI client.

		Returns:
			AsyncAzureOpenAIClient: An instance of the asynchronous OpenAI client.
		"""
		if self.client:
			return self.client

		_client_params: dict[str, Any] = self._get_client_params()

		if self.http_client:
			_client_params['http_client'] = self.http_client
		else:
			# Create a new async HTTP client with custom limits
			_client_params['http_client'] = httpx.AsyncClient(
				limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
			)

		self.client = AsyncAzureOpenAIClient(**_client_params)

		return self.client

from openai import APIError
from openai import APITimeoutError
from browser_use.llm.deepseek.serializer import DeepSeekMessageSerializer
from browser_use.llm.exceptions import ModelRateLimitError

# From deepseek/chat.py
class ChatDeepSeek(BaseChatModel):
	"""DeepSeek /chat/completions 封装（OpenAI-compatible）。"""

	model: str = 'deepseek-chat'

	# 生成参数
	max_tokens: int | None = None
	temperature: float | None = None
	top_p: float | None = None
	seed: int | None = None

	# 连接参数
	api_key: str | None = None
	base_url: str | httpx.URL | None = 'https://api.deepseek.com/v1'
	timeout: float | httpx.Timeout | None = None
	client_params: dict[str, Any] | None = None

	@property
	def provider(self) -> str:
		return 'deepseek'

	def _client(self) -> AsyncOpenAI:
		return AsyncOpenAI(
			api_key=self.api_key,
			base_url=self.base_url,
			timeout=self.timeout,
			**(self.client_params or {}),
		)

	@property
	def name(self) -> str:
		return self.model

	@overload
	async def ainvoke(
		self,
		messages: list[BaseMessage],
		output_format: None = None,
		tools: list[dict[str, Any]] | None = None,
		stop: list[str] | None = None,
	) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(
		self,
		messages: list[BaseMessage],
		output_format: type[T],
		tools: list[dict[str, Any]] | None = None,
		stop: list[str] | None = None,
	) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self,
		messages: list[BaseMessage],
		output_format: type[T] | None = None,
		tools: list[dict[str, Any]] | None = None,
		stop: list[str] | None = None,
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		"""
		DeepSeek ainvoke 支持:
		1. 普通文本/多轮对话
		2. Function Calling
		3. JSON Output (response_format)
		4. 对话前缀续写 (beta, prefix, stop)
		"""
		client = self._client()
		ds_messages = DeepSeekMessageSerializer.serialize_messages(messages)
		common: dict[str, Any] = {}

		if self.temperature is not None:
			common['temperature'] = self.temperature
		if self.max_tokens is not None:
			common['max_tokens'] = self.max_tokens
		if self.top_p is not None:
			common['top_p'] = self.top_p
		if self.seed is not None:
			common['seed'] = self.seed

		# Beta 对话前缀续写（见官方文档）
		if self.base_url and str(self.base_url).endswith('/beta'):
			# 最后一个 assistant 必须 prefix
			if ds_messages and isinstance(ds_messages[-1], dict) and ds_messages[-1].get('role') == 'assistant':
				ds_messages[-1]['prefix'] = True
			if stop:
				common['stop'] = stop

		# ① 普通多轮对话/文本输出
		if output_format is None and not tools:
			try:
				resp = await client.chat.completions.create(  # type: ignore
					model=self.model,
					messages=ds_messages,  # type: ignore
					**common,
				)
				return ChatInvokeCompletion(
					completion=resp.choices[0].message.content or '',
					usage=None,
				)
			except RateLimitError as e:
				raise ModelRateLimitError(str(e), model=self.name) from e
			except (APIError, APIConnectionError, APITimeoutError, APIStatusError) as e:
				raise ModelProviderError(str(e), model=self.name) from e
			except Exception as e:
				raise ModelProviderError(str(e), model=self.name) from e

		# ② Function Calling 路径（有 tools 或 output_format）
		if tools or (output_format is not None and hasattr(output_format, 'model_json_schema')):
			try:
				call_tools = tools
				tool_choice = None
				if output_format is not None and hasattr(output_format, 'model_json_schema'):
					tool_name = output_format.__name__
					schema = SchemaOptimizer.create_optimized_json_schema(output_format)
					schema.pop('title', None)
					call_tools = [
						{
							'type': 'function',
							'function': {
								'name': tool_name,
								'description': f'Return a JSON object of type {tool_name}',
								'parameters': schema,
							},
						}
					]
					tool_choice = {'type': 'function', 'function': {'name': tool_name}}
				resp = await client.chat.completions.create(  # type: ignore
					model=self.model,
					messages=ds_messages,  # type: ignore
					tools=call_tools,  # type: ignore
					tool_choice=tool_choice,  # type: ignore
					**common,
				)
				msg = resp.choices[0].message
				if not msg.tool_calls:
					raise ValueError('Expected tool_calls in response but got none')
				raw_args = msg.tool_calls[0].function.arguments
				if isinstance(raw_args, str):
					parsed = json.loads(raw_args)
				else:
					parsed = raw_args
				# --------- 修复点: 只有 output_format 不为 None 才能用 model_validate ----------
				if output_format is not None:
					return ChatInvokeCompletion(
						completion=output_format.model_validate(parsed),
						usage=None,
					)
				else:
					# 若无 output_format，直接返回 dict
					return ChatInvokeCompletion(
						completion=parsed,
						usage=None,
					)
			except RateLimitError as e:
				raise ModelRateLimitError(str(e), model=self.name) from e
			except (APIError, APIConnectionError, APITimeoutError, APIStatusError) as e:
				raise ModelProviderError(str(e), model=self.name) from e
			except Exception as e:
				raise ModelProviderError(str(e), model=self.name) from e

		# ③ JSON Output 路径（官方 response_format）
		if output_format is not None and hasattr(output_format, 'model_json_schema'):
			try:
				resp = await client.chat.completions.create(  # type: ignore
					model=self.model,
					messages=ds_messages,  # type: ignore
					response_format={'type': 'json_object'},
					**common,
				)
				content = resp.choices[0].message.content
				if not content:
					raise ModelProviderError('Empty JSON content in DeepSeek response', model=self.name)
				parsed = output_format.model_validate_json(content)
				return ChatInvokeCompletion(
					completion=parsed,
					usage=None,
				)
			except RateLimitError as e:
				raise ModelRateLimitError(str(e), model=self.name) from e
			except (APIError, APIConnectionError, APITimeoutError, APIStatusError) as e:
				raise ModelProviderError(str(e), model=self.name) from e
			except Exception as e:
				raise ModelProviderError(str(e), model=self.name) from e

		# 所有路径兜底
		raise ModelProviderError('No valid ainvoke execution path for DeepSeek LLM', model=self.name)


# From deepseek/serializer.py
class DeepSeekMessageSerializer:
	"""Serializer for converting browser-use messages to DeepSeek messages."""

	# -------- content 处理 --------------------------------------------------
	@staticmethod
	def _serialize_text_part(part: ContentPartTextParam) -> str:
		return part.text

	@staticmethod
	def _serialize_image_part(part: ContentPartImageParam) -> dict[str, Any]:
		url = part.image_url.url
		if url.startswith('data:'):
			return {'type': 'image_url', 'image_url': {'url': url}}
		return {'type': 'image_url', 'image_url': {'url': url}}

	@staticmethod
	def _serialize_content(content: Any) -> str | list[dict[str, Any]]:
		if content is None:
			return ''
		if isinstance(content, str):
			return content
		serialized: list[dict[str, Any]] = []
		for part in content:
			if part.type == 'text':
				serialized.append({'type': 'text', 'text': DeepSeekMessageSerializer._serialize_text_part(part)})
			elif part.type == 'image_url':
				serialized.append(DeepSeekMessageSerializer._serialize_image_part(part))
			elif part.type == 'refusal':
				serialized.append({'type': 'text', 'text': f'[Refusal] {part.refusal}'})
		return serialized

	# -------- Tool-call 处理 -------------------------------------------------
	@staticmethod
	def _serialize_tool_calls(tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
		deepseek_tool_calls: list[dict[str, Any]] = []
		for tc in tool_calls:
			try:
				arguments = json.loads(tc.function.arguments)
			except json.JSONDecodeError:
				arguments = {'arguments': tc.function.arguments}
			deepseek_tool_calls.append(
				{
					'id': tc.id,
					'type': 'function',
					'function': {
						'name': tc.function.name,
						'arguments': arguments,
					},
				}
			)
		return deepseek_tool_calls

	# -------- 单条消息序列化 -------------------------------------------------
	@overload
	@staticmethod
	def serialize(message: UserMessage) -> MessageDict: ...

	@overload
	@staticmethod
	def serialize(message: SystemMessage) -> MessageDict: ...

	@overload
	@staticmethod
	def serialize(message: AssistantMessage) -> MessageDict: ...

	@staticmethod
	def serialize(message: BaseMessage) -> MessageDict:
		if isinstance(message, UserMessage):
			return {
				'role': 'user',
				'content': DeepSeekMessageSerializer._serialize_content(message.content),
			}
		if isinstance(message, SystemMessage):
			return {
				'role': 'system',
				'content': DeepSeekMessageSerializer._serialize_content(message.content),
			}
		if isinstance(message, AssistantMessage):
			msg: MessageDict = {
				'role': 'assistant',
				'content': DeepSeekMessageSerializer._serialize_content(message.content),
			}
			if message.tool_calls:
				msg['tool_calls'] = DeepSeekMessageSerializer._serialize_tool_calls(message.tool_calls)
			return msg
		raise ValueError(f'Unknown message type: {type(message)}')

	# -------- 列表序列化 -----------------------------------------------------
	@staticmethod
	def serialize_messages(messages: list[BaseMessage]) -> list[MessageDict]:
		return [DeepSeekMessageSerializer.serialize(m) for m in messages]

from google import genai
from google.auth.credentials import Credentials
from google.genai import types
from google.genai.types import MediaModality
from browser_use.llm.google.serializer import GoogleMessageSerializer

# From google/chat.py
class ChatGoogle(BaseChatModel):
	"""
	A wrapper around Google's Gemini chat model using the genai client.

	This class accepts all genai.Client parameters while adding model,
	temperature, and config parameters for the LLM interface.

	Args:
		model: The Gemini model to use
		temperature: Temperature for response generation
		config: Additional configuration parameters to pass to generate_content
			(e.g., tools, safety_settings, etc.).
		api_key: Google API key
		vertexai: Whether to use Vertex AI
		credentials: Google credentials object
		project: Google Cloud project ID
		location: Google Cloud location
		http_options: HTTP options for the client

	Example:
		from google.genai import types

		llm = ChatGoogle(
			model='gemini-2.0-flash-exp',
			config={
				'tools': [types.Tool(code_execution=types.ToolCodeExecution())]
			}
		)
	"""

	# Model configuration
	model: VerifiedGeminiModels | str
	temperature: float | None = None
	top_p: float | None = None
	seed: int | None = None
	thinking_budget: int | None = None
	config: types.GenerateContentConfigDict | None = None

	# Client initialization parameters
	api_key: str | None = None
	vertexai: bool | None = None
	credentials: Credentials | None = None
	project: str | None = None
	location: str | None = None
	http_options: types.HttpOptions | types.HttpOptionsDict | None = None

	# Static
	@property
	def provider(self) -> str:
		return 'google'

	def _get_client_params(self) -> dict[str, Any]:
		"""Prepare client parameters dictionary."""
		# Define base client params
		base_params = {
			'api_key': self.api_key,
			'vertexai': self.vertexai,
			'credentials': self.credentials,
			'project': self.project,
			'location': self.location,
			'http_options': self.http_options,
		}

		# Create client_params dict with non-None values
		client_params = {k: v for k, v in base_params.items() if v is not None}

		return client_params

	def get_client(self) -> genai.Client:
		"""
		Returns a genai.Client instance.

		Returns:
			genai.Client: An instance of the Google genai client.
		"""
		client_params = self._get_client_params()
		return genai.Client(**client_params)

	@property
	def name(self) -> str:
		return str(self.model)

	def _get_usage(self, response: types.GenerateContentResponse) -> ChatInvokeUsage | None:
		usage: ChatInvokeUsage | None = None

		if response.usage_metadata is not None:
			image_tokens = 0
			if response.usage_metadata.prompt_tokens_details is not None:
				image_tokens = sum(
					detail.token_count or 0
					for detail in response.usage_metadata.prompt_tokens_details
					if detail.modality == MediaModality.IMAGE
				)

			usage = ChatInvokeUsage(
				prompt_tokens=response.usage_metadata.prompt_token_count or 0,
				completion_tokens=(response.usage_metadata.candidates_token_count or 0)
				+ (response.usage_metadata.thoughts_token_count or 0),
				total_tokens=response.usage_metadata.total_token_count or 0,
				prompt_cached_tokens=response.usage_metadata.cached_content_token_count,
				prompt_cache_creation_tokens=None,
				prompt_image_tokens=image_tokens,
			)

		return usage

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		"""
		Invoke the model with the given messages.

		Args:
			messages: List of chat messages
			output_format: Optional Pydantic model class for structured output

		Returns:
			Either a string response or an instance of output_format
		"""

		# Serialize messages to Google format
		contents, system_instruction = GoogleMessageSerializer.serialize_messages(messages)

		# Build config dictionary starting with user-provided config
		config: types.GenerateContentConfigDict = {}
		if self.config:
			config = self.config.copy()

		# Apply model-specific configuration (these can override config)
		if self.temperature is not None:
			config['temperature'] = self.temperature

		# Add system instruction if present
		if system_instruction:
			config['system_instruction'] = system_instruction

		if self.top_p is not None:
			config['top_p'] = self.top_p

		if self.seed is not None:
			config['seed'] = self.seed

		if self.thinking_budget is not None:
			thinking_config_dict: types.ThinkingConfigDict = {'thinking_budget': self.thinking_budget}
			config['thinking_config'] = thinking_config_dict

		async def _make_api_call():
			if output_format is None:
				# Return string response
				response = await self.get_client().aio.models.generate_content(
					model=self.model,
					contents=contents,  # type: ignore
					config=config,
				)

				# Handle case where response.text might be None
				text = response.text or ''

				usage = self._get_usage(response)

				return ChatInvokeCompletion(
					completion=text,
					usage=usage,
				)

			else:
				# Return structured response
				config['response_mime_type'] = 'application/json'
				# Convert Pydantic model to Gemini-compatible schema
				optimized_schema = SchemaOptimizer.create_optimized_json_schema(output_format)

				gemini_schema = self._fix_gemini_schema(optimized_schema)
				config['response_schema'] = gemini_schema

				response = await self.get_client().aio.models.generate_content(
					model=self.model,
					contents=contents,
					config=config,
				)

				usage = self._get_usage(response)

				# Handle case where response.parsed might be None
				if response.parsed is None:
					# When using response_schema, Gemini returns JSON as text
					if response.text:
						try:
							# Parse the JSON text and validate with the Pydantic model
							parsed_data = json.loads(response.text)
							return ChatInvokeCompletion(
								completion=output_format.model_validate(parsed_data),
								usage=usage,
							)
						except (json.JSONDecodeError, ValueError) as e:
							raise ModelProviderError(
								message=f'Failed to parse or validate response: {str(e)}',
								status_code=500,
								model=self.model,
							) from e
					else:
						raise ModelProviderError(
							message='No response from model',
							status_code=500,
							model=self.model,
						)

				# Ensure we return the correct type
				if isinstance(response.parsed, output_format):
					return ChatInvokeCompletion(
						completion=response.parsed,
						usage=usage,
					)
				else:
					# If it's not the expected type, try to validate it
					return ChatInvokeCompletion(
						completion=output_format.model_validate(response.parsed),
						usage=usage,
					)

		try:
			# Use manual retry loop for Google API calls
			last_exception = None
			for attempt in range(10):  # Match our 10 retry attempts from other providers
				try:
					return await _make_api_call()
				except Exception as e:
					last_exception = e
					if not _is_retryable_error(e) or attempt == 9:  # Last attempt
						break

					# Simple exponential backoff
					import asyncio

					delay = min(60.0, 1.0 * (2.0**attempt))  # Cap at 60s
					await asyncio.sleep(delay)

			# Re-raise the last exception if all retries failed
			if last_exception:
				raise last_exception
			else:
				# This should never happen, but ensure we don't return None
				raise ModelProviderError(
					message='All retry attempts failed without exception',
					status_code=500,
					model=self.name,
				)

		except Exception as e:
			# Handle specific Google API errors
			error_message = str(e)
			status_code: int | None = None

			# Check if this is a rate limit error
			if any(
				indicator in error_message.lower()
				for indicator in ['rate limit', 'resource exhausted', 'quota exceeded', 'too many requests', '429']
			):
				status_code = 429
			elif any(
				indicator in error_message.lower()
				for indicator in ['service unavailable', 'internal server error', 'bad gateway', '503', '502', '500']
			):
				status_code = 503

			# Try to extract status code if available
			if hasattr(e, 'response'):
				response_obj = getattr(e, 'response', None)
				if response_obj and hasattr(response_obj, 'status_code'):
					status_code = getattr(response_obj, 'status_code', None)

			raise ModelProviderError(
				message=error_message,
				status_code=status_code or 502,  # Use default if None
				model=self.name,
			) from e

	def _fix_gemini_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
		"""
		Convert a Pydantic model to a Gemini-compatible schema.

		This function removes unsupported properties like 'additionalProperties' and resolves
		$ref references that Gemini doesn't support.
		"""

		# Handle $defs and $ref resolution
		if '$defs' in schema:
			defs = schema.pop('$defs')

			def resolve_refs(obj: Any) -> Any:
				if isinstance(obj, dict):
					if '$ref' in obj:
						ref = obj.pop('$ref')
						ref_name = ref.split('/')[-1]
						if ref_name in defs:
							# Replace the reference with the actual definition
							resolved = defs[ref_name].copy()
							# Merge any additional properties from the reference
							for key, value in obj.items():
								if key != '$ref':
									resolved[key] = value
							return resolve_refs(resolved)
						return obj
					else:
						# Recursively process all dictionary values
						return {k: resolve_refs(v) for k, v in obj.items()}
				elif isinstance(obj, list):
					return [resolve_refs(item) for item in obj]
				return obj

			schema = resolve_refs(schema)

		# Remove unsupported properties
		def clean_schema(obj: Any) -> Any:
			if isinstance(obj, dict):
				# Remove unsupported properties
				cleaned = {}
				for key, value in obj.items():
					if key not in ['additionalProperties', 'title', 'default']:
						cleaned_value = clean_schema(value)
						# Handle empty object properties - Gemini doesn't allow empty OBJECT types
						if (
							key == 'properties'
							and isinstance(cleaned_value, dict)
							and len(cleaned_value) == 0
							and isinstance(obj.get('type', ''), str)
							and obj.get('type', '').upper() == 'OBJECT'
						):
							# Convert empty object to have at least one property
							cleaned['properties'] = {'_placeholder': {'type': 'string'}}
						else:
							cleaned[key] = cleaned_value

				# If this is an object type with empty properties, add a placeholder
				if (
					isinstance(cleaned.get('type', ''), str)
					and cleaned.get('type', '').upper() == 'OBJECT'
					and 'properties' in cleaned
					and isinstance(cleaned['properties'], dict)
					and len(cleaned['properties']) == 0
				):
					cleaned['properties'] = {'_placeholder': {'type': 'string'}}

				return cleaned
			elif isinstance(obj, list):
				return [clean_schema(item) for item in obj]
			return obj

		return clean_schema(schema)

# From google/chat.py
def clean_schema(obj: Any) -> Any:
			if isinstance(obj, dict):
				# Remove unsupported properties
				cleaned = {}
				for key, value in obj.items():
					if key not in ['additionalProperties', 'title', 'default']:
						cleaned_value = clean_schema(value)
						# Handle empty object properties - Gemini doesn't allow empty OBJECT types
						if (
							key == 'properties'
							and isinstance(cleaned_value, dict)
							and len(cleaned_value) == 0
							and isinstance(obj.get('type', ''), str)
							and obj.get('type', '').upper() == 'OBJECT'
						):
							# Convert empty object to have at least one property
							cleaned['properties'] = {'_placeholder': {'type': 'string'}}
						else:
							cleaned[key] = cleaned_value

				# If this is an object type with empty properties, add a placeholder
				if (
					isinstance(cleaned.get('type', ''), str)
					and cleaned.get('type', '').upper() == 'OBJECT'
					and 'properties' in cleaned
					and isinstance(cleaned['properties'], dict)
					and len(cleaned['properties']) == 0
				):
					cleaned['properties'] = {'_placeholder': {'type': 'string'}}

				return cleaned
			elif isinstance(obj, list):
				return [clean_schema(item) for item in obj]
			return obj

# From google/chat.py
def resolve_refs(obj: Any) -> Any:
				if isinstance(obj, dict):
					if '$ref' in obj:
						ref = obj.pop('$ref')
						ref_name = ref.split('/')[-1]
						if ref_name in defs:
							# Replace the reference with the actual definition
							resolved = defs[ref_name].copy()
							# Merge any additional properties from the reference
							for key, value in obj.items():
								if key != '$ref':
									resolved[key] = value
							return resolve_refs(resolved)
						return obj
					else:
						# Recursively process all dictionary values
						return {k: resolve_refs(v) for k, v in obj.items()}
				elif isinstance(obj, list):
					return [resolve_refs(item) for item in obj]
				return obj

from google.genai.types import Content
from google.genai.types import ContentListUnion
from google.genai.types import Part

# From google/serializer.py
class GoogleMessageSerializer:
	"""Serializer for converting messages to Google Gemini format."""

	@staticmethod
	def serialize_messages(messages: list[BaseMessage]) -> tuple[ContentListUnion, str | None]:
		"""
		Convert a list of BaseMessages to Google format, extracting system message.

		Google handles system instructions separately from the conversation, so we need to:
		1. Extract any system messages and return them separately as a string
		2. Convert the remaining messages to Content objects

		Args:
		    messages: List of messages to convert

		Returns:
		    A tuple of (formatted_messages, system_message) where:
		    - formatted_messages: List of Content objects for the conversation
		    - system_message: System instruction string or None
		"""

		messages = [m.model_copy(deep=True) for m in messages]

		formatted_messages: ContentListUnion = []
		system_message: str | None = None

		for message in messages:
			role = message.role if hasattr(message, 'role') else None

			# Handle system/developer messages
			if isinstance(message, SystemMessage) or role in ['system', 'developer']:
				# Extract system message content as string
				if isinstance(message.content, str):
					system_message = message.content
				elif message.content is not None:
					# Handle Iterable of content parts
					parts = []
					for part in message.content:
						if part.type == 'text':
							parts.append(part.text)
					system_message = '\n'.join(parts)
				continue

			# Determine the role for non-system messages
			if isinstance(message, UserMessage):
				role = 'user'
			elif isinstance(message, AssistantMessage):
				role = 'model'
			else:
				# Default to user for any unknown message types
				role = 'user'

			# Initialize message parts
			message_parts: list[Part] = []

			# Extract content and create parts
			if isinstance(message.content, str):
				# Regular text content
				message_parts = [Part.from_text(text=message.content)]
			elif message.content is not None:
				# Handle Iterable of content parts
				for part in message.content:
					if part.type == 'text':
						message_parts.append(Part.from_text(text=part.text))
					elif part.type == 'refusal':
						message_parts.append(Part.from_text(text=f'[Refusal] {part.refusal}'))
					elif part.type == 'image_url':
						# Handle images
						url = part.image_url.url

						# Format: data:image/png;base64,<data>
						header, data = url.split(',', 1)
						# Decode base64 to bytes
						image_bytes = base64.b64decode(data)

						# Add image part
						image_part = Part.from_bytes(data=image_bytes, mime_type='image/png')

						message_parts.append(image_part)

			# Create the Content object
			if message_parts:
				final_message = Content(role=role, parts=message_parts)
				# for some reason, the type checker is not able to infer the type of formatted_messages
				formatted_messages.append(final_message)  # type: ignore

		return formatted_messages, system_message

from groq import APIError
from groq import APIResponseValidationError
from groq import APIStatusError
from groq import AsyncGroq
from groq import NotGiven
from groq import RateLimitError
from groq import Timeout
from groq.types.chat import ChatCompletion
from groq.types.chat import ChatCompletionToolChoiceOptionParam
from groq.types.chat import ChatCompletionToolParam
from groq.types.chat.completion_create_params import ResponseFormatResponseFormatJsonSchema
from groq.types.chat.completion_create_params import ResponseFormatResponseFormatJsonSchemaJsonSchema
from httpx import URL
from browser_use.llm.base import ChatInvokeCompletion
from browser_use.llm.groq.parser import try_parse_groq_failed_generation
from browser_use.llm.groq.serializer import GroqMessageSerializer

# From groq/chat.py
class ChatGroq(BaseChatModel):
	"""
	A wrapper around AsyncGroq that implements the BaseLLM protocol.
	"""

	# Model configuration
	model: GroqVerifiedModels | str

	# Model params
	temperature: float | None = None
	service_tier: Literal['auto', 'on_demand', 'flex'] | None = None
	top_p: float | None = None
	seed: int | None = None

	# Client initialization parameters
	api_key: str | None = None
	base_url: str | URL | None = None
	timeout: float | Timeout | NotGiven | None = None
	max_retries: int = 10  # Increase default retries for automation reliability

	def get_client(self) -> AsyncGroq:
		return AsyncGroq(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout, max_retries=self.max_retries)

	@property
	def provider(self) -> str:
		return 'groq'

	@property
	def name(self) -> str:
		return str(self.model)

	def _get_usage(self, response: ChatCompletion) -> ChatInvokeUsage | None:
		usage = (
			ChatInvokeUsage(
				prompt_tokens=response.usage.prompt_tokens,
				completion_tokens=response.usage.completion_tokens,
				total_tokens=response.usage.total_tokens,
				prompt_cached_tokens=None,  # Groq doesn't support cached tokens
				prompt_cache_creation_tokens=None,
				prompt_image_tokens=None,
			)
			if response.usage is not None
			else None
		)
		return usage

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		groq_messages = GroqMessageSerializer.serialize_messages(messages)

		try:
			if output_format is None:
				return await self._invoke_regular_completion(groq_messages)
			else:
				return await self._invoke_structured_output(groq_messages, output_format)

		except RateLimitError as e:
			raise ModelRateLimitError(message=e.response.text, status_code=e.response.status_code, model=self.name) from e

		except APIResponseValidationError as e:
			raise ModelProviderError(message=e.response.text, status_code=e.response.status_code, model=self.name) from e

		except APIStatusError as e:
			if output_format is None:
				raise ModelProviderError(message=e.response.text, status_code=e.response.status_code, model=self.name) from e
			else:
				try:
					logger.debug(f'Groq failed generation: {e.response.text}; fallback to manual parsing')

					parsed_response = try_parse_groq_failed_generation(e, output_format)

					logger.debug('Manual error parsing successful ✅')

					return ChatInvokeCompletion(
						completion=parsed_response,
						usage=None,  # because this is a hacky way to get the outputs
						# TODO: @groq needs to fix their parsers and validators
					)
				except Exception as _:
					raise ModelProviderError(message=str(e), status_code=e.response.status_code, model=self.name) from e

		except APIError as e:
			raise ModelProviderError(message=e.message, model=self.name) from e
		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e

	async def _invoke_regular_completion(self, groq_messages) -> ChatInvokeCompletion[str]:
		"""Handle regular completion without structured output."""
		chat_completion = await self.get_client().chat.completions.create(
			messages=groq_messages,
			model=self.model,
			service_tier=self.service_tier,
			temperature=self.temperature,
			top_p=self.top_p,
			seed=self.seed,
		)
		usage = self._get_usage(chat_completion)
		return ChatInvokeCompletion(
			completion=chat_completion.choices[0].message.content or '',
			usage=usage,
		)

	async def _invoke_structured_output(self, groq_messages, output_format: type[T]) -> ChatInvokeCompletion[T]:
		"""Handle structured output using either tool calling or JSON schema."""
		schema = SchemaOptimizer.create_optimized_json_schema(output_format)

		if self.model in ToolCallingModels:
			response = await self._invoke_with_tool_calling(groq_messages, output_format, schema)
		else:
			response = await self._invoke_with_json_schema(groq_messages, output_format, schema)

		if not response.choices[0].message.content:
			raise ModelProviderError(
				message='No content in response',
				status_code=500,
				model=self.name,
			)

		parsed_response = output_format.model_validate_json(response.choices[0].message.content)
		usage = self._get_usage(response)

		return ChatInvokeCompletion(
			completion=parsed_response,
			usage=usage,
		)

	async def _invoke_with_tool_calling(self, groq_messages, output_format: type[T], schema) -> ChatCompletion:
		"""Handle structured output using tool calling."""
		tool = ChatCompletionToolParam(
			function={
				'name': output_format.__name__,
				'description': f'Extract information in the format of {output_format.__name__}',
				'parameters': schema,
			},
			type='function',
		)
		tool_choice: ChatCompletionToolChoiceOptionParam = 'required'

		return await self.get_client().chat.completions.create(
			model=self.model,
			messages=groq_messages,
			temperature=self.temperature,
			top_p=self.top_p,
			seed=self.seed,
			tools=[tool],
			tool_choice=tool_choice,
			service_tier=self.service_tier,
		)

	async def _invoke_with_json_schema(self, groq_messages, output_format: type[T], schema) -> ChatCompletion:
		"""Handle structured output using JSON schema."""
		return await self.get_client().chat.completions.create(
			model=self.model,
			messages=groq_messages,
			temperature=self.temperature,
			top_p=self.top_p,
			seed=self.seed,
			response_format=ResponseFormatResponseFormatJsonSchema(
				json_schema=ResponseFormatResponseFormatJsonSchemaJsonSchema(
					name=output_format.__name__,
					description='Model output schema',
					schema=schema,
				),
				type='json_schema',
			),
			service_tier=self.service_tier,
		)


# From groq/parser.py
class ParseFailedGenerationError(Exception):
	pass

# From groq/parser.py
def try_parse_groq_failed_generation(
	error: APIStatusError,
	output_format: type[T],
) -> T:
	"""Extract JSON from model output, handling both plain JSON and code-block-wrapped JSON."""
	try:
		content = error.body['error']['failed_generation']  # type: ignore

		# If content is wrapped in code blocks, extract just the JSON part
		if '```' in content:
			# Find the JSON content between code blocks
			content = content.split('```')[1]
			# Remove language identifier if present (e.g., 'json\n')
			if '\n' in content:
				content = content.split('\n', 1)[1]

		# remove html-like tags before the first { and after the last }
		# This handles cases like <|header_start|>assistant<|header_end|> and <function=AgentOutput>
		# Only remove content before { if content doesn't already start with {
		if not content.strip().startswith('{'):
			content = re.sub(r'^.*?(?=\{)', '', content, flags=re.DOTALL)

		# Remove common HTML-like tags and patterns at the end, but be more conservative
		# Look for patterns like </function>, <|header_start|>, etc. after the JSON
		content = re.sub(r'\}(\s*<[^>]*>.*?$)', '}', content, flags=re.DOTALL)
		content = re.sub(r'\}(\s*<\|[^|]*\|>.*?$)', '}', content, flags=re.DOTALL)

		# Handle extra characters after the JSON, including stray braces
		# Find the position of the last } that would close the main JSON object
		content = content.strip()

		if content.endswith('}'):
			# Try to parse and see if we get valid JSON
			try:
				json.loads(content)
			except json.JSONDecodeError:
				# If parsing fails, try to find the correct end of the JSON
				# by counting braces and removing anything after the balanced JSON
				brace_count = 0
				last_valid_pos = -1
				for i, char in enumerate(content):
					if char == '{':
						brace_count += 1
					elif char == '}':
						brace_count -= 1
						if brace_count == 0:
							last_valid_pos = i + 1
							break

				if last_valid_pos > 0:
					content = content[:last_valid_pos]

		# Fix control characters in JSON strings before parsing
		# This handles cases where literal control characters appear in JSON values
		content = _fix_control_characters_in_json(content)

		# Parse the cleaned content
		result_dict = json.loads(content)

		# some models occasionally respond with a list containing one dict: https://github.com/browser-use/browser-use/issues/1458
		if isinstance(result_dict, list) and len(result_dict) == 1 and isinstance(result_dict[0], dict):
			result_dict = result_dict[0]

		logger.debug(f'Successfully parsed model output: {result_dict}')
		return output_format.model_validate(result_dict)

	except KeyError as e:
		raise ParseFailedGenerationError(e) from e

	except json.JSONDecodeError as e:
		logger.warning(f'Failed to parse model output: {content} {str(e)}')
		raise ValueError(f'Could not parse response. {str(e)}')

	except Exception as e:
		raise ParseFailedGenerationError(error.response.text) from e

from groq.types.chat import ChatCompletionAssistantMessageParam
from groq.types.chat import ChatCompletionContentPartImageParam
from groq.types.chat import ChatCompletionContentPartTextParam
from groq.types.chat import ChatCompletionMessageParam
from groq.types.chat import ChatCompletionMessageToolCallParam
from groq.types.chat import ChatCompletionSystemMessageParam
from groq.types.chat import ChatCompletionUserMessageParam
from groq.types.chat.chat_completion_content_part_image_param import ImageURL
from groq.types.chat.chat_completion_message_tool_call_param import Function

# From groq/serializer.py
class GroqMessageSerializer:
	"""Serializer for converting between custom message types and OpenAI message param types."""

	@staticmethod
	def _serialize_content_part_text(part: ContentPartTextParam) -> ChatCompletionContentPartTextParam:
		return ChatCompletionContentPartTextParam(text=part.text, type='text')

	@staticmethod
	def _serialize_content_part_image(part: ContentPartImageParam) -> ChatCompletionContentPartImageParam:
		return ChatCompletionContentPartImageParam(
			image_url=ImageURL(url=part.image_url.url, detail=part.image_url.detail),
			type='image_url',
		)

	@staticmethod
	def _serialize_user_content(
		content: str | list[ContentPartTextParam | ContentPartImageParam],
	) -> str | list[ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam]:
		"""Serialize content for user messages (text and images allowed)."""
		if isinstance(content, str):
			return content

		serialized_parts: list[ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam] = []
		for part in content:
			if part.type == 'text':
				serialized_parts.append(GroqMessageSerializer._serialize_content_part_text(part))
			elif part.type == 'image_url':
				serialized_parts.append(GroqMessageSerializer._serialize_content_part_image(part))
		return serialized_parts

	@staticmethod
	def _serialize_system_content(
		content: str | list[ContentPartTextParam],
	) -> str:
		"""Serialize content for system messages (text only)."""
		if isinstance(content, str):
			return content

		serialized_parts: list[str] = []
		for part in content:
			if part.type == 'text':
				serialized_parts.append(GroqMessageSerializer._serialize_content_part_text(part)['text'])

		return '\n'.join(serialized_parts)

	@staticmethod
	def _serialize_assistant_content(
		content: str | list[ContentPartTextParam | ContentPartRefusalParam] | None,
	) -> str | None:
		"""Serialize content for assistant messages (text and refusal allowed)."""
		if content is None:
			return None
		if isinstance(content, str):
			return content

		serialized_parts: list[str] = []
		for part in content:
			if part.type == 'text':
				serialized_parts.append(GroqMessageSerializer._serialize_content_part_text(part)['text'])

		return '\n'.join(serialized_parts)

	@staticmethod
	def _serialize_tool_call(tool_call: ToolCall) -> ChatCompletionMessageToolCallParam:
		return ChatCompletionMessageToolCallParam(
			id=tool_call.id,
			function=Function(name=tool_call.function.name, arguments=tool_call.function.arguments),
			type='function',
		)

	# endregion

	# region - Serialize overloads
	@overload
	@staticmethod
	def serialize(message: UserMessage) -> ChatCompletionUserMessageParam: ...

	@overload
	@staticmethod
	def serialize(message: SystemMessage) -> ChatCompletionSystemMessageParam: ...

	@overload
	@staticmethod
	def serialize(message: AssistantMessage) -> ChatCompletionAssistantMessageParam: ...

	@staticmethod
	def serialize(message: BaseMessage) -> ChatCompletionMessageParam:
		"""Serialize a custom message to an OpenAI message param."""

		if isinstance(message, UserMessage):
			user_result: ChatCompletionUserMessageParam = {
				'role': 'user',
				'content': GroqMessageSerializer._serialize_user_content(message.content),
			}
			if message.name is not None:
				user_result['name'] = message.name
			return user_result

		elif isinstance(message, SystemMessage):
			system_result: ChatCompletionSystemMessageParam = {
				'role': 'system',
				'content': GroqMessageSerializer._serialize_system_content(message.content),
			}
			if message.name is not None:
				system_result['name'] = message.name
			return system_result

		elif isinstance(message, AssistantMessage):
			# Handle content serialization
			content = None
			if message.content is not None:
				content = GroqMessageSerializer._serialize_assistant_content(message.content)

			assistant_result: ChatCompletionAssistantMessageParam = {'role': 'assistant'}

			# Only add content if it's not None
			if content is not None:
				assistant_result['content'] = content

			if message.name is not None:
				assistant_result['name'] = message.name

			if message.tool_calls:
				assistant_result['tool_calls'] = [GroqMessageSerializer._serialize_tool_call(tc) for tc in message.tool_calls]

			return assistant_result

		else:
			raise ValueError(f'Unknown message type: {type(message)}')

	@staticmethod
	def serialize_messages(messages: list[BaseMessage]) -> list[ChatCompletionMessageParam]:
		return [GroqMessageSerializer.serialize(m) for m in messages]

from anthropic import NOT_GIVEN
from anthropic import APIConnectionError
from anthropic import APIStatusError
from anthropic import AsyncAnthropicBedrock
from anthropic.types import CacheControlEphemeralParam
from anthropic.types import Message
from anthropic.types import ToolParam
from anthropic.types.text_block import TextBlock
from anthropic.types.tool_choice_tool_param import ToolChoiceToolParam
from browser_use.llm.anthropic.serializer import AnthropicMessageSerializer
from browser_use.llm.aws.chat_bedrock import ChatAWSBedrock
from boto3.session import Session

# From aws/chat_anthropic.py
class ChatAnthropicBedrock(ChatAWSBedrock):
	"""
	AWS Bedrock Anthropic Claude chat model.

	This is a convenience class that provides Claude-specific defaults
	for the AWS Bedrock service. It inherits all functionality from
	ChatAWSBedrock but sets Anthropic Claude as the default model.
	"""

	# Anthropic Claude specific defaults
	model: str = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
	max_tokens: int = 8192
	temperature: float | None = None
	top_p: float | None = None
	top_k: int | None = None
	stop_sequences: list[str] | None = None
	seed: int | None = None

	# AWS credentials and configuration
	aws_access_key: str | None = None
	aws_secret_key: str | None = None
	aws_session_token: str | None = None
	aws_region: str | None = None
	session: 'Session | None' = None

	# Client initialization parameters
	max_retries: int = 10
	default_headers: Mapping[str, str] | None = None
	default_query: Mapping[str, object] | None = None

	@property
	def provider(self) -> str:
		return 'anthropic_bedrock'

	def _get_client_params(self) -> dict[str, Any]:
		"""Prepare client parameters dictionary for Bedrock."""
		client_params: dict[str, Any] = {}

		if self.session:
			credentials = self.session.get_credentials()
			client_params.update(
				{
					'aws_access_key': credentials.access_key,
					'aws_secret_key': credentials.secret_key,
					'aws_session_token': credentials.token,
					'aws_region': self.session.region_name,
				}
			)
		else:
			# Use individual credentials
			if self.aws_access_key:
				client_params['aws_access_key'] = self.aws_access_key
			if self.aws_secret_key:
				client_params['aws_secret_key'] = self.aws_secret_key
			if self.aws_region:
				client_params['aws_region'] = self.aws_region
			if self.aws_session_token:
				client_params['aws_session_token'] = self.aws_session_token

		# Add optional parameters
		if self.max_retries:
			client_params['max_retries'] = self.max_retries
		if self.default_headers:
			client_params['default_headers'] = self.default_headers
		if self.default_query:
			client_params['default_query'] = self.default_query

		return client_params

	def _get_client_params_for_invoke(self) -> dict[str, Any]:
		"""Prepare client parameters dictionary for invoke."""
		client_params = {}

		if self.temperature is not None:
			client_params['temperature'] = self.temperature
		if self.max_tokens is not None:
			client_params['max_tokens'] = self.max_tokens
		if self.top_p is not None:
			client_params['top_p'] = self.top_p
		if self.top_k is not None:
			client_params['top_k'] = self.top_k
		if self.seed is not None:
			client_params['seed'] = self.seed
		if self.stop_sequences is not None:
			client_params['stop_sequences'] = self.stop_sequences

		return client_params

	def get_client(self) -> AsyncAnthropicBedrock:
		"""
		Returns an AsyncAnthropicBedrock client.

		Returns:
			AsyncAnthropicBedrock: An instance of the AsyncAnthropicBedrock client.
		"""
		client_params = self._get_client_params()
		return AsyncAnthropicBedrock(**client_params)

	@property
	def name(self) -> str:
		return str(self.model)

	def _get_usage(self, response: Message) -> ChatInvokeUsage | None:
		"""Extract usage information from the response."""
		usage = ChatInvokeUsage(
			prompt_tokens=response.usage.input_tokens
			+ (
				response.usage.cache_read_input_tokens or 0
			),  # Total tokens in Anthropic are a bit fucked, you have to add cached tokens to the prompt tokens
			completion_tokens=response.usage.output_tokens,
			total_tokens=response.usage.input_tokens + response.usage.output_tokens,
			prompt_cached_tokens=response.usage.cache_read_input_tokens,
			prompt_cache_creation_tokens=response.usage.cache_creation_input_tokens,
			prompt_image_tokens=None,
		)
		return usage

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		anthropic_messages, system_prompt = AnthropicMessageSerializer.serialize_messages(messages)

		try:
			if output_format is None:
				# Normal completion without structured output
				response = await self.get_client().messages.create(
					model=self.model,
					messages=anthropic_messages,
					system=system_prompt or NOT_GIVEN,
					**self._get_client_params_for_invoke(),
				)

				usage = self._get_usage(response)

				# Extract text from the first content block
				first_content = response.content[0]
				if isinstance(first_content, TextBlock):
					response_text = first_content.text
				else:
					# If it's not a text block, convert to string
					response_text = str(first_content)

				return ChatInvokeCompletion(
					completion=response_text,
					usage=usage,
				)

			else:
				# Use tool calling for structured output
				# Create a tool that represents the output format
				tool_name = output_format.__name__
				schema = output_format.model_json_schema()

				# Remove title from schema if present (Anthropic doesn't like it in parameters)
				if 'title' in schema:
					del schema['title']

				tool = ToolParam(
					name=tool_name,
					description=f'Extract information in the format of {tool_name}',
					input_schema=schema,
					cache_control=CacheControlEphemeralParam(type='ephemeral'),
				)

				# Force the model to use this tool
				tool_choice = ToolChoiceToolParam(type='tool', name=tool_name)

				response = await self.get_client().messages.create(
					model=self.model,
					messages=anthropic_messages,
					tools=[tool],
					system=system_prompt or NOT_GIVEN,
					tool_choice=tool_choice,
					**self._get_client_params_for_invoke(),
				)

				usage = self._get_usage(response)

				# Extract the tool use block
				for content_block in response.content:
					if hasattr(content_block, 'type') and content_block.type == 'tool_use':
						# Parse the tool input as the structured output
						try:
							return ChatInvokeCompletion(completion=output_format.model_validate(content_block.input), usage=usage)
						except Exception as e:
							# If validation fails, try to parse it as JSON first
							if isinstance(content_block.input, str):
								data = json.loads(content_block.input)
								return ChatInvokeCompletion(
									completion=output_format.model_validate(data),
									usage=usage,
								)
							raise e

				# If no tool use block found, raise an error
				raise ValueError('Expected tool use in response but none found')

		except APIConnectionError as e:
			raise ModelProviderError(message=e.message, model=self.name) from e
		except RateLimitError as e:
			raise ModelRateLimitError(message=e.message, model=self.name) from e
		except APIStatusError as e:
			raise ModelProviderError(message=e.message, status_code=e.status_code, model=self.name) from e
		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e

from os import getenv
from browser_use.llm.aws.serializer import AWSBedrockMessageSerializer
from boto3 import client
from botocore.exceptions import ClientError

# From aws/chat_bedrock.py
class ChatAWSBedrock(BaseChatModel):
	"""
	AWS Bedrock chat model supporting multiple providers (Anthropic, Meta, etc.).

	This class provides access to various models via AWS Bedrock,
	supporting both text generation and structured output via tool calling.

	To use this model, you need to either:
	1. Set the following environment variables:
	   - AWS_ACCESS_KEY_ID
	   - AWS_SECRET_ACCESS_KEY
	   - AWS_REGION
	2. Or provide a boto3 Session object
	3. Or use AWS SSO authentication
	"""

	# Model configuration
	model: str = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
	max_tokens: int | None = 4096
	temperature: float | None = None
	top_p: float | None = None
	seed: int | None = None
	stop_sequences: list[str] | None = None

	# AWS credentials and configuration
	aws_access_key_id: str | None = None
	aws_secret_access_key: str | None = None
	aws_region: str | None = None
	aws_sso_auth: bool = False
	session: 'Session | None' = None

	# Request parameters
	request_params: dict[str, Any] | None = None

	# Static
	@property
	def provider(self) -> str:
		return 'aws_bedrock'

	def _get_client(self) -> 'AwsClient':  # type: ignore
		"""Get the AWS Bedrock client."""
		try:
			from boto3 import client as AwsClient  # type: ignore
		except ImportError:
			raise ImportError(
				'`boto3` not installed. Please install using `pip install browser-use[aws] or pip install browser-use[all]`'
			)

		if self.session:
			return self.session.client('bedrock-runtime')

		# Get credentials from environment or instance parameters
		access_key = self.aws_access_key_id or getenv('AWS_ACCESS_KEY_ID')
		secret_key = self.aws_secret_access_key or getenv('AWS_SECRET_ACCESS_KEY')
		region = self.aws_region or getenv('AWS_REGION') or getenv('AWS_DEFAULT_REGION')

		if self.aws_sso_auth:
			return AwsClient(service_name='bedrock-runtime', region_name=region)
		else:
			if not access_key or not secret_key:
				raise ModelProviderError(
					message='AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables or provide a boto3 session.',
					model=self.name,
				)

			return AwsClient(
				service_name='bedrock-runtime',
				region_name=region,
				aws_access_key_id=access_key,
				aws_secret_access_key=secret_key,
			)

	@property
	def name(self) -> str:
		return str(self.model)

	def _get_inference_config(self) -> dict[str, Any]:
		"""Get the inference configuration for the request."""
		config = {}
		if self.max_tokens is not None:
			config['maxTokens'] = self.max_tokens
		if self.temperature is not None:
			config['temperature'] = self.temperature
		if self.top_p is not None:
			config['topP'] = self.top_p
		if self.stop_sequences is not None:
			config['stopSequences'] = self.stop_sequences
		if self.seed is not None:
			config['seed'] = self.seed
		return config

	def _format_tools_for_request(self, output_format: type[BaseModel]) -> list[dict[str, Any]]:
		"""Format a Pydantic model as a tool for structured output."""
		schema = output_format.model_json_schema()

		# Convert Pydantic schema to Bedrock tool format
		properties = {}
		required = []

		for prop_name, prop_info in schema.get('properties', {}).items():
			properties[prop_name] = {
				'type': prop_info.get('type', 'string'),
				'description': prop_info.get('description', ''),
			}

		# Add required fields
		required = schema.get('required', [])

		return [
			{
				'toolSpec': {
					'name': f'extract_{output_format.__name__.lower()}',
					'description': f'Extract information in the format of {output_format.__name__}',
					'inputSchema': {'json': {'type': 'object', 'properties': properties, 'required': required}},
				}
			}
		]

	def _get_usage(self, response: dict[str, Any]) -> ChatInvokeUsage | None:
		"""Extract usage information from the response."""
		if 'usage' not in response:
			return None

		usage_data = response['usage']
		return ChatInvokeUsage(
			prompt_tokens=usage_data.get('inputTokens', 0),
			completion_tokens=usage_data.get('outputTokens', 0),
			total_tokens=usage_data.get('totalTokens', 0),
			prompt_cached_tokens=None,  # Bedrock doesn't provide this
			prompt_cache_creation_tokens=None,
			prompt_image_tokens=None,
		)

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		"""
		Invoke the AWS Bedrock model with the given messages.

		Args:
			messages: List of chat messages
			output_format: Optional Pydantic model class for structured output

		Returns:
			Either a string response or an instance of output_format
		"""
		try:
			from botocore.exceptions import ClientError  # type: ignore
		except ImportError:
			raise ImportError(
				'`boto3` not installed. Please install using `pip install browser-use[aws] or pip install browser-use[all]`'
			)

		bedrock_messages, system_message = AWSBedrockMessageSerializer.serialize_messages(messages)

		try:
			# Prepare the request body
			body: dict[str, Any] = {}

			if system_message:
				body['system'] = system_message

			inference_config = self._get_inference_config()
			if inference_config:
				body['inferenceConfig'] = inference_config

			# Handle structured output via tool calling
			if output_format is not None:
				tools = self._format_tools_for_request(output_format)
				body['toolConfig'] = {'tools': tools}

			# Add any additional request parameters
			if self.request_params:
				body.update(self.request_params)

			# Filter out None values
			body = {k: v for k, v in body.items() if v is not None}

			# Make the API call
			client = self._get_client()
			response = client.converse(modelId=self.model, messages=bedrock_messages, **body)

			usage = self._get_usage(response)

			# Extract the response content
			if 'output' in response and 'message' in response['output']:
				message = response['output']['message']
				content = message.get('content', [])

				if output_format is None:
					# Return text response
					text_content = []
					for item in content:
						if 'text' in item:
							text_content.append(item['text'])

					response_text = '\n'.join(text_content) if text_content else ''
					return ChatInvokeCompletion(
						completion=response_text,
						usage=usage,
					)
				else:
					# Handle structured output from tool calls
					for item in content:
						if 'toolUse' in item:
							tool_use = item['toolUse']
							tool_input = tool_use.get('input', {})

							try:
								# Validate and return the structured output
								return ChatInvokeCompletion(
									completion=output_format.model_validate(tool_input),
									usage=usage,
								)
							except Exception as e:
								# If validation fails, try to parse as JSON first
								if isinstance(tool_input, str):
									try:
										data = json.loads(tool_input)
										return ChatInvokeCompletion(
											completion=output_format.model_validate(data),
											usage=usage,
										)
									except json.JSONDecodeError:
										pass
								raise ModelProviderError(
									message=f'Failed to validate structured output: {str(e)}',
									model=self.name,
								) from e

					# If no tool use found but output_format was requested
					raise ModelProviderError(
						message='Expected structured output but no tool use found in response',
						model=self.name,
					)

			# If no valid content found
			if output_format is None:
				return ChatInvokeCompletion(
					completion='',
					usage=usage,
				)
			else:
				raise ModelProviderError(
					message='No valid content found in response',
					model=self.name,
				)

		except ClientError as e:
			error_code = e.response.get('Error', {}).get('Code', 'Unknown')
			error_message = e.response.get('Error', {}).get('Message', str(e))

			if error_code in ['ThrottlingException', 'TooManyRequestsException']:
				raise ModelRateLimitError(message=error_message, model=self.name) from e
			else:
				raise ModelProviderError(message=error_message, model=self.name) from e
		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e


# From aws/serializer.py
class AWSBedrockMessageSerializer:
	"""Serializer for converting between custom message types and AWS Bedrock message format."""

	@staticmethod
	def _is_base64_image(url: str) -> bool:
		"""Check if the URL is a base64 encoded image."""
		return url.startswith('data:image/')

	@staticmethod
	def _is_url_image(url: str) -> bool:
		"""Check if the URL is a regular HTTP/HTTPS image URL."""
		return url.startswith(('http://', 'https://')) and any(
			url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
		)

	@staticmethod
	def _parse_base64_url(url: str) -> tuple[str, bytes]:
		"""Parse a base64 data URL to extract format and raw bytes."""
		# Format: data:image/jpeg;base64,<data>
		if not url.startswith('data:'):
			raise ValueError(f'Invalid base64 URL: {url}')

		header, data = url.split(',', 1)

		# Extract format from mime type
		mime_match = re.search(r'image/(\w+)', header)
		if mime_match:
			format_name = mime_match.group(1).lower()
			# Map common formats
			format_mapping = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', 'gif': 'gif', 'webp': 'webp'}
			image_format = format_mapping.get(format_name, 'jpeg')
		else:
			image_format = 'jpeg'  # Default format

		# Decode base64 data
		try:
			image_bytes = base64.b64decode(data)
		except Exception as e:
			raise ValueError(f'Failed to decode base64 image data: {e}')

		return image_format, image_bytes

	@staticmethod
	def _download_and_convert_image(url: str) -> tuple[str, bytes]:
		"""Download an image from URL and convert to base64 bytes."""
		try:
			import httpx
		except ImportError:
			raise ImportError('httpx not available. Please install it to use URL images with AWS Bedrock.')

		try:
			response = httpx.get(url, timeout=30)
			response.raise_for_status()

			# Detect format from content type or URL
			content_type = response.headers.get('content-type', '').lower()
			if 'jpeg' in content_type or url.lower().endswith(('.jpg', '.jpeg')):
				image_format = 'jpeg'
			elif 'png' in content_type or url.lower().endswith('.png'):
				image_format = 'png'
			elif 'gif' in content_type or url.lower().endswith('.gif'):
				image_format = 'gif'
			elif 'webp' in content_type or url.lower().endswith('.webp'):
				image_format = 'webp'
			else:
				image_format = 'jpeg'  # Default format

			return image_format, response.content

		except Exception as e:
			raise ValueError(f'Failed to download image from {url}: {e}')

	@staticmethod
	def _serialize_content_part_text(part: ContentPartTextParam) -> dict[str, Any]:
		"""Convert a text content part to AWS Bedrock format."""
		return {'text': part.text}

	@staticmethod
	def _serialize_content_part_image(part: ContentPartImageParam) -> dict[str, Any]:
		"""Convert an image content part to AWS Bedrock format."""
		url = part.image_url.url

		if AWSBedrockMessageSerializer._is_base64_image(url):
			# Handle base64 encoded images
			image_format, image_bytes = AWSBedrockMessageSerializer._parse_base64_url(url)
		elif AWSBedrockMessageSerializer._is_url_image(url):
			# Download and convert URL images
			image_format, image_bytes = AWSBedrockMessageSerializer._download_and_convert_image(url)
		else:
			raise ValueError(f'Unsupported image URL format: {url}')

		return {
			'image': {
				'format': image_format,
				'source': {
					'bytes': image_bytes,
				},
			}
		}

	@staticmethod
	def _serialize_user_content(
		content: str | list[ContentPartTextParam | ContentPartImageParam],
	) -> list[dict[str, Any]]:
		"""Serialize content for user messages."""
		if isinstance(content, str):
			return [{'text': content}]

		content_blocks: list[dict[str, Any]] = []
		for part in content:
			if part.type == 'text':
				content_blocks.append(AWSBedrockMessageSerializer._serialize_content_part_text(part))
			elif part.type == 'image_url':
				content_blocks.append(AWSBedrockMessageSerializer._serialize_content_part_image(part))

		return content_blocks

	@staticmethod
	def _serialize_system_content(
		content: str | list[ContentPartTextParam],
	) -> list[dict[str, Any]]:
		"""Serialize content for system messages."""
		if isinstance(content, str):
			return [{'text': content}]

		content_blocks: list[dict[str, Any]] = []
		for part in content:
			if part.type == 'text':
				content_blocks.append(AWSBedrockMessageSerializer._serialize_content_part_text(part))

		return content_blocks

	@staticmethod
	def _serialize_assistant_content(
		content: str | list[ContentPartTextParam | ContentPartRefusalParam] | None,
	) -> list[dict[str, Any]]:
		"""Serialize content for assistant messages."""
		if content is None:
			return []
		if isinstance(content, str):
			return [{'text': content}]

		content_blocks: list[dict[str, Any]] = []
		for part in content:
			if part.type == 'text':
				content_blocks.append(AWSBedrockMessageSerializer._serialize_content_part_text(part))
			# Skip refusal content parts - AWS Bedrock doesn't need them

		return content_blocks

	@staticmethod
	def _serialize_tool_call(tool_call: ToolCall) -> dict[str, Any]:
		"""Convert a tool call to AWS Bedrock format."""
		try:
			arguments = json.loads(tool_call.function.arguments)
		except json.JSONDecodeError:
			# If arguments aren't valid JSON, wrap them
			arguments = {'arguments': tool_call.function.arguments}

		return {
			'toolUse': {
				'toolUseId': tool_call.id,
				'name': tool_call.function.name,
				'input': arguments,
			}
		}

	# region - Serialize overloads
	@overload
	@staticmethod
	def serialize(message: UserMessage) -> dict[str, Any]: ...

	@overload
	@staticmethod
	def serialize(message: SystemMessage) -> SystemMessage: ...

	@overload
	@staticmethod
	def serialize(message: AssistantMessage) -> dict[str, Any]: ...

	@staticmethod
	def serialize(message: BaseMessage) -> dict[str, Any] | SystemMessage:
		"""Serialize a custom message to AWS Bedrock format."""

		if isinstance(message, UserMessage):
			return {
				'role': 'user',
				'content': AWSBedrockMessageSerializer._serialize_user_content(message.content),
			}

		elif isinstance(message, SystemMessage):
			# System messages are handled separately in AWS Bedrock
			return message

		elif isinstance(message, AssistantMessage):
			content_blocks: list[dict[str, Any]] = []

			# Add content blocks if present
			if message.content is not None:
				content_blocks.extend(AWSBedrockMessageSerializer._serialize_assistant_content(message.content))

			# Add tool use blocks if present
			if message.tool_calls:
				for tool_call in message.tool_calls:
					content_blocks.append(AWSBedrockMessageSerializer._serialize_tool_call(tool_call))

			# AWS Bedrock requires at least one content block
			if not content_blocks:
				content_blocks = [{'text': ''}]

			return {
				'role': 'assistant',
				'content': content_blocks,
			}

		else:
			raise ValueError(f'Unknown message type: {type(message)}')

	@staticmethod
	def serialize_messages(messages: list[BaseMessage]) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
		"""
		Serialize a list of messages, extracting any system message.

		Returns:
			Tuple of (bedrock_messages, system_message) where system_message is extracted
			from any SystemMessage in the list.
		"""
		bedrock_messages: list[dict[str, Any]] = []
		system_message: list[dict[str, Any]] | None = None

		for message in messages:
			if isinstance(message, SystemMessage):
				# Extract system message content
				system_message = AWSBedrockMessageSerializer._serialize_system_content(message.content)
			else:
				# Serialize and add to regular messages
				serialized = AWSBedrockMessageSerializer.serialize(message)
				bedrock_messages.append(serialized)

		return bedrock_messages, system_message

from browser_use.llm.openrouter.serializer import OpenRouterMessageSerializer

# From openrouter/chat.py
class ChatOpenRouter(BaseChatModel):
	"""
	A wrapper around OpenRouter's chat API, which provides access to various LLM models
	through a unified OpenAI-compatible interface.

	This class implements the BaseChatModel protocol for OpenRouter's API.
	"""

	# Model configuration
	model: str

	# Model params
	temperature: float | None = None
	top_p: float | None = None
	seed: int | None = None

	# Client initialization parameters
	api_key: str | None = None
	http_referer: str | None = None  # OpenRouter specific parameter for tracking
	base_url: str | httpx.URL = 'https://openrouter.ai/api/v1'
	timeout: float | httpx.Timeout | None = None
	max_retries: int = 10
	default_headers: Mapping[str, str] | None = None
	default_query: Mapping[str, object] | None = None
	http_client: httpx.AsyncClient | None = None
	_strict_response_validation: bool = False

	# Static
	@property
	def provider(self) -> str:
		return 'openrouter'

	def _get_client_params(self) -> dict[str, Any]:
		"""Prepare client parameters dictionary."""
		# Define base client params
		base_params = {
			'api_key': self.api_key,
			'base_url': self.base_url,
			'timeout': self.timeout,
			'max_retries': self.max_retries,
			'default_headers': self.default_headers,
			'default_query': self.default_query,
			'_strict_response_validation': self._strict_response_validation,
			'top_p': self.top_p,
			'seed': self.seed,
		}

		# Create client_params dict with non-None values
		client_params = {k: v for k, v in base_params.items() if v is not None}

		# Add http_client if provided
		if self.http_client is not None:
			client_params['http_client'] = self.http_client

		return client_params

	def get_client(self) -> AsyncOpenAI:
		"""
		Returns an AsyncOpenAI client configured for OpenRouter.

		Returns:
		    AsyncOpenAI: An instance of the AsyncOpenAI client with OpenRouter base URL.
		"""
		if not hasattr(self, '_client'):
			client_params = self._get_client_params()
			self._client = AsyncOpenAI(**client_params)
		return self._client

	@property
	def name(self) -> str:
		return str(self.model)

	def _get_usage(self, response: ChatCompletion) -> ChatInvokeUsage | None:
		"""Extract usage information from the OpenRouter response."""
		if response.usage is None:
			return None

		prompt_details = getattr(response.usage, 'prompt_tokens_details', None)
		cached_tokens = prompt_details.cached_tokens if prompt_details else None

		return ChatInvokeUsage(
			prompt_tokens=response.usage.prompt_tokens,
			prompt_cached_tokens=cached_tokens,
			prompt_cache_creation_tokens=None,
			prompt_image_tokens=None,
			# Completion
			completion_tokens=response.usage.completion_tokens,
			total_tokens=response.usage.total_tokens,
		)

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		"""
		Invoke the model with the given messages through OpenRouter.

		Args:
		    messages: List of chat messages
		    output_format: Optional Pydantic model class for structured output

		Returns:
		    Either a string response or an instance of output_format
		"""
		openrouter_messages = OpenRouterMessageSerializer.serialize_messages(messages)

		# Set up extra headers for OpenRouter
		extra_headers = {}
		if self.http_referer:
			extra_headers['HTTP-Referer'] = self.http_referer

		try:
			if output_format is None:
				# Return string response
				response = await self.get_client().chat.completions.create(
					model=self.model,
					messages=openrouter_messages,
					temperature=self.temperature,
					top_p=self.top_p,
					seed=self.seed,
					extra_headers=extra_headers,
				)

				usage = self._get_usage(response)
				return ChatInvokeCompletion(
					completion=response.choices[0].message.content or '',
					usage=usage,
				)

			else:
				# Create a JSON schema for structured output
				schema = SchemaOptimizer.create_optimized_json_schema(output_format)

				response_format_schema: JSONSchema = {
					'name': 'agent_output',
					'strict': True,
					'schema': schema,
				}

				# Return structured response
				response = await self.get_client().chat.completions.create(
					model=self.model,
					messages=openrouter_messages,
					temperature=self.temperature,
					top_p=self.top_p,
					seed=self.seed,
					response_format=ResponseFormatJSONSchema(
						json_schema=response_format_schema,
						type='json_schema',
					),
					extra_headers=extra_headers,
				)

				if response.choices[0].message.content is None:
					raise ModelProviderError(
						message='Failed to parse structured output from model response',
						status_code=500,
						model=self.name,
					)
				usage = self._get_usage(response)

				parsed = output_format.model_validate_json(response.choices[0].message.content)

				return ChatInvokeCompletion(
					completion=parsed,
					usage=usage,
				)

		except RateLimitError as e:
			raise ModelRateLimitError(message=e.message, model=self.name) from e

		except APIConnectionError as e:
			raise ModelProviderError(message=str(e), model=self.name) from e

		except APIStatusError as e:
			raise ModelProviderError(message=e.message, status_code=e.status_code, model=self.name) from e

		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e


# From openrouter/serializer.py
class OpenRouterMessageSerializer:
	"""
	Serializer for converting between custom message types and OpenRouter message formats.

	OpenRouter uses the OpenAI-compatible API, so we can reuse the OpenAI serializer.
	"""

	@staticmethod
	def serialize_messages(messages: list[BaseMessage]) -> list[ChatCompletionMessageParam]:
		"""
		Serialize a list of browser_use messages to OpenRouter-compatible messages.

		Args:
		    messages: List of browser_use messages

		Returns:
		    List of OpenRouter-compatible messages (identical to OpenAI format)
		"""
		# OpenRouter uses the same message format as OpenAI
		return OpenAIMessageSerializer.serialize_messages(messages)

from ollama import AsyncClient
from browser_use.llm.ollama.serializer import OllamaMessageSerializer

# From ollama/chat.py
class ChatOllama(BaseChatModel):
	"""
	A wrapper around Ollama's chat model.
	"""

	model: str

	# # Model params
	# TODO (matic): Why is this commented out?
	# temperature: float | None = None

	# Client initialization parameters
	host: str | None = None
	timeout: float | httpx.Timeout | None = None
	client_params: dict[str, Any] | None = None

	# Static
	@property
	def provider(self) -> str:
		return 'ollama'

	def _get_client_params(self) -> dict[str, Any]:
		"""Prepare client parameters dictionary."""
		return {
			'host': self.host,
			'timeout': self.timeout,
			'client_params': self.client_params,
		}

	def get_client(self) -> OllamaAsyncClient:
		"""
		Returns an OllamaAsyncClient client.
		"""
		return OllamaAsyncClient(host=self.host, timeout=self.timeout, **self.client_params or {})

	@property
	def name(self) -> str:
		return self.model

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		ollama_messages = OllamaMessageSerializer.serialize_messages(messages)

		try:
			if output_format is None:
				response = await self.get_client().chat(
					model=self.model,
					messages=ollama_messages,
				)

				return ChatInvokeCompletion(completion=response.message.content or '', usage=None)
			else:
				schema = output_format.model_json_schema()

				response = await self.get_client().chat(
					model=self.model,
					messages=ollama_messages,
					format=schema,
				)

				completion = response.message.content or ''
				if output_format is not None:
					completion = output_format.model_validate_json(completion)

				return ChatInvokeCompletion(completion=completion, usage=None)

		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e

from ollama._types import Image
from ollama._types import Message

# From ollama/serializer.py
class OllamaMessageSerializer:
	"""Serializer for converting between custom message types and Ollama message types."""

	@staticmethod
	def _extract_text_content(content: Any) -> str:
		"""Extract text content from message content, ignoring images."""
		if content is None:
			return ''
		if isinstance(content, str):
			return content

		text_parts: list[str] = []
		for part in content:
			if hasattr(part, 'type'):
				if part.type == 'text':
					text_parts.append(part.text)
				elif part.type == 'refusal':
					text_parts.append(f'[Refusal] {part.refusal}')
			# Skip image parts as they're handled separately

		return '\n'.join(text_parts)

	@staticmethod
	def _extract_images(content: Any) -> list[Image]:
		"""Extract images from message content."""
		if content is None or isinstance(content, str):
			return []

		images: list[Image] = []
		for part in content:
			if hasattr(part, 'type') and part.type == 'image_url':
				url = part.image_url.url
				if url.startswith('data:'):
					# Handle base64 encoded images
					# Format: data:image/png;base64,<data>
					_, data = url.split(',', 1)
					# Decode base64 to bytes
					image_bytes = base64.b64decode(data)
					images.append(Image(value=image_bytes))
				else:
					# Handle URL images (Ollama will download them)
					images.append(Image(value=url))

		return images

	@staticmethod
	def _serialize_tool_calls(tool_calls: list[ToolCall]) -> list[Message.ToolCall]:
		"""Convert browser-use ToolCalls to Ollama ToolCalls."""
		ollama_tool_calls: list[Message.ToolCall] = []

		for tool_call in tool_calls:
			# Parse arguments from JSON string to dict for Ollama
			try:
				arguments_dict = json.loads(tool_call.function.arguments)
			except json.JSONDecodeError:
				# If parsing fails, wrap in a dict
				arguments_dict = {'arguments': tool_call.function.arguments}

			ollama_tool_call = Message.ToolCall(
				function=Message.ToolCall.Function(name=tool_call.function.name, arguments=arguments_dict)
			)
			ollama_tool_calls.append(ollama_tool_call)

		return ollama_tool_calls

	# region - Serialize overloads
	@overload
	@staticmethod
	def serialize(message: UserMessage) -> Message: ...

	@overload
	@staticmethod
	def serialize(message: SystemMessage) -> Message: ...

	@overload
	@staticmethod
	def serialize(message: AssistantMessage) -> Message: ...

	@staticmethod
	def serialize(message: BaseMessage) -> Message:
		"""Serialize a custom message to an Ollama Message."""

		if isinstance(message, UserMessage):
			text_content = OllamaMessageSerializer._extract_text_content(message.content)
			images = OllamaMessageSerializer._extract_images(message.content)

			ollama_message = Message(
				role='user',
				content=text_content if text_content else None,
			)

			if images:
				ollama_message.images = images

			return ollama_message

		elif isinstance(message, SystemMessage):
			text_content = OllamaMessageSerializer._extract_text_content(message.content)

			return Message(
				role='system',
				content=text_content if text_content else None,
			)

		elif isinstance(message, AssistantMessage):
			# Handle content
			text_content = None
			if message.content is not None:
				text_content = OllamaMessageSerializer._extract_text_content(message.content)

			ollama_message = Message(
				role='assistant',
				content=text_content if text_content else None,
			)

			# Handle tool calls
			if message.tool_calls:
				ollama_message.tool_calls = OllamaMessageSerializer._serialize_tool_calls(message.tool_calls)

			return ollama_message

		else:
			raise ValueError(f'Unknown message type: {type(message)}')

	@staticmethod
	def serialize_messages(messages: list[BaseMessage]) -> list[Message]:
		"""Serialize a list of browser_use messages to Ollama Messages."""
		return [OllamaMessageSerializer.serialize(m) for m in messages]

from anthropic import AsyncAnthropic
from anthropic import NotGiven
from anthropic.types.model_param import ModelParam
from httpx import Timeout

# From anthropic/chat.py
class ChatAnthropic(BaseChatModel):
	"""
	A wrapper around Anthropic's chat model.
	"""

	# Model configuration
	model: str | ModelParam
	max_tokens: int = 8192
	temperature: float | None = None
	top_p: float | None = None
	seed: int | None = None

	# Client initialization parameters
	api_key: str | None = None
	auth_token: str | None = None
	base_url: str | httpx.URL | None = None
	timeout: float | Timeout | None | NotGiven = NotGiven()
	max_retries: int = 10
	default_headers: Mapping[str, str] | None = None
	default_query: Mapping[str, object] | None = None

	# Static
	@property
	def provider(self) -> str:
		return 'anthropic'

	def _get_client_params(self) -> dict[str, Any]:
		"""Prepare client parameters dictionary."""
		# Define base client params
		base_params = {
			'api_key': self.api_key,
			'auth_token': self.auth_token,
			'base_url': self.base_url,
			'timeout': self.timeout,
			'max_retries': self.max_retries,
			'default_headers': self.default_headers,
			'default_query': self.default_query,
		}

		# Create client_params dict with non-None values and non-NotGiven values
		client_params = {}
		for k, v in base_params.items():
			if v is not None and v is not NotGiven():
				client_params[k] = v

		return client_params

	def _get_client_params_for_invoke(self):
		"""Prepare client parameters dictionary for invoke."""

		client_params = {}

		if self.temperature is not None:
			client_params['temperature'] = self.temperature

		if self.max_tokens is not None:
			client_params['max_tokens'] = self.max_tokens

		if self.top_p is not None:
			client_params['top_p'] = self.top_p

		if self.seed is not None:
			client_params['seed'] = self.seed

		return client_params

	def get_client(self) -> AsyncAnthropic:
		"""
		Returns an AsyncAnthropic client.

		Returns:
			AsyncAnthropic: An instance of the AsyncAnthropic client.
		"""
		client_params = self._get_client_params()
		return AsyncAnthropic(**client_params)

	@property
	def name(self) -> str:
		return str(self.model)

	def _get_usage(self, response: Message) -> ChatInvokeUsage | None:
		usage = ChatInvokeUsage(
			prompt_tokens=response.usage.input_tokens
			+ (
				response.usage.cache_read_input_tokens or 0
			),  # Total tokens in Anthropic are a bit fucked, you have to add cached tokens to the prompt tokens
			completion_tokens=response.usage.output_tokens,
			total_tokens=response.usage.input_tokens + response.usage.output_tokens,
			prompt_cached_tokens=response.usage.cache_read_input_tokens,
			prompt_cache_creation_tokens=response.usage.cache_creation_input_tokens,
			prompt_image_tokens=None,
		)
		return usage

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		anthropic_messages, system_prompt = AnthropicMessageSerializer.serialize_messages(messages)

		try:
			if output_format is None:
				# Normal completion without structured output
				response = await self.get_client().messages.create(
					model=self.model,
					messages=anthropic_messages,
					system=system_prompt or NOT_GIVEN,
					**self._get_client_params_for_invoke(),
				)

				# Ensure we have a valid Message object before accessing attributes
				if not isinstance(response, Message):
					raise ModelProviderError(
						message=f'Unexpected response type from Anthropic API: {type(response).__name__}. Response: {str(response)[:200]}',
						status_code=502,
						model=self.name,
					)

				usage = self._get_usage(response)

				# Extract text from the first content block
				first_content = response.content[0]
				if isinstance(first_content, TextBlock):
					response_text = first_content.text
				else:
					# If it's not a text block, convert to string
					response_text = str(first_content)

				return ChatInvokeCompletion(
					completion=response_text,
					usage=usage,
				)

			else:
				# Use tool calling for structured output
				# Create a tool that represents the output format
				tool_name = output_format.__name__
				schema = SchemaOptimizer.create_optimized_json_schema(output_format)

				# Remove title from schema if present (Anthropic doesn't like it in parameters)
				if 'title' in schema:
					del schema['title']

				tool = ToolParam(
					name=tool_name,
					description=f'Extract information in the format of {tool_name}',
					input_schema=schema,
					cache_control=CacheControlEphemeralParam(type='ephemeral'),
				)

				# Force the model to use this tool
				tool_choice = ToolChoiceToolParam(type='tool', name=tool_name)

				response = await self.get_client().messages.create(
					model=self.model,
					messages=anthropic_messages,
					tools=[tool],
					system=system_prompt or NOT_GIVEN,
					tool_choice=tool_choice,
					**self._get_client_params_for_invoke(),
				)

				# Ensure we have a valid Message object before accessing attributes
				if not isinstance(response, Message):
					raise ModelProviderError(
						message=f'Unexpected response type from Anthropic API: {type(response).__name__}. Response: {str(response)[:200]}',
						status_code=502,
						model=self.name,
					)

				usage = self._get_usage(response)

				# Extract the tool use block
				for content_block in response.content:
					if hasattr(content_block, 'type') and content_block.type == 'tool_use':
						# Parse the tool input as the structured output
						try:
							return ChatInvokeCompletion(completion=output_format.model_validate(content_block.input), usage=usage)
						except Exception as e:
							# If validation fails, try to parse it as JSON first
							if isinstance(content_block.input, str):
								data = json.loads(content_block.input)
								return ChatInvokeCompletion(
									completion=output_format.model_validate(data),
									usage=usage,
								)
							raise e

				# If no tool use block found, raise an error
				raise ValueError('Expected tool use in response but none found')

		except APIConnectionError as e:
			raise ModelProviderError(message=e.message, model=self.name) from e
		except RateLimitError as e:
			raise ModelRateLimitError(message=e.message, model=self.name) from e
		except APIStatusError as e:
			raise ModelProviderError(message=e.message, status_code=e.status_code, model=self.name) from e
		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e

from anthropic.types import Base64ImageSourceParam
from anthropic.types import ImageBlockParam
from anthropic.types import MessageParam
from anthropic.types import TextBlockParam
from anthropic.types import ToolUseBlockParam
from anthropic.types import URLImageSourceParam
from browser_use.llm.messages import SupportedImageMediaType

# From anthropic/serializer.py
class AnthropicMessageSerializer:
	"""Serializer for converting between custom message types and Anthropic message param types."""

	@staticmethod
	def _is_base64_image(url: str) -> bool:
		"""Check if the URL is a base64 encoded image."""
		return url.startswith('data:image/')

	@staticmethod
	def _parse_base64_url(url: str) -> tuple[SupportedImageMediaType, str]:
		"""Parse a base64 data URL to extract media type and data."""
		# Format: data:image/jpeg;base64,<data>
		if not url.startswith('data:'):
			raise ValueError(f'Invalid base64 URL: {url}')

		header, data = url.split(',', 1)
		media_type = header.split(';')[0].replace('data:', '')

		# Ensure it's a supported media type
		supported_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
		if media_type not in supported_types:
			# Default to png if not recognized
			media_type = 'image/png'

		return media_type, data  # type: ignore

	@staticmethod
	def _serialize_cache_control(use_cache: bool) -> CacheControlEphemeralParam | None:
		"""Serialize cache control."""
		if use_cache:
			return CacheControlEphemeralParam(type='ephemeral')
		return None

	@staticmethod
	def _serialize_content_part_text(part: ContentPartTextParam, use_cache: bool) -> TextBlockParam:
		"""Convert a text content part to Anthropic's TextBlockParam."""
		return TextBlockParam(
			text=part.text, type='text', cache_control=AnthropicMessageSerializer._serialize_cache_control(use_cache)
		)

	@staticmethod
	def _serialize_content_part_image(part: ContentPartImageParam) -> ImageBlockParam:
		"""Convert an image content part to Anthropic's ImageBlockParam."""
		url = part.image_url.url

		if AnthropicMessageSerializer._is_base64_image(url):
			# Handle base64 encoded images
			media_type, data = AnthropicMessageSerializer._parse_base64_url(url)
			return ImageBlockParam(
				source=Base64ImageSourceParam(
					data=data,
					media_type=media_type,
					type='base64',
				),
				type='image',
			)
		else:
			# Handle URL images
			return ImageBlockParam(source=URLImageSourceParam(url=url, type='url'), type='image')

	@staticmethod
	def _serialize_content_to_str(
		content: str | list[ContentPartTextParam], use_cache: bool = False
	) -> list[TextBlockParam] | str:
		"""Serialize content to a string."""
		cache_control = AnthropicMessageSerializer._serialize_cache_control(use_cache)

		if isinstance(content, str):
			if cache_control:
				return [TextBlockParam(text=content, type='text', cache_control=cache_control)]
			else:
				return content

		serialized_blocks: list[TextBlockParam] = []
		for part in content:
			if part.type == 'text':
				serialized_blocks.append(AnthropicMessageSerializer._serialize_content_part_text(part, use_cache))

		return serialized_blocks

	@staticmethod
	def _serialize_content(
		content: str | list[ContentPartTextParam | ContentPartImageParam],
		use_cache: bool = False,
	) -> str | list[TextBlockParam | ImageBlockParam]:
		"""Serialize content to Anthropic format."""
		if isinstance(content, str):
			if use_cache:
				return [TextBlockParam(text=content, type='text', cache_control=CacheControlEphemeralParam(type='ephemeral'))]
			else:
				return content

		serialized_blocks: list[TextBlockParam | ImageBlockParam] = []
		for part in content:
			if part.type == 'text':
				serialized_blocks.append(AnthropicMessageSerializer._serialize_content_part_text(part, use_cache))
			elif part.type == 'image_url':
				serialized_blocks.append(AnthropicMessageSerializer._serialize_content_part_image(part))

		return serialized_blocks

	@staticmethod
	def _serialize_tool_calls_to_content(tool_calls, use_cache: bool = False) -> list[ToolUseBlockParam]:
		"""Convert tool calls to Anthropic's ToolUseBlockParam format."""
		blocks: list[ToolUseBlockParam] = []
		for tool_call in tool_calls:
			# Parse the arguments JSON string to object

			try:
				input_obj = json.loads(tool_call.function.arguments)
			except json.JSONDecodeError:
				# If arguments aren't valid JSON, use as string
				input_obj = {'arguments': tool_call.function.arguments}

			blocks.append(
				ToolUseBlockParam(
					id=tool_call.id,
					input=input_obj,
					name=tool_call.function.name,
					type='tool_use',
					cache_control=AnthropicMessageSerializer._serialize_cache_control(use_cache),
				)
			)
		return blocks

	# region - Serialize overloads
	@overload
	@staticmethod
	def serialize(message: UserMessage) -> MessageParam: ...

	@overload
	@staticmethod
	def serialize(message: SystemMessage) -> SystemMessage: ...

	@overload
	@staticmethod
	def serialize(message: AssistantMessage) -> MessageParam: ...

	@staticmethod
	def serialize(message: BaseMessage) -> MessageParam | SystemMessage:
		"""Serialize a custom message to an Anthropic MessageParam.

		Note: Anthropic doesn't have a 'system' role. System messages should be
		handled separately as the system parameter in the API call, not as a message.
		If a SystemMessage is passed here, it will be converted to a user message.
		"""
		if isinstance(message, UserMessage):
			content = AnthropicMessageSerializer._serialize_content(message.content, use_cache=message.cache)
			return MessageParam(role='user', content=content)

		elif isinstance(message, SystemMessage):
			# Anthropic doesn't have system messages in the messages array
			# System prompts are passed separately. Convert to user message.
			return message

		elif isinstance(message, AssistantMessage):
			# Handle content and tool calls
			blocks: list[TextBlockParam | ToolUseBlockParam] = []

			# Add content blocks if present
			if message.content is not None:
				if isinstance(message.content, str):
					blocks.append(
						TextBlockParam(
							text=message.content,
							type='text',
							cache_control=AnthropicMessageSerializer._serialize_cache_control(message.cache),
						)
					)
				else:
					# Process content parts (text and refusal)
					for part in message.content:
						if part.type == 'text':
							blocks.append(AnthropicMessageSerializer._serialize_content_part_text(part, use_cache=message.cache))
						# # Note: Anthropic doesn't have a specific refusal block type,
						# # so we convert refusals to text blocks
						# elif part.type == 'refusal':
						# 	blocks.append(TextBlockParam(text=f'[Refusal] {part.refusal}', type='text'))

			# Add tool use blocks if present
			if message.tool_calls:
				tool_blocks = AnthropicMessageSerializer._serialize_tool_calls_to_content(
					message.tool_calls, use_cache=message.cache
				)
				blocks.extend(tool_blocks)

			# If no content or tool calls, add empty text block
			# (Anthropic requires at least one content block)
			if not blocks:
				blocks.append(
					TextBlockParam(
						text='', type='text', cache_control=AnthropicMessageSerializer._serialize_cache_control(message.cache)
					)
				)

			# If caching is enabled or we have multiple blocks, return blocks as-is
			# Otherwise, simplify single text blocks to plain string
			if message.cache or len(blocks) > 1:
				content = blocks
			else:
				# Only simplify when no caching and single block
				single_block = blocks[0]
				if single_block['type'] == 'text' and not single_block.get('cache_control'):
					content = single_block['text']
				else:
					content = blocks

			return MessageParam(
				role='assistant',
				content=content,
			)

		else:
			raise ValueError(f'Unknown message type: {type(message)}')

	@staticmethod
	def _clean_cache_messages(messages: list[NonSystemMessage]) -> list[NonSystemMessage]:
		"""Clean cache settings so only the last cache=True message remains cached.

		Because of how Claude caching works, only the last cache message matters.
		This method automatically removes cache=True from all messages except the last one.

		Args:
			messages: List of non-system messages to clean

		Returns:
			List of messages with cleaned cache settings
		"""
		if not messages:
			return messages

		# Create a copy to avoid modifying the original
		cleaned_messages = [msg.model_copy(deep=True) for msg in messages]

		# Find the last message with cache=True
		last_cache_index = -1
		for i in range(len(cleaned_messages) - 1, -1, -1):
			if cleaned_messages[i].cache:
				last_cache_index = i
				break

		# If we found a cached message, disable cache for all others
		if last_cache_index != -1:
			for i, msg in enumerate(cleaned_messages):
				if i != last_cache_index and msg.cache:
					# Set cache to False for all messages except the last cached one
					msg.cache = False

		return cleaned_messages

	@staticmethod
	def serialize_messages(messages: list[BaseMessage]) -> tuple[list[MessageParam], list[TextBlockParam] | str | None]:
		"""Serialize a list of messages, extracting any system message.

		Returns:
		    A tuple of (messages, system_message) where system_message is extracted
		    from any SystemMessage in the list.
		"""
		messages = [m.model_copy(deep=True) for m in messages]

		# Separate system messages from normal messages
		normal_messages: list[NonSystemMessage] = []
		system_message: SystemMessage | None = None

		for message in messages:
			if isinstance(message, SystemMessage):
				system_message = message
			else:
				normal_messages.append(message)

		# Clean cache messages so only the last cache=True message remains cached
		normal_messages = AnthropicMessageSerializer._clean_cache_messages(normal_messages)

		# Serialize normal messages
		serialized_messages: list[MessageParam] = []
		for message in normal_messages:
			serialized_messages.append(AnthropicMessageSerializer.serialize(message))

		# Serialize system message
		serialized_system_message: list[TextBlockParam] | str | None = None
		if system_message:
			serialized_system_message = AnthropicMessageSerializer._serialize_content_to_str(
				system_message.content, use_cache=system_message.cache
			)

		return serialized_messages, serialized_system_message

import socketserver
from unittest.mock import AsyncMock
import pytest
from pytest_httpserver import HTTPServer
from browser_use.llm import BaseChatModel
from browser_use.sync.service import CloudSync

# From ci/conftest.py
class EventCollector:
		def __init__(self):
			self.events = events
			self.event_order = event_order

		async def collect_event(self, event: BaseEvent):
			self.events.append(event)
			self.event_order.append(event.event_type)
			return 'collected'

		def get_events_by_type(self, event_type: str) -> list[BaseEvent]:
			return [e for e in self.events if e.event_type == event_type]

		def clear(self):
			self.events.clear()
			self.event_order.clear()

# From ci/conftest.py
def setup_test_environment():
	"""
	Automatically set up test environment for all tests.
	"""

	# Create a temporary directory for test config
	config_dir = tempfile.mkdtemp(prefix='browseruse_tests_')

	original_env = {}
	test_env_vars = {
		'SKIP_LLM_API_KEY_VERIFICATION': 'true',
		'ANONYMIZED_TELEMETRY': 'false',
		'BROWSER_USE_CLOUD_SYNC': 'true',
		'BROWSER_USE_CLOUD_API_URL': 'http://placeholder-will-be-replaced-by-specific-test-fixtures',
		'BROWSER_USE_CLOUD_UI_URL': 'http://placeholder-will-be-replaced-by-specific-test-fixtures',
		'BROWSER_USE_CONFIG_DIR': config_dir,
	}

	for key, value in test_env_vars.items():
		original_env[key] = os.environ.get(key)
		os.environ[key] = value

	yield

	# Restore original environment
	for key, value in original_env.items():
		if value is None:
			os.environ.pop(key, None)
		else:
			os.environ[key] = value

# From ci/conftest.py
def create_mock_llm(actions: list[str] | None = None) -> BaseChatModel:
	"""Create a mock LLM that returns specified actions or a default done action.

	Args:
		actions: Optional list of JSON strings representing actions to return in sequence.
			If not provided, returns a single done action.
			After all actions are exhausted, returns a done action.

	Returns:
		Mock LLM that will return the actions in order, or just a done action if no actions provided.
	"""
	controller = Controller()
	ActionModel = controller.registry.create_action_model()
	AgentOutputWithActions = AgentOutput.type_with_custom_actions(ActionModel)

	llm = AsyncMock(spec=BaseChatModel)
	llm.model = 'mock-llm'
	llm._verified_api_keys = True

	# Add missing properties from BaseChatModel protocol
	llm.provider = 'mock'
	llm.name = 'mock-llm'
	llm.model_name = 'mock-llm'  # Ensure this returns a string, not a mock

	# Default done action
	default_done_action = """
	{
		"thinking": "null",
		"evaluation_previous_goal": "Successfully completed the task",
		"memory": "Task completed",
		"next_goal": "Task completed",
		"action": [
			{
				"done": {
					"text": "Task completed successfully",
					"success": true
				}
			}
		]
	}
	"""

	# Unified logic for both cases
	action_index = 0

	def get_next_action() -> str:
		nonlocal action_index
		if actions is not None and action_index < len(actions):
			action = actions[action_index]
			action_index += 1
			return action
		else:
			return default_done_action

	async def mock_ainvoke(*args, **kwargs):
		# Check if output_format is provided (2nd argument or in kwargs)
		output_format = None
		if len(args) >= 2:
			output_format = args[1]
		elif 'output_format' in kwargs:
			output_format = kwargs['output_format']

		action_json = get_next_action()

		if output_format is None:
			# Return string completion
			return ChatInvokeCompletion(completion=action_json, usage=None)
		else:
			# Parse with provided output_format (could be AgentOutputWithActions or another model)
			if output_format == AgentOutputWithActions:
				parsed = AgentOutputWithActions.model_validate_json(action_json)
			else:
				# For other output formats, try to parse the JSON with that model
				parsed = output_format.model_validate_json(action_json)
			return ChatInvokeCompletion(completion=parsed, usage=None)

	llm.ainvoke.side_effect = mock_ainvoke

	return llm

# From ci/conftest.py
def cloud_sync(httpserver: HTTPServer):
	"""
	Create a CloudSync instance configured for testing.

	This fixture creates a real CloudSync instance and sets up the test environment
	to use the httpserver URLs.
	"""

	# Set up test environment
	test_http_server_url = httpserver.url_for('')
	os.environ['BROWSER_USE_CLOUD_API_URL'] = test_http_server_url
	os.environ['BROWSER_USE_CLOUD_UI_URL'] = test_http_server_url
	os.environ['BROWSER_USE_CLOUD_SYNC'] = 'true'

	# Create CloudSync with test server URL
	cloud_sync = CloudSync(
		base_url=test_http_server_url,
		enable_auth=False,  # Disable auth for most tests, they can override this if needed
	)

	return cloud_sync

# From ci/conftest.py
def mock_llm():
	"""Create a mock LLM that just returns the done action if queried"""
	return create_mock_llm(actions=None)

# From ci/conftest.py
def agent_with_cloud(browser_session, mock_llm, cloud_sync):
	"""Create agent with cloud sync enabled (using real CloudSync)."""
	agent = Agent(
		task='Test task',
		llm=mock_llm,
		browser_session=browser_session,
		cloud_sync=cloud_sync,
	)
	return agent

# From ci/conftest.py
def event_collector():
	"""Helper to collect all events emitted during tests"""
	events = []
	event_order = []

	class EventCollector:
		def __init__(self):
			self.events = events
			self.event_order = event_order

		async def collect_event(self, event: BaseEvent):
			self.events.append(event)
			self.event_order.append(event.event_type)
			return 'collected'

		def get_events_by_type(self, event_type: str) -> list[BaseEvent]:
			return [e for e in self.events if e.event_type == event_type]

		def clear(self):
			self.events.clear()
			self.event_order.clear()

	return EventCollector()

# From ci/conftest.py
def get_next_action() -> str:
		nonlocal action_index
		if actions is not None and action_index < len(actions):
			action = actions[action_index]
			action_index += 1
			return action
		else:
			return default_done_action

# From ci/conftest.py
def get_events_by_type(self, event_type: str) -> list[BaseEvent]:
			return [e for e in self.events if e.event_type == event_type]

# From ci/conftest.py
def clear(self):
			self.events.clear()
			self.event_order.clear()

import argparse
import glob
import warnings
import yaml
from browser_use.agent.service import Agent
from browser_use.llm import ChatOpenAI

# From ci/evaluate_tasks.py
class JudgeResponse(BaseModel):
	success: bool
	explanation: str




import aiohttp



from browser_use.mcp.client import MCPClient



# From mcp/advanced_server.py
class TaskResult:
	"""Result of executing a task."""

	success: bool
	data: Any
	error: str | None = None
	timestamp: datetime | None = None

	def __post_init__(self):
		if self.timestamp is None:
			self.timestamp = datetime.now()

# From mcp/advanced_server.py
class AIAssistant:
	"""An AI assistant that uses MCP servers to perform complex tasks."""

	def __init__(self):
		self.servers: dict[str, ClientSession] = {}
		self.tools: dict[str, Tool] = {}
		self.history: list[TaskResult] = []

	async def connect_server(self, name: str, command: str, args: list[str], env: dict[str, str] | None = None):
		"""Connect to an MCP server and discover its tools."""
		print(f'\n🔌 Connecting to {name} server...')

		server_params = StdioServerParameters(command=command, args=args, env=env or {})

		try:
			# Create connection
			read, write = await stdio_client(server_params).__aenter__()
			session = ClientSession(read, write)
			await session.__aenter__()
			await session.initialize()

			# Store session
			self.servers[name] = session

			# Discover tools
			tools_result = await session.list_tools()
			tools = tools_result.tools
			for tool in tools:
				# Prefix tool names with server name to avoid conflicts
				prefixed_name = f'{name}.{tool.name}'
				self.tools[prefixed_name] = tool
				print(f'  ✓ Discovered: {prefixed_name}')

			print(f'✅ Connected to {name} with {len(tools)} tools')

		except Exception as e:
			print(f'❌ Failed to connect to {name}: {e}')
			raise

	async def disconnect_all(self):
		"""Disconnect from all MCP servers."""
		for name, session in self.servers.items():
			try:
				await session.__aexit__(None, None, None)
				print(f'📴 Disconnected from {name}')
			except Exception as e:
				print(f'⚠️ Error disconnecting from {name}: {e}')

	async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> TaskResult:
		"""Call a tool on the appropriate MCP server."""
		# Parse server and tool name
		if '.' not in tool_name:
			return TaskResult(False, None, "Invalid tool name format. Use 'server.tool'")

		server_name, actual_tool_name = tool_name.split('.', 1)

		# Check if server is connected
		if server_name not in self.servers:
			return TaskResult(False, None, f"Server '{server_name}' not connected")

		# Call the tool
		try:
			session = self.servers[server_name]
			result = await session.call_tool(actual_tool_name, arguments)

			# Extract text content
			text_content = [c.text for c in result.content if isinstance(c, TextContent)]
			data = text_content[0] if text_content else str(result.content)

			task_result = TaskResult(True, data)
			self.history.append(task_result)
			return task_result

		except Exception as e:
			error_result = TaskResult(False, None, str(e))
			self.history.append(error_result)
			return error_result

	async def search_and_save(self, query: str, output_file: str) -> TaskResult:
		"""Search for information and save results to a file."""
		print(f'\n🔍 Searching for: {query}')

		# Step 1: Navigate to search engine
		print('  1️⃣ Opening DuckDuckGo...')
		nav_result = await self.call_tool('browser.browser_navigate', {'url': f'https://duckduckgo.com/?q={query}'})
		if not nav_result.success:
			return nav_result

		await asyncio.sleep(2)  # Wait for page load

		# Step 2: Get search results
		print('  2️⃣ Extracting search results...')
		extract_result = await self.call_tool(
			'browser.browser_extract_content',
			{'query': 'Extract the top 5 search results with titles and descriptions', 'extract_links': True},
		)
		if not extract_result.success:
			return extract_result

		# Step 3: Save to file (if filesystem server is connected)
		if 'filesystem' in self.servers:
			print(f'  3️⃣ Saving results to {output_file}...')
			save_result = await self.call_tool(
				'filesystem.write_file',
				{'path': output_file, 'content': f'Search Query: {query}\n\nResults:\n{extract_result.data}'},
			)
			if save_result.success:
				print(f'  ✅ Results saved to {output_file}')
		else:
			print('  ⚠️ Filesystem server not connected, skipping save')

		return extract_result

	async def monitor_page_changes(self, url: str, duration: int = 10, interval: int = 2):
		"""Monitor a webpage for changes over time."""
		print(f'\n📊 Monitoring {url} for {duration} seconds...')

		# Navigate to page
		await self.call_tool('browser.browser_navigate', {'url': url})
		await asyncio.sleep(2)

		changes = []
		start_time = datetime.now()

		while (datetime.now() - start_time).seconds < duration:
			# Get current state
			state_result = await self.call_tool('browser.browser_get_state', {'include_screenshot': False})

			if state_result.success:
				state = json.loads(state_result.data)
				changes.append(
					{
						'timestamp': datetime.now().isoformat(),
						'title': state.get('title', ''),
						'element_count': len(state.get('interactive_elements', [])),
					}
				)
				print(f'  📸 Captured state at {changes[-1]["timestamp"]}')

			await asyncio.sleep(interval)

		return TaskResult(True, changes)

	async def fill_form_workflow(self, form_url: str, form_data: dict[str, str]):
		"""Navigate to a form and fill it out."""
		print(f'\n📝 Form filling workflow for {form_url}')

		# Step 1: Navigate to form
		print('  1️⃣ Navigating to form...')
		nav_result = await self.call_tool('browser.browser_navigate', {'url': form_url})
		if not nav_result.success:
			return nav_result

		await asyncio.sleep(2)

		# Step 2: Get form elements
		print('  2️⃣ Analyzing form elements...')
		state_result = await self.call_tool('browser.browser_get_state', {'include_screenshot': False})

		if not state_result.success:
			return state_result

		state = json.loads(state_result.data)

		# Step 3: Fill form fields
		print('  3️⃣ Filling form fields...')
		filled_fields = []

		for element in state.get('interactive_elements', []):
			# Look for input fields
			if element.get('tag') in ['input', 'textarea']:
				# Try to match field by placeholder or nearby text
				for field_name, field_value in form_data.items():
					element_text = str(element).lower()
					if field_name.lower() in element_text:
						print(f'    ✏️ Filling {field_name}...')
						type_result = await self.call_tool(
							'browser.browser_type', {'index': element['index'], 'text': field_value}
						)
						if type_result.success:
							filled_fields.append(field_name)
						await asyncio.sleep(0.5)
						break

		return TaskResult(True, {'filled_fields': filled_fields, 'form_data': form_data, 'url': form_url})

import gradio
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# From ui/gradio_demo.py
def parse_agent_history(history_str: str) -> None:
	console = Console()

	# Split the content into sections based on ActionResult entries
	sections = history_str.split('ActionResult(')

	for i, section in enumerate(sections[1:], 1):  # Skip first empty section
		# Extract relevant information
		content = ''
		if 'extracted_content=' in section:
			content = section.split('extracted_content=')[1].split(',')[0].strip("'")

		if content:
			header = Text(f'Step {i}', style='bold blue')
			panel = Panel(content, title=header, border_style='blue')
			console.print(panel)
			console.print()

	return None

# From ui/gradio_demo.py
def create_ui():
	with gr.Blocks(title='Browser Use GUI') as interface:
		gr.Markdown('# Browser Use Task Automation')

		with gr.Row():
			with gr.Column():
				api_key = gr.Textbox(label='OpenAI API Key', placeholder='sk-...', type='password')
				task = gr.Textbox(
					label='Task Description',
					placeholder='E.g., Find flights from New York to London for next week',
					lines=3,
				)
				model = gr.Dropdown(choices=['gpt-4', 'gpt-3.5-turbo'], label='Model', value='gpt-4')
				headless = gr.Checkbox(label='Run Headless', value=True)
				submit_btn = gr.Button('Run Task')

			with gr.Column():
				output = gr.Textbox(label='Output', lines=10, interactive=False)

		submit_btn.click(
			fn=lambda *args: asyncio.run(run_browser_task(*args)),
			inputs=[task, api_key, model, headless],
			outputs=output,
		)

	return interface

import streamlit
from browser_use.llm import ChatAnthropic

# From ui/streamlit_demo.py
def initialize_agent(query: str, provider: str):
	llm = get_llm(provider)
	controller = Controller()
	browser_session = BrowserSession()

	return Agent(
		task=query,
		llm=llm,  # type: ignore
		controller=controller,
		browser_session=browser_session,
		use_vision=True,
		max_actions_per_step=1,
	), browser_session


# From ui/command_line.py
def parse_arguments():
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description='Automate browser tasks using an LLM agent.')
	parser.add_argument(
		'--query', type=str, help='The query to process', default='go to reddit and search for posts about browser-use'
	)
	parser.add_argument(
		'--provider',
		type=str,
		choices=['openai', 'anthropic'],
		default='openai',
		help='The model provider to use (default: openai)',
	)
	return parser.parse_args()





from onepassword.client import Client
from browser_use import ActionResult


# From custom-functions/perplexity_search.py
class Person(BaseModel):
	name: str
	email: str | None = None

# From custom-functions/perplexity_search.py
class PersonList(BaseModel):
	people: list[Person]


# From custom-functions/file_upload.py
def create_file(file_type: str = 'txt'):
	with open(f'tmp.{file_type}', 'w') as f:
		f.write('test')
	file_path = Path.cwd() / f'tmp.{file_type}'
	logger.info(f'Created file: {file_path}')
	return str(file_path)


# From custom-functions/hover_element.py
class HoverAction(BaseModel):
	index: int | None = None
	xpath: str | None = None
	selector: str | None = None

from amazoncaptcha import AmazonCaptcha
from browser_use.browser import BrowserConfig

from io import BytesIO

# From custom-functions/cua.py
class OpenAICUAAction(BaseModel):
	"""Parameters for OpenAI Computer Use Assistant action."""

	description: str = Field(..., description='Description of your next goal')


# From custom-functions/drag_and_drop.py
class Position(BaseModel):
	"""Represents a position with x and y coordinates."""

	x: int = Field(..., description='X coordinate')
	y: int = Field(..., description='Y coordinate')

# From custom-functions/drag_and_drop.py
class DragDropAction(BaseModel):
	"""Parameters for drag and drop operations."""

	# Element-based approach
	element_source: str | None = Field(None, description='CSS selector or XPath for the source element to drag')
	element_target: str | None = Field(None, description='CSS selector or XPath for the target element to drop on')
	element_source_offset: Position | None = Field(None, description='Optional offset from source element center (x, y)')
	element_target_offset: Position | None = Field(None, description='Optional offset from target element center (x, y)')

	# Coordinate-based approach
	coord_source_x: int | None = Field(None, description='Source X coordinate for drag start')
	coord_source_y: int | None = Field(None, description='Source Y coordinate for drag start')
	coord_target_x: int | None = Field(None, description='Target X coordinate for drag end')
	coord_target_y: int | None = Field(None, description='Target Y coordinate for drag end')

	# Operation parameters
	steps: int | None = Field(10, description='Number of intermediate steps during drag (default: 10)')
	delay_ms: int | None = Field(5, description='Delay in milliseconds between steps (default: 5)')

import http.client


# From custom-functions/clipboard.py
def copy_to_clipboard(text: str):
	pyperclip.copy(text)
	return ActionResult(extracted_content=text)

from browser_use.agent.service import Controller

# From custom-functions/action_filters.py
def is_login_page(page: Page) -> bool:
	return 'login' in page.url.lower() or 'signin' in page.url.lower()

from mistralai import Mistral

# From custom-functions/extract_pdf_content.py
class PdfExtractParams(BaseModel):
	url: str = Field(description='URL to a PDF document')

# From custom-functions/extract_pdf_content.py
def extract_mistral_ocr(params: PdfExtractParams, browser: BrowserContext) -> ActionResult:
	"""
	Process a PDF URL using Mistral OCR API and return the OCR response.

	Args:
	    url: URL to a PDF document

	Returns:
	    OCR response object from Mistral API
	"""
	api_key = os.getenv('MISTRAL_API_KEY')
	client = Mistral(api_key=api_key)

	response = client.ocr.process(
		model='mistral-ocr-latest',
		document={
			'type': 'document_url',
			'document_url': params.url,
		},
		include_image_base64=False,
	)

	markdown = '\n\n'.join(f'### Page {i + 1}\n{response.pages[i].markdown}' for i in range(len(response.pages)))
	return ActionResult(
		extracted_content=markdown,
		include_in_memory=False,  ## PDF content can be very large, so we don't include it in memory
	)

import pyotp

import prettyprinter
from fastapi import FastAPI
from fastapi import Request
import requests
from pyobjtojson import obj_to_json
import uvicorn

# From custom-functions/custom_hooks_before_after_step.py
def b64_to_png(b64_string: str, output_file):
	"""
	Convert a Base64-encoded string to a PNG file.

	:param b64_string: A string containing Base64-encoded data
	:param output_file: The path to the output PNG file
	"""
	with open(output_file, 'wb') as f:
		f.write(base64.b64decode(b64_string))

# From custom-functions/custom_hooks_before_after_step.py
def send_agent_history_step(data):
	url = 'http://127.0.0.1:9000/post_agent_history_step'
	response = requests.post(url, json=data)
	return response.json()


# From custom-functions/save_to_file_hugging_face.py
class Model(BaseModel):
	title: str
	url: str
	likes: int
	license: str

# From custom-functions/save_to_file_hugging_face.py
class Models(BaseModel):
	models: list[Model]

# From custom-functions/save_to_file_hugging_face.py
def save_models(params: Models):
	with open('models.txt', 'a') as f:
		for model in params.models:
			f.write(f'{model.title} ({model.url}): {model.likes} likes, {model.license}\n')


import yagmail

import pathlib



from browser_use.llm import ChatGroq


from browser_use.llm import ChatAnthropicBedrock
from browser_use.llm import ChatAWSBedrock



from browser_use.llm import ChatAzureOpenAI

from browser_use.llm import ChatDeepSeek



from browser_use.llm import ChatGoogle


# From features/custom_output.py
class Post(BaseModel):
	post_title: str
	post_url: str
	num_comments: int
	hours_since_post: int

# From features/custom_output.py
class Posts(BaseModel):
	posts: list[Post]




# From features/validate_output.py
class DoneResult(BaseModel):
	title: str
	comments: str
	hours_since_start: int



import threading

# From features/pause_agent.py
class AgentController:
	def __init__(self):
		llm = ChatOpenAI(model='gpt-4.1')
		self.agent = Agent(
			task='open in one action https://www.google.com, https://www.wikipedia.org, https://www.youtube.com, https://www.github.com, https://amazon.com',
			llm=llm,
		)
		self.running = False

	async def run_agent(self):
		"""Run the agent"""
		self.running = True
		await self.agent.run()

	def start(self):
		"""Start the agent in a separate thread"""
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		loop.run_until_complete(self.run_agent())

	def pause(self):
		"""Pause the agent"""
		self.agent.pause()

	def resume(self):
		"""Resume the agent"""
		self.agent.resume()

	def stop(self):
		"""Stop the agent"""
		self.agent.stop()
		self.running = False

# From features/pause_agent.py
def print_menu():
	print('\nAgent Control Menu:')
	print('1. Start')
	print('2. Pause')
	print('3. Resume')
	print('4. Stop')
	print('5. Exit')

# From features/pause_agent.py
def start(self):
		"""Start the agent in a separate thread"""
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		loop.run_until_complete(self.run_agent())





from pprint import pprint





from aiohttp import web





from stagehand import Stagehand
from stagehand import StagehandConfig

from browser_use.integrations.gmail import GmailService
from browser_use.integrations.gmail import register_gmail_actions

# From integrations/gmail_2fa_integration.py
class GmailGrantManager:
	"""
	Manages Gmail OAuth credential grants and authentication flows.
	Provides a robust mechanism for setting up and maintaining Gmail API access.
	"""

	def __init__(self):
		self.config_dir = CONFIG.BROWSER_USE_CONFIG_DIR
		self.credentials_file = self.config_dir / 'gmail_credentials.json'
		self.token_file = self.config_dir / 'gmail_token.json'
		print(f'GmailGrantManager initialized with config_dir: {self.config_dir}')
		print(f'GmailGrantManager initialized with credentials_file: {self.credentials_file}')
		print(f'GmailGrantManager initialized with token_file: {self.token_file}')

	def check_credentials_exist(self) -> bool:
		"""Check if OAuth credentials file exists."""
		return self.credentials_file.exists()

	def check_token_exists(self) -> bool:
		"""Check if saved token file exists."""
		return self.token_file.exists()

	def validate_credentials_format(self) -> tuple[bool, str]:
		"""
		Validate that the credentials file has the correct format.
		Returns (is_valid, error_message)
		"""
		if not self.check_credentials_exist():
			return False, 'Credentials file not found'

		try:
			with open(self.credentials_file) as f:
				creds = json.load(f)

			required_fields = ['web']
			web = creds['web']
			if not web:
				return False, "Invalid credentials format - missing 'web' section"

			return True, 'Credentials file is valid'

		except json.JSONDecodeError:
			return False, 'Credentials file is not valid JSON'
		except Exception as e:
			return False, f'Error reading credentials file: {e}'

	async def setup_oauth_credentials(self) -> bool:
		"""
		Guide user through OAuth credentials setup process.
		Returns True if setup is successful.
		"""
		print('\n🔐 Gmail OAuth Credentials Setup Required')
		print('=' * 50)

		if not self.check_credentials_exist():
			print('❌ Gmail credentials file not found')
		else:
			is_valid, error = self.validate_credentials_format()
			if not is_valid:
				print(f'❌ Gmail credentials file is invalid: {error}')

		print('\n📋 To set up Gmail API access:')
		print('1. Go to https://console.cloud.google.com/')
		print('2. Create a new project or select an existing one')
		print('3. Enable the Gmail API:')
		print('   - Go to "APIs & Services" > "Library"')
		print('   - Search for "Gmail API" and enable it')
		print('4. Create OAuth 2.0 credentials:')
		print('   - Go to "APIs & Services" > "Credentials"')
		print('   - Click "Create Credentials" > "OAuth client ID"')
		print('   - Choose "Desktop application"')
		print('   - Download the JSON file')
		print(f'5. Save the JSON file as: {self.credentials_file}')
		print(f'6. Ensure the directory exists: {self.config_dir}')

		# Create config directory if it doesn't exist
		self.config_dir.mkdir(parents=True, exist_ok=True)
		print(f'\n✅ Created config directory: {self.config_dir}')

		# Wait for user to set up credentials
		while True:
			user_input = input('\n❓ Have you saved the credentials file? (y/n/skip): ').lower().strip()

			if user_input == 'skip':
				print('⏭️  Skipping credential validation for now')
				return False
			elif user_input == 'y':
				if self.check_credentials_exist():
					is_valid, error = self.validate_credentials_format()
					if is_valid:
						print('✅ Credentials file found and validated!')
						return True
					else:
						print(f'❌ Credentials file is invalid: {error}')
						print('Please check the file format and try again.')
				else:
					print(f'❌ Credentials file still not found at: {self.credentials_file}')
			elif user_input == 'n':
				print('⏳ Please complete the setup steps above and try again.')
			else:
				print('Please enter y, n, or skip')

	async def test_authentication(self, gmail_service: GmailService) -> tuple[bool, str]:
		"""
		Test Gmail authentication and return status.
		Returns (success, message)
		"""
		try:
			print('🔍 Testing Gmail authentication...')
			success = await gmail_service.authenticate()

			if success and gmail_service.is_authenticated():
				print('✅ Gmail authentication successful!')
				return True, 'Authentication successful'
			else:
				return False, 'Authentication failed - invalid credentials or OAuth flow failed'

		except Exception as e:
			return False, f'Authentication error: {e}'

	async def handle_authentication_failure(self, gmail_service: GmailService, error_msg: str) -> bool:
		"""
		Handle authentication failures with fallback mechanisms.
		Returns True if recovery was successful.
		"""
		print(f'\n❌ Gmail authentication failed: {error_msg}')
		print('\n🔧 Attempting recovery...')

		# Option 1: Try removing old token file
		if self.token_file.exists():
			print('🗑️  Removing old token file to force re-authentication...')
			try:
				self.token_file.unlink()
				print('✅ Old token file removed')

				# Try authentication again
				success = await gmail_service.authenticate()
				if success:
					print('✅ Re-authentication successful!')
					return True
			except Exception as e:
				print(f'❌ Failed to remove token file: {e}')

		# Option 2: Validate and potentially re-setup credentials
		is_valid, cred_error = self.validate_credentials_format()
		if not is_valid:
			print(f'\n❌ Credentials file issue: {cred_error}')
			print('🔧 Initiating credential re-setup...')

			return await self.setup_oauth_credentials()

		# Option 3: Provide manual troubleshooting steps
		print('\n🔧 Manual troubleshooting steps:')
		print('1. Check that Gmail API is enabled in Google Cloud Console')
		print('2. Verify OAuth consent screen is configured')
		print('3. Ensure redirect URIs include http://localhost:8080')
		print('4. Check if credentials file is for the correct project')
		print('5. Try regenerating OAuth credentials in Google Cloud Console')

		retry = input('\n❓ Would you like to retry authentication? (y/n): ').lower().strip()
		if retry == 'y':
			success = await gmail_service.authenticate()
			return success

		return False

# From integrations/gmail_2fa_integration.py
def check_credentials_exist(self) -> bool:
		"""Check if OAuth credentials file exists."""
		return self.credentials_file.exists()

# From integrations/gmail_2fa_integration.py
def check_token_exists(self) -> bool:
		"""Check if saved token file exists."""
		return self.token_file.exists()

# From integrations/gmail_2fa_integration.py
def validate_credentials_format(self) -> tuple[bool, str]:
		"""
		Validate that the credentials file has the correct format.
		Returns (is_valid, error_message)
		"""
		if not self.check_credentials_exist():
			return False, 'Credentials file not found'

		try:
			with open(self.credentials_file) as f:
				creds = json.load(f)

			required_fields = ['web']
			web = creds['web']
			if not web:
				return False, "Invalid credentials format - missing 'web' section"

			return True, 'Credentials file is valid'

		except json.JSONDecodeError:
			return False, 'Credentials file is not valid JSON'
		except Exception as e:
			return False, f'Error reading credentials file: {e}'

import csv
from PyPDF2 import PdfReader

# From use-cases/find_and_apply_to_jobs.py
class Job(BaseModel):
	title: str
	link: str
	company: str
	fit_score: float
	location: str | None = None
	salary: str | None = None

# From use-cases/find_and_apply_to_jobs.py
def save_jobs(job: Job):
	with open('jobs.csv', 'a', newline='') as f:
		writer = csv.writer(f)
		writer.writerow([job.title, job.company, job.link, job.salary, job.location])

	return 'Saved job to file'

# From use-cases/find_and_apply_to_jobs.py
def read_jobs():
	with open('jobs.csv') as f:
		return f.read()

# From use-cases/find_and_apply_to_jobs.py
def read_cv():
	pdf = PdfReader(CV)
	text = ''
	for page in pdf.pages:
		text += page.extract_text() or ''
	logger.info(f'Read cv with {len(text)} characters')
	return ActionResult(extracted_content=text, include_in_memory=True)


import chess
from bs4 import BeautifulSoup

# From use-cases/play_chess.py
class PlayMoveParams(BaseModel):
	move: str = Field(
		description="The move in Standard Algebraic Notation (SAN) exactly as provided in the 'Legal Moves' list (e.g., 'Nf3', 'e4', 'Qh7#')."
	)

# From use-cases/play_chess.py
def to_px(val: float) -> str:
	"""Convert float to px string, e.g. 42.0 -> '42px'."""
	s = f'{val:.1f}'.rstrip('0').rstrip('.')
	return f'{s}px'

# From use-cases/play_chess.py
def from_px(px: str) -> float:
	"""Convert px string to float, e.g. '42px' -> 42.0."""
	return float(px.replace('px', '').strip())

# From use-cases/play_chess.py
def parse_transform(style: str) -> tuple[float, float] | None:
	"""Extracts x and y pixel coordinates from a CSS transform string."""
	try:
		parts = style.split('(')[1].split(')')[0].split(',')
		x_px_str = float(parts[0].strip().replace('px', ''))
		y_px_str = float(parts[1].strip().replace('px', ''))
		return x_px_str, y_px_str
	except Exception as e:
		logger.error(f'Error parsing transform style: {e}')
		return None

# From use-cases/play_chess.py
def algebraic_to_pixels(square: str, square_size: float) -> tuple[str, str]:
	"""Converts algebraic notation to Lichess pixel coordinates using dynamic size."""
	file_char = square[0].lower()
	rank_char = square[1]

	if file_char not in FILES or rank_char not in RANKS:
		raise ValueError(f'Invalid square: {square}')

	x_index = FILES.index(file_char)
	y_index = RANKS.index(rank_char)

	x_px = x_index * square_size
	y_px = y_index * square_size
	return to_px(x_px), to_px(y_px)

# From use-cases/play_chess.py
def pixels_to_algebraic(x_px: float, y_px: float, square_size: float) -> str:
	"""Converts Lichess pixel coordinates to algebraic notation using dynamic size."""
	if not square_size:
		raise ValueError('Square size cannot be zero or None.')

	x_index = int(round(x_px / square_size))
	y_index = int(round(y_px / square_size))

	if 0 <= x_index < 8 and 0 <= y_index < 8:
		return f'{FILES[x_index]}{RANKS[y_index]}'

	raise ValueError(f'Pixel coordinates out of bounds: ({x_px}, {y_px})')

# From use-cases/play_chess.py
def get_piece_symbol(class_list: list[str]) -> str:
	color = class_list[0]
	ptype = class_list[1]
	symbols = {'king': 'k', 'queen': 'q', 'rook': 'r', 'bishop': 'b', 'knight': 'n', 'pawn': 'p'}
	symbol = symbols.get(ptype, '?')
	return symbol.upper() if color == 'white' else symbol

# From use-cases/play_chess.py
def create_fen_board(board_state: dict) -> str:
	fen = ''
	for rank_num in RANKS:
		empty_count = 0
		for file_char in FILES:
			square = f'{file_char}{rank_num}'
			if square in board_state:
				if empty_count > 0:
					fen += str(empty_count)
					empty_count = 0
				fen += board_state[square]
			else:
				empty_count += 1
		if empty_count > 0:
			fen += str(empty_count)
		if rank_num != RANKS[-1]:
			fen += '/'
	return fen


# From use-cases/post-twitter.py
class TwitterConfig:
	"""Configuration for Twitter posting"""

	openai_api_key: str
	chrome_path: str
	target_user: str  # Twitter handle without @
	message: str
	reply_url: str
	headless: bool = False
	model: str = 'gpt-4.1-mini'
	base_url: str = 'https://x.com/home'

# From use-cases/post-twitter.py
def create_twitter_agent(config: TwitterConfig) -> Agent:
	llm = ChatOpenAI(model=config.model, api_key=config.openai_api_key)

	browser_profile = BrowserProfile(
		headless=config.headless,
		executable_path=config.chrome_path,
	)
	browser_session = BrowserSession(browser_profile=browser_profile)

	controller = Controller()

	# Construct the full message with tag
	full_message = f'@{config.target_user} {config.message}'

	# Create the agent with detailed instructions
	agent = Agent(
		task=f"""Navigate to Twitter and create a post and reply to a tweet.

        Here are the specific steps:

        1. Go to {config.base_url}. See the text input field at the top of the page that says "What's happening?"
        2. Look for the text input field at the top of the page that says "What's happening?"
        3. Click the input field and type exactly this message:
        "{full_message}"
        4. Find and click the "Post" button (look for attributes: 'button' and 'data-testid="tweetButton"')
        5. Do not click on the '+' button which will add another tweet.

        6. Navigate to {config.reply_url}
        7. Before replying, understand the context of the tweet by scrolling down and reading the comments.
        8. Reply to the tweet under 50 characters.

        Important:
        - Wait for each element to load before interacting
        - Make sure the message is typed exactly as shown
        - Verify the post button is clickable before clicking
        - Do not click on the '+' button which will add another tweet
        """,
		llm=llm,
		controller=controller,
		browser_session=browser_session,
	)
	return agent







# From use-cases/find_influencer_profiles.py
class Profile(BaseModel):
	platform: str
	profile_url: str

# From use-cases/find_influencer_profiles.py
class Profiles(BaseModel):
	profiles: list[Profile]



# From use-cases/check_appointment.py
class WebpageInfo(BaseModel):
	"""Model for webpage link."""

	link: str = 'https://appointment.mfa.gr/en/reservations/aero/ireland-grcon-dub/'

# From use-cases/check_appointment.py
def go_to_webpage(webpage_info: WebpageInfo):
	"""Returns the webpage link."""
	return webpage_info.link




from imgcat import imgcat



# From browser/window_sizing.py
def validate_window_size(configured: dict[str, Any], actual: dict[str, Any]) -> None:
	"""Compare configured window size with actual size and report differences.

	Raises:
		Exception: If the window size difference exceeds tolerance
	"""
	# Allow for small differences due to browser chrome, scrollbars, etc.
	width_diff = abs(configured['width'] - actual['width'])
	height_diff = abs(configured['height'] - actual['height'])

	# Tolerance of 5% or 20px, whichever is greater
	width_tolerance = max(configured['width'] * 0.05, 20)
	height_tolerance = max(configured['height'] * 0.05, 20)

	if width_diff > width_tolerance or height_diff > height_tolerance:
		print(f'⚠️  WARNING: Significant difference between expected and actual page size! ±{width_diff}x{height_diff}px')
		raise Exception('Window size validation failed')
	else:
		print('✅ Window size validation passed: actual size matches configured size within tolerance')

	return None

from examples.models.langchain.serializer import LangChainMessageSerializer
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

# From langchain/chat.py
class ChatLangchain(BaseChatModel):
	"""
	A wrapper around LangChain BaseChatModel that implements the browser-use BaseChatModel protocol.

	This class allows you to use any LangChain-compatible model with browser-use.
	"""

	# The LangChain model to wrap
	chat: 'LangChainBaseChatModel'

	@property
	def model(self) -> str:
		return self.name

	@property
	def provider(self) -> str:
		"""Return the provider name based on the LangChain model class."""
		model_class_name = self.chat.__class__.__name__.lower()
		if 'openai' in model_class_name:
			return 'openai'
		elif 'anthropic' in model_class_name or 'claude' in model_class_name:
			return 'anthropic'
		elif 'google' in model_class_name or 'gemini' in model_class_name:
			return 'google'
		elif 'groq' in model_class_name:
			return 'groq'
		elif 'ollama' in model_class_name:
			return 'ollama'
		elif 'deepseek' in model_class_name:
			return 'deepseek'
		else:
			return 'langchain'

	@property
	def name(self) -> str:
		"""Return the model name."""
		# Try to get model name from the LangChain model using getattr to avoid type errors
		model_name = getattr(self.chat, 'model_name', None)
		if model_name:
			return str(model_name)

		model_attr = getattr(self.chat, 'model', None)
		if model_attr:
			return str(model_attr)

		return self.chat.__class__.__name__

	def _get_usage(self, response: 'LangChainAIMessage') -> ChatInvokeUsage | None:
		usage = response.usage_metadata
		if usage is None:
			return None

		prompt_tokens = usage['input_tokens'] or 0
		completion_tokens = usage['output_tokens'] or 0
		total_tokens = usage['total_tokens'] or 0

		input_token_details = usage.get('input_token_details', None)

		if input_token_details is not None:
			prompt_cached_tokens = input_token_details.get('cache_read', None)
			prompt_cache_creation_tokens = input_token_details.get('cache_creation', None)
		else:
			prompt_cached_tokens = None
			prompt_cache_creation_tokens = None

		return ChatInvokeUsage(
			prompt_tokens=prompt_tokens,
			prompt_cached_tokens=prompt_cached_tokens,
			prompt_cache_creation_tokens=prompt_cache_creation_tokens,
			prompt_image_tokens=None,
			completion_tokens=completion_tokens,
			total_tokens=total_tokens,
		)

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		"""
		Invoke the LangChain model with the given messages.

		Args:
			messages: List of browser-use chat messages
			output_format: Optional Pydantic model class for structured output (not supported in basic LangChain integration)

		Returns:
			Either a string response or an instance of output_format
		"""

		# Convert browser-use messages to LangChain messages
		langchain_messages = LangChainMessageSerializer.serialize_messages(messages)

		try:
			if output_format is None:
				# Return string response
				response = await self.chat.ainvoke(langchain_messages)  # type: ignore

				# Import at runtime for isinstance check
				from langchain_core.messages import AIMessage as LangChainAIMessage  # type: ignore

				if not isinstance(response, LangChainAIMessage):
					raise ModelProviderError(
						message=f'Response is not an AIMessage: {type(response)}',
						model=self.name,
					)

				# Extract content from LangChain response
				content = response.content if hasattr(response, 'content') else str(response)

				usage = self._get_usage(response)
				return ChatInvokeCompletion(
					completion=str(content),
					usage=usage,
				)

			else:
				# Use LangChain's structured output capability
				try:
					structured_chat = self.chat.with_structured_output(output_format)
					parsed_object = await structured_chat.ainvoke(langchain_messages)

					# For structured output, usage metadata is typically not available
					# in the parsed object since it's a Pydantic model, not an AIMessage
					usage = None

					# Type cast since LangChain's with_structured_output returns the correct type
					return ChatInvokeCompletion(
						completion=parsed_object,  # type: ignore
						usage=usage,
					)
				except AttributeError:
					# Fall back to manual parsing if with_structured_output is not available
					response = await self.chat.ainvoke(langchain_messages)  # type: ignore

					if not isinstance(response, 'LangChainAIMessage'):
						raise ModelProviderError(
							message=f'Response is not an AIMessage: {type(response)}',
							model=self.name,
						)

					content = response.content if hasattr(response, 'content') else str(response)

					try:
						if isinstance(content, str):
							import json

							parsed_data = json.loads(content)
							if isinstance(parsed_data, dict):
								parsed_object = output_format(**parsed_data)
							else:
								raise ValueError('Parsed JSON is not a dictionary')
						else:
							raise ValueError('Content is not a string and structured output not supported')
					except Exception as e:
						raise ModelProviderError(
							message=f'Failed to parse response as {output_format.__name__}: {e}',
							model=self.name,
						) from e

					usage = self._get_usage(response)
					return ChatInvokeCompletion(
						completion=parsed_object,
						usage=usage,
					)

		except Exception as e:
			# Convert any LangChain errors to browser-use ModelProviderError
			raise ModelProviderError(
				message=f'LangChain model error: {str(e)}',
				model=self.name,
			) from e

# From langchain/chat.py
def model(self) -> str:
		return self.name

from langchain_openai import ChatOpenAI
from examples.models.langchain.chat import ChatLangchain

from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolCall
from langchain_core.messages.base import BaseMessage

# From langchain/serializer.py
class LangChainMessageSerializer:
	"""Serializer for converting between browser-use message types and LangChain message types."""

	@staticmethod
	def _serialize_user_content(
		content: str | list[ContentPartTextParam | ContentPartImageParam],
	) -> str | list[str | dict]:
		"""Convert user message content for LangChain compatibility."""
		if isinstance(content, str):
			return content

		serialized_parts = []
		for part in content:
			if part.type == 'text':
				serialized_parts.append(
					{
						'type': 'text',
						'text': part.text,
					}
				)
			elif part.type == 'image_url':
				# LangChain format for images
				serialized_parts.append(
					{'type': 'image_url', 'image_url': {'url': part.image_url.url, 'detail': part.image_url.detail}}
				)

		return serialized_parts

	@staticmethod
	def _serialize_system_content(
		content: str | list[ContentPartTextParam],
	) -> str:
		"""Convert system message content to text string for LangChain compatibility."""
		if isinstance(content, str):
			return content

		text_parts = []
		for part in content:
			if part.type == 'text':
				text_parts.append(part.text)

		return '\n'.join(text_parts)

	@staticmethod
	def _serialize_assistant_content(
		content: str | list[ContentPartTextParam | ContentPartRefusalParam] | None,
	) -> str:
		"""Convert assistant message content to text string for LangChain compatibility."""
		if content is None:
			return ''
		if isinstance(content, str):
			return content

		text_parts = []
		for part in content:
			if part.type == 'text':
				text_parts.append(part.text)
			# elif part.type == 'refusal':
			# 	# Include refusal content as text
			# 	text_parts.append(f'[Refusal: {part.refusal}]')

		return '\n'.join(text_parts)

	@staticmethod
	def _serialize_tool_call(tool_call: ToolCall) -> LangChainToolCall:
		"""Convert browser-use ToolCall to LangChain ToolCall."""
		# Parse the arguments string to a dict for LangChain
		try:
			args_dict = json.loads(tool_call.function.arguments)
		except json.JSONDecodeError:
			# If parsing fails, wrap in a dict
			args_dict = {'arguments': tool_call.function.arguments}

		return LangChainToolCall(
			name=tool_call.function.name,
			args=args_dict,
			id=tool_call.id,
		)

	# region - Serialize overloads
	@overload
	@staticmethod
	def serialize(message: UserMessage) -> HumanMessage: ...

	@overload
	@staticmethod
	def serialize(message: BrowserUseSystemMessage) -> SystemMessage: ...

	@overload
	@staticmethod
	def serialize(message: AssistantMessage) -> AIMessage: ...

	@staticmethod
	def serialize(message: BaseMessage) -> LangChainBaseMessage:
		"""Serialize a browser-use message to a LangChain message."""

		if isinstance(message, UserMessage):
			content = LangChainMessageSerializer._serialize_user_content(message.content)
			return HumanMessage(content=content, name=message.name)

		elif isinstance(message, BrowserUseSystemMessage):
			content = LangChainMessageSerializer._serialize_system_content(message.content)
			return SystemMessage(content=content, name=message.name)

		elif isinstance(message, AssistantMessage):
			# Handle content
			content = LangChainMessageSerializer._serialize_assistant_content(message.content)

			# For simplicity, we'll ignore tool calls in LangChain integration
			# as requested by the user
			return AIMessage(
				content=content,
				name=message.name,
			)

		else:
			raise ValueError(f'Unknown message type: {type(message)}')

	@staticmethod
	def serialize_messages(messages: list[BaseMessage]) -> list[LangChainBaseMessage]:
		"""Serialize a list of browser-use messages to LangChain messages."""
		return [LangChainMessageSerializer.serialize(m) for m in messages]

from examples.integrations.slack.slack_api import SlackBot
from examples.integrations.slack.slack_api import app

from typing import Annotated
from fastapi import Depends
from fastapi import HTTPException
from slack_sdk.errors import SlackApiError
from slack_sdk.signature import SignatureVerifier
from slack_sdk.web.async_client import AsyncWebClient

# From slack/slack_api.py
class SlackBot:
	def __init__(
		self,
		llm: BaseChatModel,
		bot_token: str,
		signing_secret: str,
		ack: bool = False,
		browser_profile: BrowserProfile = BrowserProfile(headless=True),
	):
		if not bot_token or not signing_secret:
			raise ValueError('Bot token and signing secret must be provided')

		self.llm = llm
		self.ack = ack
		self.browser_profile = browser_profile
		self.client = AsyncWebClient(token=bot_token)
		self.signature_verifier = SignatureVerifier(signing_secret)
		self.processed_events = set()
		logger.info('SlackBot initialized')

	async def handle_event(self, event, event_id):
		try:
			logger.info(f'Received event id: {event_id}')
			if not event_id:
				logger.warning('Event ID missing in event data')
				return

			if event_id in self.processed_events:
				logger.info(f'Event {event_id} already processed')
				return
			self.processed_events.add(event_id)

			if 'subtype' in event and event['subtype'] == 'bot_message':
				return

			text = event.get('text')
			user_id = event.get('user')
			if text and text.startswith('$bu '):
				task = text[len('$bu ') :].strip()
				if self.ack:
					try:
						await self.send_message(
							event['channel'], f'<@{user_id}> Starting browser use task...', thread_ts=event.get('ts')
						)
					except Exception as e:
						logger.error(f'Error sending start message: {e}')

				try:
					agent_message = await self.run_agent(task)
					await self.send_message(event['channel'], f'<@{user_id}> {agent_message}', thread_ts=event.get('ts'))
				except Exception as e:
					await self.send_message(event['channel'], f'Error during task execution: {str(e)}', thread_ts=event.get('ts'))
		except Exception as e:
			logger.error(f'Error in handle_event: {str(e)}')

	async def run_agent(self, task: str) -> str:
		try:
			browser_session = BrowserSession(browser_profile=self.browser_profile)
			agent = Agent(task=task, llm=self.llm, browser_session=browser_session)
			result = await agent.run()

			agent_message = None
			if result.is_done():
				agent_message = result.history[-1].result[0].extracted_content

			if agent_message is None:
				agent_message = 'Oops! Something went wrong while running Browser-Use.'

			return agent_message

		except Exception as e:
			logger.error(f'Error during task execution: {str(e)}')
			return f'Error during task execution: {str(e)}'

	async def send_message(self, channel, text, thread_ts=None):
		try:
			await self.client.chat_postMessage(channel=channel, text=text, thread_ts=thread_ts)
		except SlackApiError as e:
			logger.error(f'Error sending message: {e.response["error"]}')

import discord
from discord.ext import commands

# From discord/discord_api.py
class DiscordBot(commands.Bot):
	"""Discord bot implementation for Browser-Use tasks.

	This bot allows users to run browser automation tasks through Discord messages.
	Processes tasks asynchronously and sends the result back to the user in response to the message.
	Messages must start with the configured prefix (default: "$bu") followed by the task description.

	Args:
	    llm (BaseChatModel): Language model instance to use for task processing
	    prefix (str, optional): Command prefix for triggering browser tasks. Defaults to "$bu"
	    ack (bool, optional): Whether to acknowledge task receipt with a message. Defaults to False
	    browser_profile (BrowserProfile, optional): Browser profile settings.
	        Defaults to headless mode

	Usage:
	    ```python
	    from browser_use.llm import ChatOpenAI

	    llm = ChatOpenAI()
	    bot = DiscordBot(llm=llm, prefix='$bu', ack=True)
	    bot.run('YOUR_DISCORD_TOKEN')
	    ```

	Discord Usage:
	    Send messages starting with the prefix:
	    "$bu search for python tutorials"
	"""

	def __init__(
		self,
		llm: BaseChatModel,
		prefix: str = '$bu',
		ack: bool = False,
		browser_profile: BrowserProfile = BrowserProfile(headless=True),
	):
		self.llm = llm
		self.prefix = prefix.strip()
		self.ack = ack
		self.browser_profile = browser_profile

		# Define intents.
		intents = discord.Intents.default()  # type: ignore
		intents.message_content = True  # Enable message content intent
		intents.members = True  # Enable members intent for user info

		# Initialize the bot with a command prefix and intents.
		super().__init__(command_prefix='!', intents=intents)  # You may not need prefix, just here for flexibility

		# self.tree = app_commands.CommandTree(self) # Initialize command tree for slash commands.

	async def on_ready(self):
		"""Called when the bot is ready."""
		try:
			print(f'We have logged in as {self.user}')
			cmds = await self.tree.sync()  # Sync the command tree with discord

		except Exception as e:
			print(f'Error during bot startup: {e}')

	async def on_message(self, message):
		"""Called when a message is received."""
		try:
			if message.author == self.user:  # Ignore the bot's messages
				return
			if message.content.strip().startswith(f'{self.prefix} '):
				if self.ack:
					try:
						await message.reply(
							'Starting browser use task...',
							mention_author=True,  # Don't ping the user
						)
					except Exception as e:
						print(f'Error sending start message: {e}')

				try:
					agent_message = await self.run_agent(message.content.replace(f'{self.prefix} ', '').strip())
					await message.channel.send(content=f'{agent_message}', reference=message, mention_author=True)
				except Exception as e:
					await message.channel.send(
						content=f'Error during task execution: {str(e)}',
						reference=message,
						mention_author=True,
					)

		except Exception as e:
			print(f'Error in message handling: {e}')

	#    await self.process_commands(message)  # Needed to process bot commands

	async def run_agent(self, task: str) -> str:
		try:
			browser_session = BrowserSession(browser_profile=self.browser_profile)
			agent = Agent(task=(task), llm=self.llm, browser_session=browser_session)
			result = await agent.run()

			agent_message = None
			if result.is_done():
				agent_message = result.history[-1].result[0].extracted_content

			if agent_message is None:
				agent_message = 'Oops! Something went wrong while running Browser-Use.'

			return agent_message

		except Exception as e:
			raise Exception(f'Browser-use task failed: {str(e)}')

from examples.integrations.discord.discord_api import DiscordBot

