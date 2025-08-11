#!/usr/bin/env python3
"""
Unify code across repositories within each functional category.

This script creates unified interfaces and adapters for each functional category,
allowing users to access functionality from any repository through a consistent API.
"""

import os
import sys
import importlib
import inspect
import shutil
from pathlib import Path
import ast
import re

# Base directories
FUNCTIONS_DIR = Path('functions')

# Define the functional categories
CATEGORIES = [
    'agent_frameworks',
    'user_interfaces',
    'os_interaction',
    'browser_automation',
    'code_execution',
    'cognitive_systems',
    'evolution_optimization',
    'integration',
    'nlp',
    'documentation'
]

# Define the unified interface for each category
UNIFIED_INTERFACES = {
    'agent_frameworks': {
        'name': 'UnifiedAgent',
        'methods': [
            'initialize',
            'plan',
            'execute',
            'run',
            'add_tool',
            'add_memory',
            'get_state'
        ],
        'base_class': 'Agent'
    },
    'user_interfaces': {
        'name': 'UnifiedUI',
        'methods': [
            'initialize',
            'render',
            'handle_input',
            'update',
            'get_state'
        ],
        'base_class': 'UserInterface'
    },
    'os_interaction': {
        'name': 'UnifiedOSInteraction',
        'methods': [
            'initialize',
            'capture_screen',
            'control_keyboard',
            'control_mouse',
            'execute_command',
            'get_state'
        ],
        'base_class': 'OSInteraction'
    },
    'browser_automation': {
        'name': 'UnifiedBrowser',
        'methods': [
            'initialize',
            'navigate',
            'click',
            'type',
            'extract_content',
            'get_state'
        ],
        'base_class': 'BrowserAutomation'
    },
    'code_execution': {
        'name': 'UnifiedCodeExecutor',
        'methods': [
            'initialize',
            'execute',
            'analyze',
            'generate',
            'get_state'
        ],
        'base_class': 'CodeExecutor'
    },
    'cognitive_systems': {
        'name': 'UnifiedCognitiveSystem',
        'methods': [
            'initialize',
            'process',
            'learn',
            'reason',
            'get_state'
        ],
        'base_class': 'CognitiveSystem'
    },
    'evolution_optimization': {
        'name': 'UnifiedEvolutionOptimizer',
        'methods': [
            'initialize',
            'evolve',
            'evaluate',
            'select',
            'get_state'
        ],
        'base_class': 'EvolutionOptimizer'
    },
    'integration': {
        'name': 'UnifiedIntegration',
        'methods': [
            'initialize',
            'connect',
            'process',
            'transform',
            'get_state'
        ],
        'base_class': 'Integration'
    },
    'nlp': {
        'name': 'UnifiedNLP',
        'methods': [
            'initialize',
            'process_text',
            'generate_text',
            'analyze_text',
            'get_state'
        ],
        'base_class': 'NLPProcessor'
    },
    'documentation': {
        'name': 'UnifiedDocumentation',
        'methods': [
            'initialize',
            'get_documentation',
            'search',
            'generate_documentation',
            'get_state'
        ],
        'base_class': 'Documentation'
    }
}

def analyze_repository(category, repo):
    """Analyze a repository to identify classes and methods that match the unified interface."""
    repo_path = FUNCTIONS_DIR / category / repo
    if not repo_path.exists() or repo_path.name == 'common':
        return None
    
    print(f"Analyzing repository: {repo}")
    
    # Find Python files in the repository
    python_files = list(repo_path.glob('**/*.py'))
    
    # Analyze each Python file
    classes = []
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the Python file
            tree = ast.parse(content)
            
            # Find classes in the file
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'file_path': file_path,
                        'methods': []
                    }
                    
                    # Find methods in the class
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            class_info['methods'].append(child.name)
                    
                    classes.append(class_info)
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    return classes

def find_matching_classes(category, classes):
    """Find classes that match the unified interface for a category."""
    if not classes:
        return []
    
    interface = UNIFIED_INTERFACES[category]
    matching_classes = []
    
    for class_info in classes:
        # Check if the class has methods that match the unified interface
        method_matches = sum(1 for method in interface['methods'] if method in class_info['methods'])
        match_percentage = method_matches / len(interface['methods'])
        
        if match_percentage >= 0.3:  # At least 30% match
            class_info['match_percentage'] = match_percentage
            matching_classes.append(class_info)
    
    # Sort by match percentage
    matching_classes.sort(key=lambda x: x['match_percentage'], reverse=True)
    
    return matching_classes

def create_base_class(category):
    """Create a base class for the unified interface."""
    interface = UNIFIED_INTERFACES[category]
    base_class_path = FUNCTIONS_DIR / category / 'unified' / f"{interface['base_class'].lower()}.py"
    
    # Create the directory if it doesn't exist
    base_class_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the base class file
    with open(base_class_path, 'w', encoding='utf-8') as f:
        f.write(f"""# Base class for {category}
class {interface['base_class']}:
    \"\"\"Base class for {category} functionality.\"\"\"
    
    def __init__(self, config=None):
        \"\"\"Initialize with optional configuration.\"\"\"
        self.config = config or {{}}
        self.state = "initialized"
    
""")
        
        # Add method stubs
        for method in interface['methods']:
            if method == 'initialize':
                continue  # Already covered by __init__
            
            f.write(f"""    def {method}(self, *args, **kwargs):
        \"\"\"
        {method.replace('_', ' ').capitalize()} operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the {method} operation.
        \"\"\"
        raise NotImplementedError("Subclasses must implement {method}()")
    
""")
    
    # Create an __init__.py file
    init_path = FUNCTIONS_DIR / category / 'unified' / '__init__.py'
    with open(init_path, 'w', encoding='utf-8') as f:
        f.write(f"""# Unified interface for {category}
from .{interface['base_class'].lower()} import {interface['base_class']}
from .{interface['name'].lower()} import {interface['name']}
from .factory import {category.replace('_', ' ').title().replace(' ', '')}Factory

__all__ = ['{interface['base_class']}', '{interface['name']}', '{category.replace('_', ' ').title().replace(' ', '')}Factory']
""")
    
    return base_class_path

def create_unified_interface(category):
    """Create a unified interface for a category."""
    interface = UNIFIED_INTERFACES[category]
    interface_path = FUNCTIONS_DIR / category / 'unified' / f"{interface['name'].lower()}.py"
    
    # Create the directory if it doesn't exist
    interface_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the unified interface file
    with open(interface_path, 'w', encoding='utf-8') as f:
        f.write(f"""# Unified interface for {category}
from .{interface['base_class'].lower()} import {interface['base_class']}

class {interface['name']}({interface['base_class']}):
    \"\"\"Unified interface for {category} functionality.\"\"\"
    
    def __init__(self, implementation=None, config=None):
        \"\"\"
        Initialize the unified interface.
        
        Args:
            implementation: The specific implementation to use.
            config: Optional configuration.
        \"\"\"
        super().__init__(config)
        self.implementation = implementation
    
""")
        
        # Add method implementations
        for method in interface['methods']:
            if method == 'initialize':
                continue  # Already covered by __init__
            
            f.write(f"""    def {method}(self, *args, **kwargs):
        \"\"\"
        {method.replace('_', ' ').capitalize()} operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the {method} operation.
        \"\"\"
        if self.implementation and hasattr(self.implementation, '{method}'):
            return getattr(self.implementation, '{method}')(*args, **kwargs)
        return super().{method}(*args, **kwargs)
    
""")
    
    return interface_path

def create_adapter(category, repo, class_info):
    """Create an adapter for a specific class in a repository."""
    interface = UNIFIED_INTERFACES[category]
    adapter_dir = FUNCTIONS_DIR / category / 'unified' / 'adapters'
    adapter_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the adapter file
    adapter_path = adapter_dir / f"{repo.lower()}_{class_info['name'].lower()}_adapter.py"
    
    # Handle placeholder classes
    if repo == 'placeholder':
        with open(adapter_path, 'w', encoding='utf-8') as f:
            f.write(f"""# Placeholder adapter for {category}
from ..{interface['base_class'].lower()} import {interface['base_class']}

class Placeholder{class_info['name']}Adapter({interface['base_class']}):
    \"\"\"Placeholder adapter for {category}.\"\"\"
    
    def __init__(self, config=None):
        \"\"\"Initialize the adapter.\"\"\"
        super().__init__(config)
        self.implementation = None
    
""")
            
            # Add method implementations
            for method in interface['methods']:
                if method == 'initialize':
                    continue  # Already covered by __init__
                
                f.write(f"""    def {method}(self, *args, **kwargs):
        \"\"\"
        {method.replace('_', ' ').capitalize()} operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the {method} operation.
        \"\"\"
        # Placeholder implementation
        print(f"Placeholder {method} called with args={{args}} kwargs={{kwargs}}")
        return f"Placeholder {method} result"
    
""")
        
        # Add the adapter to the __init__.py file
        init_path = adapter_dir / '__init__.py'
        adapter_class_name = f"Placeholder{class_info['name']}Adapter"
        
        if init_path.exists():
            with open(init_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if the adapter is already in the __init__.py file
            if adapter_class_name not in content:
                with open(init_path, 'a', encoding='utf-8') as f:
                    f.write(f"from .{repo.lower()}_{class_info['name'].lower()}_adapter import {adapter_class_name}\n")
                    f.write(f"__all__.append('{adapter_class_name}')\n")
        else:
            with open(init_path, 'w', encoding='utf-8') as f:
                f.write(f"""# Adapters for {category}
from .{repo.lower()}_{class_info['name'].lower()}_adapter import {adapter_class_name}

__all__ = ['{adapter_class_name}']
""")
        
        return adapter_path
    
    # Handle common classes
    elif repo == 'common':
        # Get the relative import path
        rel_path = class_info['file_path'].relative_to(FUNCTIONS_DIR)
        import_path = '.'.join(str(rel_path).replace('.py', '').split('/'))
        
        with open(adapter_path, 'w', encoding='utf-8') as f:
            f.write(f"""# Adapter for common/{class_info['name']}
from functions.{import_path} import {class_info['name']}
from ..{interface['base_class'].lower()} import {interface['base_class']}

class Common{class_info['name']}Adapter({interface['base_class']}):
    \"\"\"Adapter for common/{class_info['name']}.\"\"\"
    
    def __init__(self, config=None):
        \"\"\"Initialize the adapter.\"\"\"
        super().__init__(config)
        self.implementation = {class_info['name']}(**config) if config else {class_info['name']}()
    
""")
            
            # Add method implementations
            for method in interface['methods']:
                if method == 'initialize':
                    continue  # Already covered by __init__
                
                # Check if the method exists in the original class
                method_exists = method in class_info['methods']
                
                if method_exists:
                    f.write(f"""    def {method}(self, *args, **kwargs):
        \"\"\"
        {method.replace('_', ' ').capitalize()} operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the {method} operation.
        \"\"\"
        if hasattr(self.implementation, '{method}'):
            return getattr(self.implementation, '{method}')(*args, **kwargs)
        return super().{method}(*args, **kwargs)
    
""")
                else:
                    f.write(f"""    def {method}(self, *args, **kwargs):
        \"\"\"
        {method.replace('_', ' ').capitalize()} operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the {method} operation.
        \"\"\"
        # Method not implemented in the original class
        return super().{method}(*args, **kwargs)
    
""")
        
        # Add the adapter to the __init__.py file
        init_path = adapter_dir / '__init__.py'
        adapter_class_name = f"Common{class_info['name']}Adapter"
        
        if init_path.exists():
            with open(init_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if the adapter is already in the __init__.py file
            if adapter_class_name not in content:
                with open(init_path, 'a', encoding='utf-8') as f:
                    f.write(f"from .{repo.lower()}_{class_info['name'].lower()}_adapter import {adapter_class_name}\n")
                    f.write(f"__all__.append('{adapter_class_name}')\n")
        else:
            with open(init_path, 'w', encoding='utf-8') as f:
                f.write(f"""# Adapters for {category}
from .{repo.lower()}_{class_info['name'].lower()}_adapter import {adapter_class_name}

__all__ = ['{adapter_class_name}']
""")
        
        return adapter_path
    
    # Handle regular repository classes
    else:
        # Get the relative import path
        rel_path = class_info['file_path'].relative_to(FUNCTIONS_DIR)
        import_path = '.'.join(str(rel_path).replace('.py', '').split('/'))
        
        with open(adapter_path, 'w', encoding='utf-8') as f:
            f.write(f"""# Adapter for {repo}/{class_info['name']}
from functions.{import_path} import {class_info['name']}
from ..{interface['base_class'].lower()} import {interface['base_class']}

class {repo.replace('-', '_').title()}{class_info['name']}Adapter({interface['base_class']}):
    \"\"\"Adapter for {repo}/{class_info['name']}.\"\"\"
    
    def __init__(self, config=None):
        \"\"\"Initialize the adapter.\"\"\"
        super().__init__(config)
        self.implementation = {class_info['name']}(**config) if config else {class_info['name']}()
    
""")
            
            # Add method implementations
            for method in interface['methods']:
                if method == 'initialize':
                    continue  # Already covered by __init__
                
                # Check if the method exists in the original class
                method_exists = method in class_info['methods']
                
                # Method mappings for more flexible matching
                method_mappings = {
                    'initialize': ['__init__', 'init', 'setup', 'configure', 'create'],
                    'plan': ['create_plan', 'plan_task', 'strategize', 'prepare', 'think'],
                    'execute': ['run_task', 'perform', 'do', 'run', 'execute_task', 'start', 'perform_task', 'process'],
                    'run': ['execute_task', 'start', 'perform_task', 'execute', 'do', 'perform', 'process'],
                    'add_tool': ['register_tool', 'add_capability', 'extend', 'add_function', 'register'],
                    'add_memory': ['add_to_memory', 'store_memory', 'remember', 'memorize', 'store'],
                    'get_state': ['get_status', 'status', 'state', 'get_info', 'info'],
                    'render': ['display', 'show', 'draw', 'paint', 'view'],
                    'handle_input': ['process_input', 'on_input', 'input', 'handle_event', 'on_event'],
                    'update': ['refresh', 'redraw', 'sync', 'update_state', 'update_view'],
                    'capture_screen': ['screenshot', 'screen_capture', 'grab_screen', 'capture', 'get_screen'],
                    'control_keyboard': ['keyboard', 'type', 'send_keys', 'key_press', 'input'],
                    'control_mouse': ['mouse', 'click', 'move_mouse', 'mouse_move', 'mouse_click'],
                    'execute_command': ['run_command', 'shell', 'cmd', 'command', 'system'],
                    'navigate': ['goto', 'open', 'browse', 'visit', 'load'],
                    'click': ['press', 'select', 'choose', 'activate', 'trigger'],
                    'type': ['input', 'enter', 'fill', 'write', 'send_text'],
                    'extract_content': ['scrape', 'get_content', 'parse', 'extract', 'get_text'],
                    'analyze': ['examine', 'inspect', 'review', 'check', 'evaluate'],
                    'generate': ['create', 'produce', 'make', 'build', 'construct'],
                    'process': ['handle', 'compute', 'work', 'execute', 'run'],
                    'learn': ['train', 'adapt', 'improve', 'update', 'fit'],
                    'reason': ['think', 'infer', 'deduce', 'conclude', 'analyze'],
                    'evolve': ['mutate', 'adapt', 'change', 'modify', 'transform'],
                    'evaluate': ['assess', 'judge', 'rate', 'score', 'measure'],
                    'select': ['choose', 'pick', 'filter', 'find', 'get'],
                    'connect': ['link', 'join', 'integrate', 'bind', 'attach'],
                    'transform': ['convert', 'change', 'modify', 'alter', 'translate'],
                    'process_text': ['parse_text', 'analyze_text', 'handle_text', 'text_process', 'tokenize'],
                    'generate_text': ['create_text', 'produce_text', 'write', 'compose', 'author'],
                    'analyze_text': ['examine_text', 'inspect_text', 'review_text', 'text_analysis', 'parse'],
                    'get_documentation': ['get_docs', 'fetch_documentation', 'docs', 'help', 'manual'],
                    'search': ['find', 'lookup', 'query', 'seek', 'locate'],
                    'generate_documentation': ['create_docs', 'document', 'write_docs', 'generate_docs', 'document_code']
                }
                
                # Find a matching method if the exact name doesn't exist
                implementation_method = method
                if not method_exists and method in method_mappings:
                    for alt_method in method_mappings[method]:
                        if alt_method in class_info['methods']:
                            implementation_method = alt_method
                            method_exists = True
                            break
                
                if method_exists:
                    f.write(f"""    def {method}(self, *args, **kwargs):
        \"\"\"
        {method.replace('_', ' ').capitalize()} operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the {method} operation.
        \"\"\"
        if hasattr(self.implementation, '{implementation_method}'):
            return getattr(self.implementation, '{implementation_method}')(*args, **kwargs)
        return super().{method}(*args, **kwargs)
    
""")
                else:
                    f.write(f"""    def {method}(self, *args, **kwargs):
        \"\"\"
        {method.replace('_', ' ').capitalize()} operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the {method} operation.
        \"\"\"
        # Method not implemented in the original class
        return super().{method}(*args, **kwargs)
    
""")
        
        # Add the adapter to the __init__.py file
        init_path = adapter_dir / '__init__.py'
        adapter_class_name = f"{repo.replace('-', '_').title()}{class_info['name']}Adapter"
        
        if init_path.exists():
            with open(init_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if the adapter is already in the __init__.py file
            if adapter_class_name not in content:
                with open(init_path, 'a', encoding='utf-8') as f:
                    f.write(f"from .{repo.lower()}_{class_info['name'].lower()}_adapter import {adapter_class_name}\n")
                    f.write(f"__all__.append('{adapter_class_name}')\n")
        else:
            with open(init_path, 'w', encoding='utf-8') as f:
                f.write(f"""# Adapters for {category}
from .{repo.lower()}_{class_info['name'].lower()}_adapter import {adapter_class_name}

__all__ = ['{adapter_class_name}']
""")
        
        return adapter_path

def create_factory(category, adapters):
    """Create a factory for a category."""
    factory_path = FUNCTIONS_DIR / category / 'unified' / 'factory.py'
    
    # Create the factory file
    with open(factory_path, 'w', encoding='utf-8') as f:
        f.write(f"""# Factory for {category}
from .{UNIFIED_INTERFACES[category]['name'].lower()} import {UNIFIED_INTERFACES[category]['name']}
""")
        
        # Import adapters
        for adapter in adapters:
            if adapter['repo'] == 'placeholder':
                f.write(f"from .adapters.{adapter['repo'].lower()}_{adapter['class'].lower()}_adapter import Placeholder{adapter['class']}Adapter\n")
            elif adapter['repo'] == 'common':
                f.write(f"from .adapters.{adapter['repo'].lower()}_{adapter['class'].lower()}_adapter import Common{adapter['class']}Adapter\n")
            else:
                f.write(f"from .adapters.{adapter['repo'].lower()}_{adapter['class'].lower()}_adapter import {adapter['adapter_class']}\n")
        
        f.write(f"""
class {category.replace('_', ' ').title().replace(' ', '')}Factory:
    \"\"\"Factory for creating {category} implementations.\"\"\"
    
    @staticmethod
    def create(implementation=None, config=None):
        \"\"\"
        Create a {category} implementation.
        
        Args:
            implementation (str): The name of the implementation to create.
            config (dict): Optional configuration.
            
        Returns:
            {UNIFIED_INTERFACES[category]['name']}: A unified interface for the implementation.
        \"\"\"
        config = config or {{}}
        
        # Create the appropriate implementation
""")
        
        # Add cases for each adapter
        for i, adapter in enumerate(adapters):
            condition = "if" if i == 0 else "elif"
            
            if adapter['repo'] == 'placeholder':
                adapter_class = f"Placeholder{adapter['class']}Adapter"
            elif adapter['repo'] == 'common':
                adapter_class = f"Common{adapter['class']}Adapter"
            else:
                adapter_class = adapter['adapter_class']
            
            f.write(f"""        {condition} implementation == '{adapter['repo']}/{adapter['class']}':
            return {UNIFIED_INTERFACES[category]['name']}({adapter_class}(config), config)
""")
        
        # Add default case
        if adapters:
            if adapters[0]['repo'] == 'placeholder':
                default_adapter = f"Placeholder{adapters[0]['class']}Adapter"
            elif adapters[0]['repo'] == 'common':
                default_adapter = f"Common{adapters[0]['class']}Adapter"
            else:
                default_adapter = adapters[0]['adapter_class']
            
            f.write(f"""        else:
            # Default to the first implementation
            return {UNIFIED_INTERFACES[category]['name']}({default_adapter}(config), config)
""")
        else:
            f.write(f"""        else:
            # No implementations available
            return {UNIFIED_INTERFACES[category]['name']}(None, config)
""")
    
    return factory_path

def create_example(category, adapters):
    """Create an example for a category."""
    example_path = FUNCTIONS_DIR / category / 'unified' / 'example.py'
    
    # Create the example file
    with open(example_path, 'w', encoding='utf-8') as f:
        f.write(f"""# Example usage of the unified interface for {category}
from .factory import {category.replace('_', ' ').title().replace(' ', '')}Factory

def main():
    \"\"\"Example usage of the unified interface.\"\"\"
    # Create a unified interface with the default implementation
    unified = {category.replace('_', ' ').title().replace(' ', '')}Factory.create()
    
    # Use the unified interface
""")
        
        # Add example usage for each method
        for method in UNIFIED_INTERFACES[category]['methods']:
            if method == 'initialize':
                continue  # Already covered by the factory
            
            f.write(f"""    try:
        # Example usage of {method}
        result = unified.{method}()
        print(f"{method} result: {{result}}")
    except NotImplementedError:
        print(f"{method} not implemented in the default implementation")
    
""")
        
        # Add examples for each adapter
        for adapter in adapters:
            f.write(f"""    # Create a unified interface with the {adapter['repo']}/{adapter['class']} implementation
    unified = {category.replace('_', ' ').title().replace(' ', '')}Factory.create('{adapter['repo']}/{adapter['class']}')
    
    # Use the unified interface
    try:
        # Example usage
        result = unified.{UNIFIED_INTERFACES[category]['methods'][1]}()
        print(f"{adapter['repo']}/{adapter['class']} {UNIFIED_INTERFACES[category]['methods'][1]} result: {{result}}")
    except NotImplementedError:
        print(f"{UNIFIED_INTERFACES[category]['methods'][1]} not implemented in {adapter['repo']}/{adapter['class']}")
    
""")
        
        f.write("""
if __name__ == "__main__":
    main()
""")
    
    return example_path

def unify_category(category):
    """Unify the code in a category."""
    print(f"\nUnifying {category}...")
    
    # Get repositories in the category
    repos = [d.name for d in (FUNCTIONS_DIR / category).iterdir() if d.is_dir() and d.name != 'common' and d.name != 'unified']
    
    # Analyze repositories
    all_classes = []
    for repo in repos:
        classes = analyze_repository(category, repo)
        if classes:
            for class_info in classes:
                # Filter out classes with no methods or very short names (likely not main classes)
                if len(class_info['methods']) > 0 and len(class_info['name']) > 2:
                    all_classes.append({
                        'repo': repo,
                        'class': class_info['name'],
                        'methods': class_info['methods'],
                        'file_path': class_info['file_path']
                    })
    
    # Find matching classes
    matching_classes = []
    
    # Method mappings for more flexible matching
    method_mappings = {
        'initialize': ['__init__', 'init', 'setup', 'configure', 'create'],
        'plan': ['create_plan', 'plan_task', 'strategize', 'prepare', 'think'],
        'execute': ['run_task', 'perform', 'do', 'run', 'execute_task', 'start', 'perform_task', 'process'],
        'run': ['execute_task', 'start', 'perform_task', 'execute', 'do', 'perform', 'process'],
        'add_tool': ['register_tool', 'add_capability', 'extend', 'add_function', 'register'],
        'add_memory': ['add_to_memory', 'store_memory', 'remember', 'memorize', 'store'],
        'get_state': ['get_status', 'status', 'state', 'get_info', 'info'],
        'render': ['display', 'show', 'draw', 'paint', 'view'],
        'handle_input': ['process_input', 'on_input', 'input', 'handle_event', 'on_event'],
        'update': ['refresh', 'redraw', 'sync', 'update_state', 'update_view'],
        'capture_screen': ['screenshot', 'screen_capture', 'grab_screen', 'capture', 'get_screen'],
        'control_keyboard': ['keyboard', 'type', 'send_keys', 'key_press', 'input'],
        'control_mouse': ['mouse', 'click', 'move_mouse', 'mouse_move', 'mouse_click'],
        'execute_command': ['run_command', 'shell', 'cmd', 'command', 'system'],
        'navigate': ['goto', 'open', 'browse', 'visit', 'load'],
        'click': ['press', 'select', 'choose', 'activate', 'trigger'],
        'type': ['input', 'enter', 'fill', 'write', 'send_text'],
        'extract_content': ['scrape', 'get_content', 'parse', 'extract', 'get_text'],
        'analyze': ['examine', 'inspect', 'review', 'check', 'evaluate'],
        'generate': ['create', 'produce', 'make', 'build', 'construct'],
        'process': ['handle', 'compute', 'work', 'execute', 'run'],
        'learn': ['train', 'adapt', 'improve', 'update', 'fit'],
        'reason': ['think', 'infer', 'deduce', 'conclude', 'analyze'],
        'evolve': ['mutate', 'adapt', 'change', 'modify', 'transform'],
        'evaluate': ['assess', 'judge', 'rate', 'score', 'measure'],
        'select': ['choose', 'pick', 'filter', 'find', 'get'],
        'connect': ['link', 'join', 'integrate', 'bind', 'attach'],
        'transform': ['convert', 'change', 'modify', 'alter', 'translate'],
        'process_text': ['parse_text', 'analyze_text', 'handle_text', 'text_process', 'tokenize'],
        'generate_text': ['create_text', 'produce_text', 'write', 'compose', 'author'],
        'analyze_text': ['examine_text', 'inspect_text', 'review_text', 'text_analysis', 'parse'],
        'get_documentation': ['get_docs', 'fetch_documentation', 'docs', 'help', 'manual'],
        'search': ['find', 'lookup', 'query', 'seek', 'locate'],
        'generate_documentation': ['create_docs', 'document', 'write_docs', 'generate_docs', 'document_code']
    }
    
    for class_info in all_classes:
        # Check if the class name is relevant to the category
        class_name_lower = class_info['class'].lower()
        category_keywords = {
            'agent_frameworks': ['agent', 'bot', 'assistant', 'ai', 'gpt', 'llm', 'task'],
            'user_interfaces': ['ui', 'interface', 'view', 'component', 'widget', 'display', 'screen', 'page'],
            'os_interaction': ['os', 'system', 'computer', 'keyboard', 'mouse', 'screen', 'window', 'process'],
            'browser_automation': ['browser', 'web', 'page', 'navigate', 'scrape', 'selenium', 'puppeteer'],
            'code_execution': ['code', 'execute', 'compiler', 'interpreter', 'runtime', 'eval', 'program'],
            'cognitive_systems': ['cognitive', 'brain', 'mind', 'think', 'reason', 'knowledge', 'memory'],
            'evolution_optimization': ['evolution', 'genetic', 'optimize', 'fitness', 'selection', 'mutation'],
            'integration': ['integration', 'connect', 'system', 'service', 'api', 'interface', 'bridge'],
            'nlp': ['nlp', 'language', 'text', 'token', 'parse', 'grammar', 'semantic', 'syntax'],
            'documentation': ['doc', 'documentation', 'manual', 'guide', 'help', 'reference', 'tutorial']
        }
        
        # Check if class name contains any of the category keywords
        name_relevance = any(keyword in class_name_lower for keyword in category_keywords.get(category, []))
        
        # Check if the class has methods that match the unified interface
        method_matches = 0
        for method in UNIFIED_INTERFACES[category]['methods']:
            # Check for exact match
            if method in class_info['methods']:
                method_matches += 1
                continue
                
            # Check for alternative method names
            for alt_method in method_mappings.get(method, []):
                if alt_method in class_info['methods']:
                    method_matches += 0.8  # Slightly lower score for alternative names
                    break
        
        # Calculate match percentage
        match_percentage = method_matches / len(UNIFIED_INTERFACES[category]['methods'])
        
        # Boost score if class name is relevant to the category
        if name_relevance:
            match_percentage += 0.2
        
        # Accept classes with at least 10% match or relevant name
        if match_percentage >= 0.1 or name_relevance:
            class_info['match_percentage'] = min(match_percentage, 1.0)  # Cap at 100%
            matching_classes.append(class_info)
    
    # Sort by match percentage
    matching_classes.sort(key=lambda x: x['match_percentage'], reverse=True)
    
    # Take the top 3 matching classes or create placeholder classes if none found
    top_matches = matching_classes[:3]
    
    # If no matches found, create placeholder classes based on the common components
    if not top_matches:
        print(f"No matching classes found for {category}, creating placeholder classes...")
        common_dir = FUNCTIONS_DIR / category / 'common'
        if common_dir.exists():
            for file_path in common_dir.glob('*.py'):
                if file_path.name != '__init__.py' and file_path.name != 'README.md':
                    # Parse the file to get class name
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef):
                                # Create a placeholder class
                                top_matches.append({
                                    'repo': 'common',
                                    'class': node.name,
                                    'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                                    'file_path': file_path,
                                    'match_percentage': 0.5  # Medium match
                                })
                                break
                    except Exception as e:
                        print(f"Error parsing {file_path}: {e}")
        
        # If still no matches, create a placeholder class
        if not top_matches:
            placeholder_class = {
                'repo': 'placeholder',
                'class': UNIFIED_INTERFACES[category]['base_class'],
                'methods': UNIFIED_INTERFACES[category]['methods'],
                'file_path': FUNCTIONS_DIR / category / 'common' / f"{UNIFIED_INTERFACES[category]['base_class'].lower()}.py",
                'match_percentage': 0.5  # Medium match
            }
            top_matches.append(placeholder_class)
    
    if not top_matches:
        print(f"No matching classes found for {category}")
        return
    
    print(f"Found {len(top_matches)} matching classes for {category}")
    
    # Create the base class
    base_class_path = create_base_class(category)
    print(f"Created base class: {base_class_path}")
    
    # Create the unified interface
    interface_path = create_unified_interface(category)
    print(f"Created unified interface: {interface_path}")
    
    # Create adapters for the top matching classes
    adapters = []
    for class_info in top_matches:
        adapter_class_name = f"{class_info['repo'].replace('-', '_').title()}{class_info['class']}Adapter"
        adapter_path = create_adapter(category, class_info['repo'], {
            'name': class_info['class'],
            'methods': class_info['methods'],
            'file_path': class_info['file_path']
        })
        adapters.append({
            'repo': class_info['repo'],
            'class': class_info['class'],
            'adapter_class': adapter_class_name
        })
        print(f"Created adapter: {adapter_path}")
    
    # Create the factory
    factory_path = create_factory(category, adapters)
    print(f"Created factory: {factory_path}")
    
    # Create an example
    example_path = create_example(category, adapters)
    print(f"Created example: {example_path}")

def main():
    """Main function."""
    print("Unifying code across repositories within each functional category...")
    
    # Unify each category
    for category in CATEGORIES:
        unify_category(category)
    
    print("\nCode unification complete!")

if __name__ == "__main__":
    main()