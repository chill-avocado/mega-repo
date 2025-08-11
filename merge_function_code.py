#!/usr/bin/env python3
"""
Merge code from different repositories into a single unified codebase for each functional category.

This script:
1. Identifies similar files across repositories within each functional category
2. Merges similar files into a single file
3. Resolves conflicts and duplications
4. Creates a unified codebase for each functional category
"""

import os
import sys
import shutil
import re
import ast
from pathlib import Path
from collections import defaultdict
import importlib.util

# Base directories
FUNCTIONS_DIR = Path('functions')
MERGED_DIR = Path('merged_functions')

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

# Define the core functionality for each category
CORE_FUNCTIONALITY = {
    'agent_frameworks': [
        'agent',
        'memory',
        'planning',
        'execution',
        'tools',
        'communication'
    ],
    'user_interfaces': [
        'components',
        'layout',
        'styling',
        'interaction',
        'rendering',
        'state_management'
    ],
    'os_interaction': [
        'keyboard',
        'mouse',
        'screen',
        'filesystem',
        'process',
        'system'
    ],
    'browser_automation': [
        'navigation',
        'interaction',
        'scraping',
        'rendering',
        'network',
        'storage'
    ],
    'code_execution': [
        'execution',
        'compilation',
        'interpretation',
        'analysis',
        'generation',
        'testing'
    ],
    'cognitive_systems': [
        'knowledge',
        'reasoning',
        'learning',
        'memory',
        'perception',
        'planning'
    ],
    'evolution_optimization': [
        'selection',
        'mutation',
        'crossover',
        'fitness',
        'population',
        'evolution'
    ],
    'integration': [
        'connectors',
        'adapters',
        'transformers',
        'orchestration',
        'messaging',
        'synchronization'
    ],
    'nlp': [
        'tokenization',
        'parsing',
        'generation',
        'understanding',
        'sentiment',
        'translation'
    ],
    'documentation': [
        'generation',
        'organization',
        'search',
        'visualization',
        'annotation',
        'versioning'
    ]
}

def extract_imports(file_path):
    """Extract imports from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for name in node.names:
                    if module:
                        imports.append(f"{module}.{name.name}")
                    else:
                        imports.append(name.name)
        
        return imports
    except Exception as e:
        print(f"Error extracting imports from {file_path}: {e}")
        return []

def extract_classes(file_path):
    """Extract class definitions from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        return classes
    except Exception as e:
        print(f"Error extracting classes from {file_path}: {e}")
        return []

def extract_functions(file_path):
    """Extract function definitions from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        
        return functions
    except Exception as e:
        print(f"Error extracting functions from {file_path}: {e}")
        return []

def categorize_file(file_path, category):
    """Categorize a file based on its content and the category's core functionality."""
    # Extract content
    imports = extract_imports(file_path)
    classes = extract_classes(file_path)
    functions = extract_functions(file_path)
    
    # Check file name
    file_name = file_path.name.lower()
    
    # Check for core functionality in file name
    for core in CORE_FUNCTIONALITY[category]:
        if core in file_name:
            return core
    
    # Check for core functionality in class names
    for core in CORE_FUNCTIONALITY[category]:
        for class_name in classes:
            if core in class_name.lower():
                return core
    
    # Check for core functionality in function names
    for core in CORE_FUNCTIONALITY[category]:
        for function_name in functions:
            if core in function_name.lower():
                return core
    
    # Default to 'utils' if no match found
    return 'utils'

def merge_python_files(files, output_file):
    """Merge multiple Python files into a single file."""
    # Create a set to track imported modules
    imported_modules = set()
    
    # Create a set to track defined classes and functions
    defined_symbols = set()
    
    # Create the output file
    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write(f"""# Merged file for {output_file.parent.name}/{output_file.stem}
# This file contains code merged from multiple repositories

""")
        
        # Process each file
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as in_file:
                    content = in_file.read()
                
                # Parse the file
                tree = ast.parse(content)
                
                # Extract imports, classes, and functions
                file_imports = []
                file_classes = []
                file_functions = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            if name.name not in imported_modules:
                                file_imports.append(f"import {name.name}")
                                imported_modules.add(name.name)
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ''
                        for name in node.names:
                            import_str = f"from {module} import {name.name}"
                            if import_str not in imported_modules:
                                file_imports.append(import_str)
                                imported_modules.add(import_str)
                    elif isinstance(node, ast.ClassDef):
                        if node.name not in defined_symbols:
                            # Get the class source code
                            class_source = ast.get_source_segment(content, node)
                            if class_source:
                                file_classes.append(f"# From {file_path.parent.name}/{file_path.name}\n{class_source}")
                                defined_symbols.add(node.name)
                    elif isinstance(node, ast.FunctionDef):
                        if node.name not in defined_symbols and not node.name.startswith('_'):
                            # Get the function source code
                            func_source = ast.get_source_segment(content, node)
                            if func_source:
                                file_functions.append(f"# From {file_path.parent.name}/{file_path.name}\n{func_source}")
                                defined_symbols.add(node.name)
                
                # Write imports
                for import_str in file_imports:
                    out_file.write(f"{import_str}\n")
                
                # Add a separator
                out_file.write("\n")
                
                # Write classes
                for class_str in file_classes:
                    out_file.write(f"{class_str}\n\n")
                
                # Write functions
                for func_str in file_functions:
                    out_file.write(f"{func_str}\n\n")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    return output_file

def merge_category(category):
    """Merge code from different repositories within a category."""
    print(f"\nMerging code for {category}...")
    
    # Create the output directory
    output_dir = MERGED_DIR / category
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all Python files in the category
    category_dir = FUNCTIONS_DIR / category
    python_files = []
    
    for repo_dir in category_dir.iterdir():
        if repo_dir.is_dir() and repo_dir.name != 'common' and repo_dir.name != 'unified':
            for file_path in repo_dir.glob('**/*.py'):
                if not file_path.name.startswith('__') and not file_path.name.startswith('test_'):
                    python_files.append(file_path)
    
    # Categorize files by core functionality
    categorized_files = defaultdict(list)
    
    for file_path in python_files:
        core = categorize_file(file_path, category)
        categorized_files[core].append(file_path)
    
    # Merge files by core functionality
    for core, files in categorized_files.items():
        if files:
            output_file = output_dir / f"{core}.py"
            merge_python_files(files, output_file)
            print(f"  Created {output_file} from {len(files)} files")
    
    # Create an __init__.py file
    init_file = output_dir / "__init__.py"
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write(f"""# Merged code for {category}
# This package contains code merged from multiple repositories

""")
        
        # Import all modules
        for core in CORE_FUNCTIONALITY[category]:
            if (output_dir / f"{core}.py").exists():
                f.write(f"from .{core} import *\n")
        
        # Import utils if it exists
        if (output_dir / "utils.py").exists():
            f.write("from .utils import *\n")
    
    # Create a README.md file
    readme_file = output_dir / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(f"""# Merged {category.replace('_', ' ').title()} Code

This directory contains code merged from multiple repositories related to {category.replace('_', ' ')}.

## Core Functionality

""")
        
        for core in CORE_FUNCTIONALITY[category]:
            if (output_dir / f"{core}.py").exists():
                f.write(f"- **{core}.py**: {core.replace('_', ' ').title()} functionality\n")
        
        if (output_dir / "utils.py").exists():
            f.write("- **utils.py**: Utility functions and classes\n")
        
        f.write("""
## Usage

You can import and use the merged code as follows:

```python
from merged_functions.{} import *

# Use the merged functionality
```

## Source Repositories

The code in this directory is merged from the following repositories:

""".format(category))
        
        # List source repositories
        for repo_dir in category_dir.iterdir():
            if repo_dir.is_dir() and repo_dir.name != 'common' and repo_dir.name != 'unified':
                f.write(f"- {repo_dir.name}\n")

def create_merged_readme():
    """Create a README.md file for the merged functions directory."""
    readme_file = MERGED_DIR / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write("""# Merged Functions

This directory contains code merged from multiple repositories, organized by functional category.

## Categories

""")
        
        for category in CATEGORIES:
            f.write(f"- **[{category.replace('_', ' ').title()}](./{category}/)**: {category.replace('_', ' ')} functionality\n")
        
        f.write("""
## Usage

You can import and use the merged code as follows:

```python
from merged_functions.agent_frameworks import *

# Use the agent functionality
```

## Structure

Each category directory contains:
- Python modules for each core functionality
- A README.md file explaining the category and its functionality
- An __init__.py file that imports all functionality

The code is organized by functionality rather than by repository, making it easier to use and understand.
""")

def main():
    """Main function."""
    print("Merging code from different repositories into a single unified codebase for each functional category...")
    
    # Create the merged functions directory
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Merge each category
    for category in CATEGORIES:
        merge_category(category)
    
    # Create a README.md file for the merged functions directory
    create_merged_readme()
    
    print("\nCode merging complete!")

if __name__ == "__main__":
    main()