#!/usr/bin/env python3
"""
Script to generate a comprehensive index of all repositories and their categorized functionality.
"""

import os
import json
from pathlib import Path
import re

# Base directories
REPO_DIR = Path('repos')
FUNCTIONS_DIR = Path('functions')

def generate_index():
    """Generate a comprehensive index of all repositories and their categorized functionality."""
    # Load the repository categories
    with open('categorize_repos.py', 'r') as f:
        content = f.read()
        repo_categories_match = re.search(r'REPO_CATEGORIES\s*=\s*{([^}]+)}', content, re.DOTALL)
        if not repo_categories_match:
            print("Could not find REPO_CATEGORIES in categorize_repos.py")
            return
        
        # Extract and evaluate the REPO_CATEGORIES dictionary
        repo_categories_str = '{' + repo_categories_match.group(1) + '}'
        repo_categories = eval(repo_categories_str)
    
    # Load the functionality index if it exists
    try:
        with open(FUNCTIONS_DIR / 'functionality_index.json', 'r') as f:
            functionality_index = json.load(f)
    except FileNotFoundError:
        functionality_index = {}
    
    # Create the index
    index = {
        'categories': {},
        'repositories': {}
    }
    
    # Add categories
    for category, repos in repo_categories.items():
        index['categories'][category] = {
            'name': category.replace('_', ' ').title(),
            'repositories': repos,
            'common_components': []
        }
        
        # Add common components
        common_dir = FUNCTIONS_DIR / category / 'common'
        if common_dir.exists():
            for file_path in common_dir.glob('*.py'):
                if file_path.name != '__init__.py':
                    component_name = file_path.stem
                    index['categories'][category]['common_components'].append(component_name)
    
    # Add repositories
    for category, repos in repo_categories.items():
        for repo in repos:
            repo_path = REPO_DIR / repo
            if not repo_path.exists():
                continue
            
            # Get repository information
            index['repositories'][repo] = {
                'name': repo,
                'category': category,
                'path': str(repo_path),
                'functionality': functionality_index.get(category, {}).get(repo, {})
            }
    
    # Save the index
    with open(FUNCTIONS_DIR / 'index.json', 'w') as f:
        json.dump(index, f, indent=2)
    
    # Create a markdown version of the index
    with open(FUNCTIONS_DIR / 'index.md', 'w') as f:
        f.write("# Repository Index\n\n")
        f.write("This document provides an index of all repositories categorized by functionality.\n\n")
        
        f.write("## Categories\n\n")
        for category, info in index['categories'].items():
            f.write(f"### {info['name']}\n\n")
            
            f.write("#### Repositories\n\n")
            for repo in info['repositories']:
                if repo in index['repositories']:
                    f.write(f"- [{repo}](../repos/{repo})\n")
                else:
                    f.write(f"- {repo} (not found)\n")
            
            f.write("\n#### Common Components\n\n")
            for component in info['common_components']:
                f.write(f"- [{component}](./common/{component}.py)\n")
            
            f.write("\n")
        
        f.write("## Repositories\n\n")
        for repo, info in index['repositories'].items():
            f.write(f"### {repo}\n\n")
            f.write(f"Category: {info['category'].replace('_', ' ').title()}\n\n")
            
            f.write("#### Functionality\n\n")
            if info['functionality']:
                for func_type, files in info['functionality'].items():
                    f.write(f"- **{func_type.replace('_', ' ').title()}**\n")
                    if files:
                        for file in sorted(files)[:5]:  # Limit to 5 files to avoid too much output
                            f.write(f"  - `{file}`\n")
                        if len(files) > 5:
                            f.write(f"  - ... and {len(files) - 5} more files\n")
            else:
                f.write("No specific functionality patterns detected.\n")
            
            f.write("\n")
    
    print("Index generation complete!")

if __name__ == "__main__":
    generate_index()