#!/usr/bin/env python3
"""
Script to categorize repositories by function and create symbolic links
to the relevant code in the appropriate category directories.
"""

import os
import shutil
import subprocess
from pathlib import Path

# Define the repository categories
REPO_CATEGORIES = {
    'agent_frameworks': [
        'AgentForge', 'AgentGPT', 'AgentK', 'Auto-GPT-MetaTrader-Plugin',
        'Free-Auto-GPT', 'agent-zero', 'autogen', 'babyagi', 'Teenage-AGI'
    ],
    'user_interfaces': [
        'open-webui', 'web-ui', 'draw-a-ui'
    ],
    'os_interaction': [
        'MacOS-Agent', 'self-operating-computer', 'mcp-remote-macos-use'
    ],
    'browser_automation': [
        'browser-use'
    ],
    'code_execution': [
        'open-interpreter', 'pyCodeAGI', 'SuperCoder', 'automata'
    ],
    'cognitive_systems': [
        'opencog', 'openagi', 'AGI-Samantha'
    ],
    'evolution_optimization': [
        'openevolve'
    ],
    'integration': [
        'Unification', 'LocalAGI'
    ],
    'nlp': [
        'senpai', 'html-agility-pack'
    ],
    'documentation': [
        'Awesome-AGI', 'awesome-agi-cocosci'
    ]
}

# Base directories
REPO_DIR = Path('repos')
FUNCTIONS_DIR = Path('functions')

def create_symlinks():
    """Create symbolic links from repositories to their functional categories."""
    for category, repos in REPO_CATEGORIES.items():
        category_dir = FUNCTIONS_DIR / category
        
        # Ensure category directory exists
        category_dir.mkdir(exist_ok=True)
        
        for repo in repos:
            repo_path = REPO_DIR / repo
            if not repo_path.exists():
                print(f"Warning: Repository {repo} not found, skipping")
                continue
            
            # Create a symlink to the repository in the category directory
            target_path = category_dir / repo
            if target_path.exists():
                if target_path.is_symlink():
                    target_path.unlink()
                else:
                    shutil.rmtree(target_path)
            
            # Create relative symlink
            rel_path = os.path.relpath(repo_path, category_dir)
            os.symlink(rel_path, target_path, target_is_directory=True)
            print(f"Created symlink: {target_path} -> {rel_path}")

def extract_common_functionality():
    """Extract common functionality from repositories in each category."""
    for category, repos in REPO_CATEGORIES.items():
        category_dir = FUNCTIONS_DIR / category
        common_dir = category_dir / 'common'
        common_dir.mkdir(exist_ok=True)
        
        # Create a file to document common functionality
        with open(common_dir / 'README.md', 'w') as f:
            f.write(f"# Common Functionality in {category.replace('_', ' ').title()}\n\n")
            f.write("This directory contains common functionality extracted from repositories in this category.\n\n")
            f.write("## Repositories Analyzed\n\n")
            for repo in repos:
                repo_path = REPO_DIR / repo
                if repo_path.exists():
                    f.write(f"- {repo}\n")
            
            f.write("\n## Common Components\n\n")
            f.write("The following components represent common functionality across repositories in this category:\n\n")
            
            # This would be where you'd add specific common functionality
            # For now, we'll just add placeholders
            if category == 'agent_frameworks':
                f.write("- Agent lifecycle management\n")
                f.write("- Task planning and execution\n")
                f.write("- Memory and context management\n")
            elif category == 'user_interfaces':
                f.write("- Web-based chat interfaces\n")
                f.write("- UI component libraries\n")
                f.write("- Visualization tools\n")
            # Add more category-specific common functionality here
            
            f.write("\n## Usage\n\n")
            f.write("To use these common components, import them from the appropriate module.\n")

def main():
    """Main function to categorize repositories."""
    # Create symbolic links
    create_symlinks()
    
    # Extract common functionality
    extract_common_functionality()
    
    print("Repository categorization complete!")

if __name__ == "__main__":
    main()