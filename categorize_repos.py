#!/usr/bin/env python3
"""
Script to categorize repositories by function and copy
the relevant code to the appropriate category directories.
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

# File extensions to copy (code files only)
CODE_EXTENSIONS = [
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cs', '.go', '.rs',
    '.c', '.cpp', '.h', '.hpp', '.rb', '.php', '.scala', '.kt', '.swift'
]

def copy_code_files():
    """Copy code files from repositories to their functional categories."""
    for category, repos in REPO_CATEGORIES.items():
        category_dir = FUNCTIONS_DIR / category
        
        # Ensure category directory exists
        category_dir.mkdir(exist_ok=True)
        
        for repo in repos:
            repo_path = REPO_DIR / repo
            if not repo_path.exists():
                print(f"Warning: Repository {repo} not found, skipping")
                continue
            
            # Create a directory for the repository in the category directory
            target_path = category_dir / repo
            if target_path.exists():
                shutil.rmtree(target_path)
            
            target_path.mkdir(parents=True)
            
            # Find and copy all code files
            code_files = []
            for ext in CODE_EXTENSIONS:
                code_files.extend(repo_path.glob(f'**/*{ext}'))
            
            # Copy each code file, preserving directory structure
            for file_path in code_files:
                rel_path = file_path.relative_to(repo_path)
                dest_path = target_path / rel_path
                
                # Skip files that would conflict with our own common directory
                if str(dest_path) == str(category_dir / 'common'):
                    continue
                
                # Create parent directories if they don't exist
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy the file
                shutil.copy2(file_path, dest_path)
            
            print(f"Copied code files from {repo} to {target_path}")

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
    # Copy code files
    copy_code_files()
    
    # Extract common functionality
    extract_common_functionality()
    
    print("Repository categorization complete!")

if __name__ == "__main__":
    main()