#!/usr/bin/env python3
"""
Script to analyze repositories and extract common functionality patterns.
This script identifies key components and creates a detailed mapping of functionality.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

# Base directories
REPO_DIR = Path('repos')
FUNCTIONS_DIR = Path('functions')

# Functionality patterns to look for
FUNCTIONALITY_PATTERNS = {
    'agent_frameworks': {
        'agent_lifecycle': ['agent.*init', 'create.*agent', 'agent.*factory'],
        'planning': ['plan', 'task.*planning', 'goal.*decomposition'],
        'memory': ['memory', 'context.*management', 'history'],
        'tool_use': ['tool', 'action', 'capability'],
        'multi_agent': ['multi.*agent', 'agent.*communication', 'collaboration']
    },
    'user_interfaces': {
        'chat_interface': ['chat', 'message', 'conversation'],
        'components': ['component', 'button', 'input', 'form'],
        'visualization': ['visualize', 'chart', 'graph', 'display'],
        'responsive': ['responsive', 'mobile', 'desktop', 'layout'],
        'authentication': ['auth', 'login', 'user.*session']
    },
    'os_interaction': {
        'screen_capture': ['screen.*capture', 'screenshot', 'image.*processing'],
        'input_control': ['keyboard', 'mouse', 'input.*control'],
        'app_control': ['launch.*app', 'application.*control', 'window.*management'],
        'file_system': ['file.*system', 'file.*operation', 'directory'],
        'system_monitoring': ['monitor', 'system.*status', 'performance']
    },
    'browser_automation': {
        'navigation': ['navigate', 'url', 'browser.*control'],
        'interaction': ['click', 'type', 'form.*fill'],
        'scraping': ['scrape', 'extract.*data', 'parse.*html'],
        'automation': ['automate', 'workflow', 'task.*sequence'],
        'headless': ['headless', 'browser.*emulation']
    },
    'code_execution': {
        'code_generation': ['generate.*code', 'code.*completion', 'programming'],
        'execution': ['execute', 'run.*code', 'sandbox'],
        'analysis': ['analyze.*code', 'lint', 'code.*quality'],
        'language_support': ['language.*support', 'python', 'javascript'],
        'development': ['development.*environment', 'ide', 'editor']
    },
    'cognitive_systems': {
        'knowledge_representation': ['knowledge.*representation', 'ontology', 'semantic'],
        'reasoning': ['reasoning', 'inference', 'logic'],
        'learning': ['learning', 'adaptation', 'training'],
        'memory_systems': ['memory.*system', 'episodic', 'semantic.*memory'],
        'cognitive_processes': ['cognitive.*process', 'attention', 'perception']
    },
    'evolution_optimization': {
        'evolutionary_algorithms': ['evolution', 'genetic.*algorithm', 'mutation'],
        'optimization': ['optimize', 'parameter.*tuning', 'hyperparameter'],
        'benchmarking': ['benchmark', 'performance.*measure', 'evaluation'],
        'adaptive_learning': ['adaptive', 'reinforcement.*learning', 'feedback'],
        'multi_objective': ['multi.*objective', 'pareto', 'trade.*off']
    },
    'integration': {
        'system_integration': ['integration', 'connect.*system', 'interoperability'],
        'api_standardization': ['api.*standard', 'interface.*definition', 'protocol'],
        'compatibility': ['compatibility', 'cross.*platform', 'portability'],
        'unified_interface': ['unified.*interface', 'common.*ui', 'standardized.*control'],
        'orchestration': ['orchestration', 'component.*management', 'service.*coordination']
    },
    'nlp': {
        'text_understanding': ['text.*understanding', 'nlp', 'language.*model'],
        'conversation': ['conversation', 'dialogue', 'chat.*management'],
        'sentiment': ['sentiment', 'emotion', 'affect'],
        'entity_recognition': ['entity.*recognition', 'ner', 'extraction'],
        'document_processing': ['document.*processing', 'text.*extraction', 'parsing']
    }
}

def analyze_repository(repo_name, category):
    """Analyze a repository for specific functionality patterns."""
    repo_path = REPO_DIR / repo_name
    if not repo_path.exists():
        return {}
    
    results = defaultdict(list)
    
    # Get all Python, JavaScript, and TypeScript files
    code_files = []
    for ext in ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cs', '.go', '.rs']:
        code_files.extend(repo_path.glob(f'**/*{ext}'))
    
    # Look for patterns in each file
    for file_path in code_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Check for each functionality pattern
                for func_type, patterns in FUNCTIONALITY_PATTERNS.get(category, {}).items():
                    for pattern in patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            rel_path = file_path.relative_to(repo_path)
                            results[func_type].append(str(rel_path))
                            break  # Found a match for this functionality type
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return dict(results)

def create_functionality_index():
    """Create an index of functionality across repositories."""
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
    
    # Analyze each repository
    functionality_index = {}
    for category, repos in repo_categories.items():
        functionality_index[category] = {}
        for repo in repos:
            print(f"Analyzing {repo} for {category} functionality...")
            functionality_index[category][repo] = analyze_repository(repo, category)
    
    # Save the functionality index
    with open(FUNCTIONS_DIR / 'functionality_index.json', 'w') as f:
        json.dump(functionality_index, f, indent=2)
    
    # Create a markdown summary
    with open(FUNCTIONS_DIR / 'functionality_summary.md', 'w') as f:
        f.write("# Functionality Summary\n\n")
        f.write("This document provides a summary of the functionality found in each repository.\n\n")
        
        for category, repos in functionality_index.items():
            f.write(f"## {category.replace('_', ' ').title()}\n\n")
            
            for repo, functions in repos.items():
                f.write(f"### {repo}\n\n")
                
                if not functions:
                    f.write("No specific functionality patterns detected.\n\n")
                    continue
                
                for func_type, files in functions.items():
                    f.write(f"#### {func_type.replace('_', ' ').title()}\n\n")
                    if files:
                        f.write("Found in files:\n")
                        for file in sorted(files)[:10]:  # Limit to 10 files to avoid too much output
                            f.write(f"- `{file}`\n")
                        if len(files) > 10:
                            f.write(f"- ... and {len(files) - 10} more files\n")
                    else:
                        f.write("No files found for this functionality.\n")
                    f.write("\n")
            
            f.write("\n")
    
    print("Functionality analysis complete!")

if __name__ == "__main__":
    create_functionality_index()