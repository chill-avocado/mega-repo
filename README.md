# AI/AGI Mega Repository

This repository contains a collection of AI and AGI-related repositories, organized by functionality.

## Repository Structure

- **`repos/`**: Contains the original repositories
- **`functions/`**: Contains the functional categorization of the repositories
  - Each subdirectory represents a specific functional category
  - Each category contains copies of the code files from relevant repositories
  - Each category has a `common/` directory with reusable components

## Functional Categories

1. **Agent Frameworks**: Repositories focused on creating autonomous AI agents
2. **User Interfaces**: Repositories focused on UI/UX for AI systems
3. **OS Interaction**: Repositories for AI interaction with operating systems
4. **Browser Automation**: Repositories for AI-driven browser interaction
5. **Code Execution**: Repositories for code generation and execution
6. **Cognitive Systems**: Repositories focused on cognitive architectures
7. **Evolution & Optimization**: Repositories for evolutionary algorithms
8. **Integration**: Repositories that combine multiple systems
9. **NLP**: Repositories focused on natural language processing
10. **Documentation**: Repositories that are collections or documentation

## How to Use This Repository

### Exploring by Functionality

If you're interested in a specific functionality, navigate to the corresponding directory in `functions/`. For example, if you're interested in agent frameworks:

```bash
cd functions/agent_frameworks
```

Here you'll find:
- A README explaining the category
- Copies of code files from relevant repositories
- Common components in the `common/` directory

### Using Common Components

Each functional category includes common components that can be reused in your projects. These components are located in the `common/` directory of each category.

For example, to use the agent base component:

```python
from functions.agent_frameworks.common.agent_base import Agent

# Create a custom agent
class MyAgent(Agent):
    def plan(self, goal):
        # Implement planning logic
        return ["Step 1", "Step 2", "Step 3"]
    
    def execute(self, plan):
        # Implement execution logic
        return "Result"

# Use the agent
agent = MyAgent("MyAgent")
result = agent.run("Achieve this goal")
```

### Finding Repositories by Functionality

To find repositories that implement a specific functionality, refer to the index:

```bash
cat functions/index.md
```

This index provides a comprehensive list of all repositories categorized by functionality.

## Scripts

The repository includes several scripts to help you work with the categorized code:

- **`categorize_repos.py`**: Copies code files from repositories to their functional categories
- **`analyze_functionality.py`**: Analyzes repositories for specific functionality patterns
- **`extract_common_components.py`**: Extracts common components from repositories
- **`generate_index.py`**: Generates a comprehensive index of repositories and their functionality

## Contributing

To contribute to this repository:

1. Add your code to the appropriate repository in `repos/`
2. Run the categorization scripts to update the functional categorization
3. Update the common components if necessary
4. Regenerate the index

```bash
python3 categorize_repos.py
python3 analyze_functionality.py
python3 extract_common_components.py
python3 generate_index.py
```