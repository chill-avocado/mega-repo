# AI/AGI Mega Repository

This repository contains a collection of AI and AGI-related repositories, organized by functionality.

## Repository Structure

- **`repos/`**: Contains the original repositories
- **`functions/`**: Contains the functional categorization of the repositories
  - Each subdirectory represents a specific functional category
  - Each category contains copies of the code files from relevant repositories
  - Each category has a `common/` directory with reusable components

## Functional Categories

1. **Agent Frameworks**: Repositories focused on creating autonomous AI agents (AgentForge, AgentGPT, AgentK, Auto-GPT-MetaTrader-Plugin, Free-Auto-GPT, agent-zero, autogen, babyagi, Teenage-AGI, openhands)
2. **User Interfaces**: Repositories focused on UI/UX for AI systems (open-webui, web-ui, draw-a-ui)
3. **OS Interaction**: Repositories for AI interaction with operating systems (MacOS-Agent, self-operating-computer, mcp-remote-macos-use)
4. **Browser Automation**: Repositories for AI-driven browser interaction (browser-use)
5. **Code Execution**: Repositories for code generation and execution (open-interpreter, pyCodeAGI, SuperCoder, automata)
6. **Cognitive Systems**: Repositories focused on cognitive architectures (opencog, openagi, AGI-Samantha)
7. **Evolution & Optimization**: Repositories for evolutionary algorithms (openevolve)
8. **Integration**: Repositories that combine multiple systems (Unification, LocalAGI)
9. **NLP**: Repositories focused on natural language processing (senpai, html-agility-pack)
10. **Documentation**: Repositories that are collections or documentation (Awesome-AGI, awesome-agi-cocosci)

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
- Unified interfaces in the `unified/` directory

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

### Using Unified Interfaces

Each functional category now includes a unified interface that allows you to access functionality from any repository through a consistent API. The unified interfaces are located in the `unified/` directory of each category.

For example, to use the agent frameworks unified interface:

```python
from functions.agent_frameworks.unified import AgentFrameworksFactory

# Create a unified interface with the default implementation
agent = AgentFrameworksFactory.create()

# Create a unified interface with a specific implementation
agent = AgentFrameworksFactory.create('AgentForge/Agent')

# Use the unified interface
result = agent.run("Achieve this goal")
```

The unified interfaces provide a consistent API across different implementations, making it easy to switch between different repositories without changing your code.

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