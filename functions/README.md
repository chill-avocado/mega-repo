# Functional Categorization of AI/AGI Repositories

This directory contains a functional categorization of the various repositories in the mega-repo. Each subdirectory represents a specific functional category and contains copies of the code files from the original repositories, as well as unified interfaces for accessing functionality across repositories.

## Categories

1. **[Agent Frameworks](./agent_frameworks/)**: Repositories focused on creating autonomous AI agents
2. **[User Interfaces](./user_interfaces/)**: Repositories focused on UI/UX for AI systems
3. **[OS Interaction](./os_interaction/)**: Repositories for AI interaction with operating systems
4. **[Browser Automation](./browser_automation/)**: Repositories for AI-driven browser interaction
5. **[Code Execution](./code_execution/)**: Repositories for code generation and execution
6. **[Cognitive Systems](./cognitive_systems/)**: Repositories focused on cognitive architectures
7. **[Evolution & Optimization](./evolution_optimization/)**: Repositories for evolutionary algorithms
8. **[Integration](./integration/)**: Repositories that combine multiple systems
9. **[NLP](./nlp/)**: Repositories focused on natural language processing
10. **[Documentation](./documentation/)**: Repositories that are collections or documentation

## How to Use This Categorization

Each category directory contains:
- A README explaining the category and included repositories
- Copies of code files from the original repositories
- Common functionality extracted and organized by purpose
- A `unified` directory with a unified interface for the category

You can navigate to a specific category to find all related code across different repositories, making it easier to understand and work with functionally similar components.

## Unified Interfaces

Each functional category now includes a `unified` directory with a unified interface that allows you to access functionality from any repository through a consistent API. The unified interfaces follow the Adapter pattern, providing a common interface for different implementations.

### Structure

Each unified interface consists of:

- **Base Class**: Defines the common interface for all implementations
- **Unified Interface**: Provides a consistent API for accessing different implementations
- **Adapters**: Adapt specific repository implementations to the unified interface
- **Factory**: Creates instances of the unified interface with specific implementations
- **Example**: Demonstrates how to use the unified interface

### Using the Unified Interfaces

To use a unified interface, you can use the factory to create an instance with a specific implementation:

```python
from functions.agent_frameworks.unified import AgentFrameworksFactory

# Create a unified interface with the default implementation
agent = AgentFrameworksFactory.create()

# Create a unified interface with a specific implementation
agent = AgentFrameworksFactory.create('AgentForge/Agent')

# Use the unified interface
result = agent.run("Achieve this goal")
```

### Available Unified Interfaces

1. **Agent Frameworks**: `functions/agent_frameworks/unified`
   - Implementations: Free-Auto-GPT/Agent, AgentForge/Agent, autogen/SingleThreadedAgentRuntime
   - Methods: initialize, plan, execute, run, add_tool, add_memory, get_state

2. **User Interfaces**: `functions/user_interfaces/unified`
   - Implementations: open-webui/CustomBuildHook, open-webui/PersistentConfig, open-webui/RedisDict
   - Methods: initialize, render, handle_input, update, get_state

3. **OS Interaction**: `functions/os_interaction/unified`
   - Implementations: self-operating-computer/OperatingSystem, mcp-remote-macos-use/VncClient, MacOS-Agent/DeferredLogger
   - Methods: initialize, capture_screen, control_keyboard, control_mouse, execute_command, get_state

4. **Browser Automation**: `functions/browser_automation/unified`
   - Implementations: browser-use/BrowserUseApp, browser-use/BrowserUseServer, browser-use/FileSystem
   - Methods: initialize, navigate, click, type, extract_content, get_state

5. **Code Execution**: `functions/code_execution/unified`
   - Implementations: automata/ToolEvaluationHarness, automata/AgentEvaluationHarness, automata/PyCodeWriterToolkitBuilder
   - Methods: initialize, execute, analyze, generate, get_state

6. **Cognitive Systems**: `functions/cognitive_systems/unified`
   - Implementations: openagi/MemoryRAGAction, openagi/BaseMemory, opencog/HobbsAgent
   - Methods: initialize, process, learn, reason, get_state

7. **Evolution & Optimization**: `functions/evolution_optimization/unified`
   - Implementations: openevolve/BulletproofMetalEvaluator, openevolve/MLIRAttentionEvaluator, openevolve/ProgramDatabase
   - Methods: initialize, evolve, evaluate, select, get_state

8. **Integration**: `functions/integration/unified`
   - Implementations: Unification/VncClient, Unification/DeferredLogger, Unification/OperatingSystem
   - Methods: initialize, connect, process, transform, get_state

9. **NLP**: `functions/nlp/unified`
   - Implementations: senpai/Agent, senpai/AgentLogger
   - Methods: initialize, process_text, generate_text, analyze_text, get_state

10. **Documentation**: `functions/documentation/unified`
    - Implementations: placeholder/Documentation
    - Methods: initialize, get_documentation, search, generate_documentation, get_state

## Benefits of Unified Interfaces

1. **Consistency**: Access functionality from different repositories through a consistent API
2. **Interoperability**: Mix and match components from different repositories
3. **Extensibility**: Easily add new implementations without changing client code
4. **Abstraction**: Hide implementation details behind a clean interface
5. **Testability**: Test code against the unified interface instead of specific implementations

## Examples

Each unified interface includes an example file that demonstrates how to use the interface with different implementations. You can run these examples to see how the unified interfaces work:

```python
# Run the agent frameworks example
python -m functions.agent_frameworks.unified.example
```