# Common agent_base component for agent_frameworks
class Agent:
    """Base agent class that provides common functionality for all agents."""
    
    def __init__(self, name, config=None):
        """Initialize the agent with a name and optional configuration."""
        self.name = name
        self.config = config or {}
        self.memory = []
        self.tools = []
        self.state = "idle"
    
    def add_tool(self, tool):
        """Add a tool to the agent's toolkit."""
        self.tools.append(tool)
    
    def add_to_memory(self, item):
        """Add an item to the agent's memory."""
        self.memory.append(item)
    
    def plan(self, goal):
        """Create a plan to achieve a goal."""
        # Implementation depends on specific agent architecture
        raise NotImplementedError("Subclasses must implement plan()")
    
    def execute(self, plan):
        """Execute a plan."""
        # Implementation depends on specific agent architecture
        raise NotImplementedError("Subclasses must implement execute()")
    
    def run(self, goal):
        """Run the agent to achieve a goal."""
        plan = self.plan(goal)
        return self.execute(plan)
