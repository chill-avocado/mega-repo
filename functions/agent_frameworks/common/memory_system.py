# Common memory_system component for agent_frameworks
class Memory:
    """Memory system for agents to store and retrieve information."""
    
    def __init__(self, capacity=100):
        """Initialize the memory system with a capacity."""
        self.capacity = capacity
        self.items = []
    
    def add(self, item):
        """Add an item to memory."""
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items.pop(0)  # Remove oldest item if capacity exceeded
    
    def get_all(self):
        """Get all items in memory."""
        return self.items
    
    def search(self, query):
        """Search memory for items matching a query."""
        # Simple string matching for now
        return [item for item in self.items if query in str(item)]
    
    def clear(self):
        """Clear all items from memory."""
        self.items = []
