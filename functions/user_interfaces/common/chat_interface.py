# Common chat_interface component for user_interfaces
class ChatInterface:
    """Basic chat interface component."""
    
    def __init__(self):
        """Initialize the chat interface."""
        self.messages = []
    
    def add_message(self, sender, content, timestamp=None):
        """Add a message to the chat."""
        import datetime
        timestamp = timestamp or datetime.datetime.now()
        message = {
            "sender": sender,
            "content": content,
            "timestamp": timestamp
        }
        self.messages.append(message)
        return message
    
    def get_messages(self, limit=None):
        """Get recent messages, optionally limited to a certain number."""
        if limit:
            return self.messages[-limit:]
        return self.messages
    
    def clear_messages(self):
        """Clear all messages."""
        self.messages = []
