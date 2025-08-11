# Adapters for nlp
from .senpai_agent_adapter import SenpaiAgentAdapter

__all__ = ['SenpaiAgentAdapter']
from .senpai_agentlogger_adapter import SenpaiAgentLoggerAdapter
__all__.append('SenpaiAgentLoggerAdapter')
