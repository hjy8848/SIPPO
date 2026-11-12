"""ROSA RL adapter exports."""

__all__ = ["ROSA_AgentManager"]


def __getattr__(name):
    if name == "ROSA_AgentManager":
        from qa_manager.ROSA.rosa_agent_manager import ROSA_AgentManager

        return ROSA_AgentManager
    raise AttributeError(name)
