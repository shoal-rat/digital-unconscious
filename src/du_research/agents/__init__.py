"""AI-powered agents for the Digital Unconscious pipeline.

Each agent wraps the AI backend with a specialised system prompt and a
task-specific interface.
"""

from du_research.agents.analysis_coder import AnalysisCoderAgent
from du_research.agents.briefing import BriefingAgent
from du_research.agents.compressor import CompressionAgent
from du_research.agents.idea_generator import IdeaGeneratorAgent
from du_research.agents.judge import JudgeAgent
from du_research.agents.reviewer import ReviewerAgent
from du_research.agents.revision import RevisionAgent
from du_research.agents.writer import WriterAgent

__all__ = [
    "AnalysisCoderAgent",
    "BriefingAgent",
    "CompressionAgent",
    "IdeaGeneratorAgent",
    "JudgeAgent",
    "ReviewerAgent",
    "RevisionAgent",
    "WriterAgent",
]
