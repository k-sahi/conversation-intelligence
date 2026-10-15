"""Real-time path. `live_notes` has no LiveKit dependency; `agent` requires livekit-agents."""
from .live_notes import LiveNotes

__all__ = ["LiveNotes"]
