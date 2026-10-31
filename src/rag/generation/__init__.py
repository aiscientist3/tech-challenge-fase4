"""Build the prompt for a later language-model call.

This package validates the retrieval payload, writes a context from the
reviews, and assembles the system and user messages. It does not call a model.
"""

from rag.generation.prompt import GenerationRequest, build_generation

__all__ = ["GenerationRequest", "build_generation"]
