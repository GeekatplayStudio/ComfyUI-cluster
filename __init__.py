"""ComfyUI-Cluster by Geekatplay Studio: Ollama-driven model routing and loading nodes for ComfyUI."""

from .nodes_ollama_router import (
    OllamaPromptPlanner,
    OllamaVisionStylePlanner,
    OllamaVisionDualPlanner,
    DynamicCheckpointLoader,
    DynamicLoraStack,
    PreviewTextMerge,
    LiveStatus,
)

NODE_CLASS_MAPPINGS = {
    "OllamaPromptPlanner": OllamaPromptPlanner,
    "OllamaVisionStylePlanner": OllamaVisionStylePlanner,
    "OllamaVisionDualPlanner": OllamaVisionDualPlanner,
    "DynamicCheckpointLoader": DynamicCheckpointLoader,
    "DynamicLoraStack": DynamicLoraStack,
    "PreviewTextMerge": PreviewTextMerge,
    "LiveStatus": LiveStatus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaPromptPlanner": "Ollama Prompt Planner",
    "OllamaVisionStylePlanner": "Ollama Vision Style Planner",
    "OllamaVisionDualPlanner": "Ollama Vision Dual Planner",
    "DynamicCheckpointLoader": "Dynamic Checkpoint Loader (Universal)",
    "DynamicLoraStack": "Dynamic LoRA Stack",
    "PreviewTextMerge": "Preview Text Merge",
    "LiveStatus": "Live Status",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
