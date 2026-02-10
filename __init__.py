from .nodes_ollama_router_new import (
    OllamaPromptPlanner,
    OllamaVisionStylePlanner,
    DynamicCheckpointLoader,
    DynamicLoraStack,
    PreviewTextMerge,
    LiveStatus,
)

NODE_CLASS_MAPPINGS = {
    "OllamaPromptPlanner": OllamaPromptPlanner,
    "OllamaVisionStylePlanner": OllamaVisionStylePlanner,
    "DynamicCheckpointLoader": DynamicCheckpointLoader,
    "DynamicLoraStack": DynamicLoraStack,
    "PreviewTextMerge": PreviewTextMerge,
    "LiveStatus": LiveStatus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaPromptPlanner": "Ollama Prompt Planner",
    "OllamaVisionStylePlanner": "Ollama Vision Style Planner",
    "DynamicCheckpointLoader": "Dynamic Checkpoint Loader (Universal)",
    "DynamicLoraStack": "Dynamic LoRA Stack",
    "PreviewTextMerge": "Preview Text Merge",
    "LiveStatus": "Live Status",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
