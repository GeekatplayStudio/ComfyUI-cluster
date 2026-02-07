from .nodes_ollama_router_new import (
    OllamaPromptPlanner,
    OllamaVisionStylePlanner,
    DynamicCheckpointLoader,
    DynamicLoraStack,
    RouteConditioningByType,
    OptionalControlNetApply,
    RouteLatentByBool,
    RouteVaeByBool,
    OllamaDebugInfo,
    OllamaRegistryInfo,
)

NODE_CLASS_MAPPINGS = {
    "OllamaPromptPlanner": OllamaPromptPlanner,
    "OllamaVisionStylePlanner": OllamaVisionStylePlanner,
    "DynamicCheckpointLoader": DynamicCheckpointLoader,
    "DynamicLoraStack": DynamicLoraStack,
    "RouteConditioningByType": RouteConditioningByType,
    "OptionalControlNetApply": OptionalControlNetApply,
    "RouteLatentByBool": RouteLatentByBool,
    "RouteVaeByBool": RouteVaeByBool,
    "OllamaDebugInfo": OllamaDebugInfo,
    "OllamaRegistryInfo": OllamaRegistryInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaPromptPlanner": "Ollama Prompt Planner",
    "OllamaVisionStylePlanner": "Ollama Vision Style Planner",
    "DynamicCheckpointLoader": "Dynamic Checkpoint Loader (Universal)",
    "DynamicLoraStack": "Dynamic LoRA Stack",
    "RouteConditioningByType": "Route Conditioning (SDXL/SD1.5)",
    "OptionalControlNetApply": "Optional ControlNet Apply",
    "RouteLatentByBool": "Route Latent (Bool)",
    "RouteVaeByBool": "Route VAE (Bool)",
    "OllamaDebugInfo": "Ollama Plan Debug Info",
    "OllamaRegistryInfo": "Ollama Registry Debug Info",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
