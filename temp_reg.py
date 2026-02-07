

class OllamaRegistryInfo:
    """
    Reads model_registry.json and outputs as string for debugging.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "registry_path": ("STRING", {"default": "model_registry.json"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_content",)
    FUNCTION = "read_registry"
    CATEGORY = "Ollama/Debug"
    
    def read_registry(self, registry_path):
        try:
           registry = _read_registry(registry_path)
           import json
           return (json.dumps(registry, indent=2),)
        except Exception as e:
           return (f"Error reading registry: {e}",)
