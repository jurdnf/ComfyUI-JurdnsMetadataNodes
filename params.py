import comfy.samplers

class JurdnsPromptParameters:
    """
    A comprehensive parameter node that handles prompts and all key generation settings.
    Outputs sampler and scheduler twice - once as their proper types for KSampler, 
    once as strings for metadata save nodes.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "STRING", "STRING", "INT", "FLOAT", "FLOAT", "INT")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "sampler_for_ksampler", "scheduler_for_ksampler", "sampler_for_metadata", "scheduler_for_metadata", "steps", "cfg", "denoise", "seed")
    FUNCTION = "get_parameters"
    CATEGORY = "Jurdns/sampling"

    def get_parameters(self, positive_prompt, negative_prompt, sampler_name, scheduler, steps, cfg, denoise, seed):
        """
        Returns all generation parameters. Sampler and scheduler are output twice:
        - Once as their proper types for KSampler
        - Once as strings for metadata save nodes
        All other parameters have single outputs that work for both uses.
        """
        return (positive_prompt, negative_prompt, sampler_name, scheduler, sampler_name, scheduler, steps, cfg, denoise, seed)

# Register the node
NODE_CLASS_MAPPINGS = {
    "JurdnsPromptParameters": JurdnsPromptParameters
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JurdnsPromptParameters": "Jurdns Prompt Parameters"
}