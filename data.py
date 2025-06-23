import os
import re
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import torch
import comfy.utils
import folder_paths
import hashlib

class JurdnsMetadataImageSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "model": ("MODEL", ),
                "model_widget_name": ("STRING", {"default": "ckpt_name"}),
                "positive_prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"forceInput": True}),
                "sampler_name": ("STRING", {"forceInput": True}),
                "scheduler": ("STRING", {"forceInput": True}),
                "steps": ("INT", {"forceInput": True}),
                "cfg": ("FLOAT", {"forceInput": True}),
                "seed": ("INT", {"forceInput": True}),
                "denoise": ("FLOAT", {"forceInput": True}),
            },
            "optional": {
                "lora_stack": ("LORA_STACK", ),
            },
            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Jurdns/image"

    def get_sha256(self, file_path: str):
        """Calculate SHA256 hash of file, using RvTools method"""
        file_no_ext = os.path.splitext(file_path)[0]
        hash_file = file_no_ext + ".sha256"

        if os.path.exists(hash_file):
            try:
                with open(hash_file, "r") as f:
                    return f.read().strip()
            except OSError as e:
                print(f"Error reading existing hash file: {e}")

        sha256_hash = hashlib.sha256()
        print(f"Hashing File (SHA256): {file_path}")

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        try:
            with open(hash_file, "w") as f:
                f.write(sha256_hash.hexdigest())
        except OSError as e:
            print(f"Error writing hash to {hash_file}: {e}")

        return sha256_hash.hexdigest()

    def get_model_info_from_source_node(self, model_widget_name, prompt=None):
        """Trace MODEL input back to source node and read widget value"""
        try:
            if prompt is None:
                return "unknown_model", "unknown_hash"
            
            # Find this node in the prompt
            current_node_id = None
            for node_id, node_data in prompt.items():
                if node_data.get("class_type") == "JurdnsMetadataImageSave":
                    current_node_id = node_id
                    break
            
            if current_node_id is None:
                print("Could not find current node in prompt")
                return "unknown_model", "unknown_hash"
            
            # Get the model input connection
            model_input = prompt[current_node_id]["inputs"].get("model")
            if not isinstance(model_input, list) or len(model_input) != 2:
                print("Model input is not a connection")
                return "unknown_model", "unknown_hash"
            
            source_node_id, source_output_index = model_input
            
            # Get the source node
            if source_node_id not in prompt:
                print(f"Source node {source_node_id} not found in prompt")
                return "unknown_model", "unknown_hash"
            
            source_node = prompt[source_node_id]
            
            # Get the widget value from source node
            model_name = source_node["inputs"].get(model_widget_name)
            if model_name is None:
                print(f"Widget '{model_widget_name}' not found in source node")
                return "unknown_model", "unknown_hash"
            
            # Find model path using RvTools method
            ckpt_path = folder_paths.get_full_path("checkpoints", model_name)
            diffusion_path = folder_paths.get_full_path("diffusion_models", model_name) 
            unet_path = folder_paths.get_full_path("unet", model_name)
            
            model_path = None
            if ckpt_path:
                model_path = ckpt_path
            elif diffusion_path:
                model_path = diffusion_path
            elif unet_path:
                model_path = unet_path
            
            if model_path:
                model_hash = self.get_sha256(model_path)[:10]
                model_filename = os.path.basename(model_name)
                return model_filename, model_hash
            else:
                print(f"Could not find model file for: {model_name}")
                return model_name, "unknown_hash"
                
        except Exception as e:
            print(f"Error tracing model from source node: {e}")
            return "unknown_model", "unknown_hash"

    def full_lora_path_for(self, lora: str):
        """Find LoRA path using RvTools method"""
        # Find the position of the last dot
        last_dot_position = lora.rfind('.')
        # Get the extension including the dot
        extension = lora[last_dot_position:] if last_dot_position != -1 else ""
        # Check if the extension is supported, if not, add .safetensors
        if extension not in folder_paths.supported_pt_extensions:
            lora += ".safetensors"

        # Find the matching lora path
        lora_list = folder_paths.get_filename_list("loras")
        matching_lora = next((x for x in lora_list if x.endswith(lora)), None)
        if matching_lora is None:
            print(f'Could not find full path to lora "{lora}"')
            return None
        return folder_paths.get_full_path("loras", matching_lora)

    def get_lora_info_from_stack(self, lora_stack):
        """Extract LoRA info from LORA_STACK using RvTools method"""
        if not lora_stack:
            return ""
        
        lora_info = []
        try:
            # LORA_STACK format: list of (lora_name, strength_model, strength_clip)
            for lora_entry in lora_stack:
                if isinstance(lora_entry, tuple) and len(lora_entry) >= 1:
                    lora_name = lora_entry[0]
                    
                    # Find LoRA path
                    lora_path = self.full_lora_path_for(lora_name)
                    if lora_path:
                        lora_hash = self.get_sha256(lora_path)[:10]
                        # Remove file extension for cleaner display
                        clean_name = os.path.splitext(lora_name)[0]
                        lora_info.append(f"{clean_name}: {lora_hash}")
                    else:
                        # Fallback if path not found
                        clean_name = os.path.splitext(lora_name)[0]
                        lora_info.append(f"{clean_name}: unknown")
            
            return ", ".join(lora_info)
            
        except Exception as e:
            print(f"Error extracting LoRA info: {e}")
            return ""

    def create_metadata_string(self, positive_prompt, negative_prompt, model_name, model_hash, 
                             sampler_name, scheduler, steps, cfg, seed, denoise, lora_hashes=""):
        """Create Civitai-compatible metadata string"""
        
        # Base parameters
        params = []
        
        # Add steps, sampler, cfg, seed, etc.
        params.append(f"Steps: {steps}")
        params.append(f"Sampler: {sampler_name}")
        params.append(f"CFG scale: {cfg}")
        params.append(f"Seed: {seed}")
        
        if denoise < 1.0:
            params.append(f"Denoising strength: {denoise}")
        
        # Model info
        params.append(f"Model hash: {model_hash}")
        params.append(f"Model: {model_name}")
        
        # LoRA info if provided
        if lora_hashes.strip():
            params.append(f"Lora hashes: \"{lora_hashes}\"")
        
        # Scheduler
        params.append(f"Schedule type: {scheduler}")
        
        # Version (ComfyUI identifier)
        params.append("Version: ComfyUI")
        
        # Combine everything
        metadata_string = positive_prompt
        if negative_prompt.strip():
            metadata_string += f"\nNegative prompt: {negative_prompt}"
        
        metadata_string += f"\n{', '.join(params)}"
        
        return metadata_string

    def save_images(self, images, filename_prefix="ComfyUI", model=None, model_widget_name="ckpt_name",
                   positive_prompt="", negative_prompt="", sampler_name="euler", scheduler="normal", 
                   steps=20, cfg=8.0, seed=0, denoise=1.0, lora_stack=None, prompt=None, extra_pnginfo=None):
        
        # Get model info by tracing back to source node
        model_name, model_hash = self.get_model_info_from_source_node(model_widget_name, prompt)
        
        # Get LoRA info from LORA_STACK
        lora_hashes = self.get_lora_info_from_stack(lora_stack)
        
        # Create metadata string
        metadata_string = self.create_metadata_string(
            positive_prompt, negative_prompt, model_name, model_hash,
            sampler_name, scheduler, steps, cfg, seed, denoise, lora_hashes
        )
        
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        
        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Create PNG info with metadata
            metadata = PngInfo()
            metadata.add_text("parameters", metadata_string)
            
            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }

# Register the node
NODE_CLASS_MAPPINGS = {
    "JurdnsMetadataImageSave": JurdnsMetadataImageSave
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JurdnsMetadataImageSave": "Jurdns Metadata Image Save"
}