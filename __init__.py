from .jigsaw_hi_res import JigsawHiRes

# Branding
print("===============================================================================")
print("  [ZEUS.ONL] Jigsaw Core Nodes successfully injected.")
print("===============================================================================")

# ===============================================================================
#  COMFYUI REGISTRY MAPPINGS
# ===============================================================================

NODE_CLASS_MAPPINGS = {
    "JigsawHiRes": JigsawHiRes
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JigsawHiRes": "🧩 Jigsaw Hi-Res"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
