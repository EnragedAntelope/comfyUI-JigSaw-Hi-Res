import torch
import torch.nn as nn
import math

from .rope import rope_dype


class JigsawPosEmbed(nn.Module):

    def __init__(
        self,
        axes_dim: list[int],
        theta: float = 10000.0,
        method: str = "yarn",
        rope_percentage: float = 1.0,
        dype: bool = True,
        ori_max_pe_len_spatial: int = 64,
        yarn_ramp_type: str = "linear",
        yarn_ratio: float = 1.0,
        yarn_beta_fast: int = 32,
        yarn_beta_slow: int = 1,
        timestep_modulation: bool = False,
        theta_period_min: float = 1.0,
        theta_period_max: float = 10000.0,
        attn_ratio: float = 1.0
    ):
        super().__init__()

        self.n_axes_init = len(axes_dim)

        for dim in axes_dim:
            assert dim % 2 == 0, f"Dimension {dim} muss gerade sein"

        if self.n_axes_init == 3:
            self.dim_axis0 = axes_dim[0]
            self.dim_spatial = axes_dim[1] + axes_dim[2]

        elif self.n_axes_init == 2:
            self.dim_spatial = axes_dim[0] + axes_dim[1]

        elif self.n_axes_init == 1:
            self.dim_axis0 = axes_dim[0]

        else:
            raise ValueError(
                "Nur 1, 2 oder 3 Achsen unterstützt"
            )

        self.axes_dim = axes_dim
        self.theta = theta
        self.method = method
        self.rope_percentage = rope_percentage
        self.dype = dype

        self.ori_max_pe_len_spatial = ori_max_pe_len_spatial

        self.yarn_ramp_type = yarn_ramp_type
        self.yarn_ratio = yarn_ratio
        self.yarn_beta_fast = yarn_beta_fast
        self.yarn_beta_slow = yarn_beta_slow

        self.timestep_modulation = timestep_modulation

        self.theta_period_min = theta_period_min
        self.theta_period_max = theta_period_max

        self.current_timestep = 0.0
        self.attn_ratio = attn_ratio


    def set_timestep(self, timestep):
        self.current_timestep = timestep


    def forward(self, ids):
        pos = ids.float()

        if pos.ndim != 3:
            raise RuntimeError(
                f"JigsawPosEmbed erwartet [B,N,Axes], bekam {pos.shape}"
            )

        n_axes = pos.shape[-1]
        if n_axes != len(self.axes_dim):
            raise RuntimeError(
                f"JigsawPosEmbed: ids hat {n_axes} Achsen, axes_dim hat aber "
                f"{len(self.axes_dim)} Eintraege ({self.axes_dim})"
            )

        shared = {
            "method": self.method,
            "rope_percentage": self.rope_percentage,
            "dype": self.dype,
            "yarn_ramp_type": self.yarn_ramp_type,
            "yarn_ratio": self.yarn_ratio,
            "yarn_beta_fast": self.yarn_beta_fast,
            "yarn_beta_slow": self.yarn_beta_slow,
            "current_timestep": self.current_timestep,
            "timestep_modulation": self.timestep_modulation,
            "theta_period_min": self.theta_period_min,
            "theta_period_max": self.theta_period_max,
            "attn_ratio": self.attn_ratio,
        }

        parts = [
            rope_dype(pos[..., i], dim=self.axes_dim[i], theta=self.theta, **shared)
            for i in range(n_axes)
        ]

        emb = torch.cat(parts, dim=-3)  # [B, N, headdim/2, 2, 2]
        return emb.unsqueeze(1).to(ids.device)  # [B, 1, N, headdim/2, 2, 2]


class JigsawHiRes:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "method": (
                    ["yarn", "ntk", "base"],
                    {"default": "yarn"}
                ),
                "rope_percentage": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}
                ),
                "dype_enabled": (
                    "BOOLEAN",
                    {"default": True}
                ),
                "max_pe_length": (
                    "INT",
                    {"default": 64, "min": 1, "max": 4096}
                ),
                "timestep_modulation": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "The 'Dy' in DyPE: blend the YaRN ratio from 0 "
                     "(fully extrapolated/untouched) at the start of sampling up to full strength "
                     "as sampling progresses, instead of applying one fixed scale for every step. "
                     "This follows the coarse-structure-first, fine-detail-later spectral progression "
                     "that the DyPE paper is built around. Off = old static-scale behavior."}
                )
            },
            "optional": {
                "latent": ("LATENT",)
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_jigsaw_patch"
    CATEGORY = "Jigsaw/Patches"

    def apply_jigsaw_patch(
        self,
        model,
        width,
        height,
        method,
        rope_percentage,
        dype_enabled,
        max_pe_length,
        timestep_modulation=True,
        latent=None
    ):
        m = model.clone()

        # Ziel-Auflösung ermitteln
        target_w = width
        target_h = height
        
        if latent is not None:
            # Latent Shape ist [B, C, H, W] - Flux und SD arbeiten mit Faktor 8
            target_h = latent["samples"].shape[2] * 8
            target_w = latent["samples"].shape[3] * 8

        # --- NEUE DIAGONAL-DÄMPFUNG GEGEN DEN GRAUSCHLEIER ---
        max_dim = max(target_w, target_h)
        linear_ratio = float(max_dim / 1024.0)
        area_ratio = float(((target_w * target_h) / (1024.0 * 1024.0)) ** 0.5)
        
        # Mittelwert fängt extreme Seitenverhältnisse ab
        calculated_ratio = (linear_ratio + area_ratio) / 2.0
        # Sanfter Deckel bei 1.25, damit der Kontrast stabil bleibt
        yarn_ratio = max(1.0, min(calculated_ratio, 1.25))
        
        print(f"[Jigsaw] Target Res: {target_w}x{target_h} | Adjusted Max-Dim YaRN Ratio: {yarn_ratio:.4f}")
        # -----------------------------------------------------

        try:
            orig = m.model.diffusion_model.pe_embedder
            theta = orig.theta
            axes_dim = orig.axes_dim
        except Exception:
            raise ValueError("Kein kompatibles Krea2T pe_embedder gefunden")

        new_embedder = JigsawPosEmbed(
            axes_dim=axes_dim,
            theta=theta,
            method=method,
            rope_percentage=rope_percentage,
            dype=dype_enabled,
            ori_max_pe_len_spatial=max_pe_length,
            yarn_ratio=yarn_ratio,
            timestep_modulation=timestep_modulation
        )

        m.add_object_patch("diffusion_model.pe_embedder", new_embedder)

        sigma_max = m.model.model_sampling.sigma_max.item()

        def jigsaw_wrapper(model_function, args_dict):
            timestep = args_dict.get("timestep")

            if timestep is not None and timestep.numel() > 0:
                sigma = timestep.item()
                if sigma_max > 0:
                    value = max(0.0, min(1.0, 1.0 - (sigma / sigma_max)))
                    new_embedder.set_timestep(value)

            return model_function(
                args_dict.get("input"),
                args_dict.get("timestep"),
                **args_dict.get("c", {})
            )

        m.set_model_unet_function_wrapper(jigsaw_wrapper)

        return (m,)


NODE_CLASS_MAPPINGS = {
    "JigsawHiRes": JigsawHiRes
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JigsawHiRes": "Jigsaw Hi-Res"
}
