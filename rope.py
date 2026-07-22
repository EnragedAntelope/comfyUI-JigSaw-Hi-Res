import math
import torch
from einops import rearrange

import comfy.model_management


def _scaled_omega(
    dim,
    theta,
    device,
    method: str = "yarn",
    rope_percentage: float = 1.0,
    dype: bool = True,
    yarn_ramp_type: str = "linear",
    yarn_ratio: float = 1.0,
    yarn_beta_fast: int = 32,
    yarn_beta_slow: int = 1,
    current_timestep: float = 0.0,
    timestep_modulation: bool = False,
    theta_period_min: float = 1.0,
    theta_period_max: float = 10000.0,
    attn_ratio: float = 1.0,
    **kwargs
):
    """
    Baut omega (= inv_freq) exakt wie comfy.ldm.flux.math.rope(), aber mit
    optionaler YaRN/NTK-Skalierung und Timestep-Blend. Nutzt float64 wie das
    Original, damit die Praezision fuer hohe Aufloesungen erhalten bleibt.
    """
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=device)
    base_omega = 1.0 / (theta ** scale)

    if not dype or method == "base":
        return base_omega * rope_percentage

    if timestep_modulation:
        blend = max(0.0, min(1.0, float(current_timestep)))
    else:
        blend = 1.0

    effective_ratio = 1.0 + (yarn_ratio - 1.0) * blend
    if effective_ratio <= 1.0 + 1e-6:
        return base_omega * rope_percentage

    if method == "ntk":
        scaled_theta = theta * (effective_ratio ** (dim / max(dim - 2, 1)))
        scaled_theta = max(theta_period_min, min(theta_period_max, scaled_theta))
        omega = 1.0 / (scaled_theta ** scale)
        return omega * rope_percentage

    # method == "yarn": pro-Frequenz unterschiedliche Interpolation
    omega_extrapolated = base_omega
    omega_interpolated = base_omega / effective_ratio

    n_freqs = base_omega.shape[0]
    idx = torch.arange(n_freqs, dtype=torch.float64, device=device)

    beta_fast = max(1, int(yarn_beta_fast))
    beta_slow = max(1, int(yarn_beta_slow))
    low = min(beta_slow, beta_fast)
    high = max(beta_slow, beta_fast)
    high = max(high, low + 1)

    if yarn_ramp_type == "linear":
        ramp = (idx - low) / (high - low)
    else:
        ramp = 0.5 - 0.5 * torch.cos(math.pi * (idx - low) / (high - low))

    ramp = torch.clamp(ramp, 0.0, 1.0)
    mask = 1.0 - ramp  # 1 = voll interpoliert (gedehnt), 0 = unveraendert (extrapoliert)

    omega = omega_interpolated * mask + omega_extrapolated * (1.0 - mask)
    return omega * rope_percentage


def rope_dype(pos: torch.Tensor, dim: int, theta: int, **kwargs) -> torch.Tensor:
    """
    Drop-in-Ersatz fuer comfy.ldm.flux.math.rope() mit identischem Ausgabeformat
    ([..., N, dim/2, 2, 2] - volle 2x2-Rotationsmatrizen), aber mit YaRN/NTK/
    Timestep-skaliertem omega statt starrem 1/theta**scale.

    WICHTIG: Format muss exakt zu comfy.ldm.flux.math.rope() passen, da Krea2s
    nativer comfy_kitchen-RoPE-Kernel dieses Format hart erwartet (siehe
    apply_rope1 in math.py). Kein eigenes cos/sin-Paar-Format mehr wie zuvor.
    """
    assert dim % 2 == 0
    if not comfy.model_management.supports_fp64(pos.device):
        device = torch.device("cpu")
    else:
        device = pos.device

    omega = _scaled_omega(dim, theta, device, **kwargs)

    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega.to(torch.float32))
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)
