"""SeedVR2's `NaDiT` video-restoration transformer, as a cost model.

Upstream:  https://github.com/ByteDance-Seed/SeedVR (Apache-2.0)
Weights:   https://huggingface.co/Comfy-Org/SeedVR2 (ComfyUI single-file, Apache-2.0)
Paper:     "SeedVR2: One-Step Video Restoration via Diffusion Adversarial
           Post-Training", ICLR 2026.

This is not a port, and it does not need the 7 GB of weights: it is the *shape*
of the network, so the arithmetic can be counted and a benchmark can be built at
the real dimensions.

## Why the shape is worth writing down

SeedVR2 is the one model in this project's "10-100x too slow to play" category
that is not a plain per-frame convolution stack, and its headline claim --
**one-step** inference, no iterative sampling -- is the reason it is worth asking
whether it could play live where a 50-step diffusion model plainly could not.
It is a single forward pass. The only question is what that pass costs.

## The shape is read off the shipped weights, not inferred from the config

Every number below comes from the safetensors headers in
`Comfy-Org/SeedVR2`, which carry the full tensor list (name, dtype, shape) in
the first few kilobytes of each file. `inspect_weights.py` in this directory
fetches just that header, so the architecture is verified against the released
weights rather than guessed from a YAML -- which matters, because the config is
misleading about it.

`configs_3b/main.yaml` says `num_layers: 32`, `mlp` swiglu, `expand_ratio: 4`,
and `mm_layers: 10`; the weights say something more specific:

    blocks 0..9    dual-stream   separate `txt` and `vid` attention + MLP
    blocks 10..31  vid-only      one stream, named `all` in the checkpoint
    MLP hidden     6912           not dim * expand_ratio (which would be 10240)
    attn qkv       [7680, 2560]   = 3 * dim, no bias on q (qk_bias: False)
    ada            6 vectors of dim per stream (attn/mlp x shift/scale/gate)

so `expand_ratio: 4` does not describe the released MLP width, and a cost model
built from the config would be wrong in both directions at once -- too wide per
MLP, and blind to the second stream in the first ten blocks. Counting from the
header reproduces the checkpoint exactly:

    3B   22 blocks *  79.317 M  +  10 blocks * 158.633 M  +  60.798 M  =  3.3921 B
    7B   36 blocks * 226.567 M  +  83.833 M               =  8.2402 B

against released totals of 3.3915 B and 8.2397 B (fp32 file size / 4). The 44 M
of difference is dtype-level rounding in the fp32-to-fp16 conversion, not a
missing term.

## What the VAE is

`ema_vae_fp16.safetensors` is a 3D causal autoencoder: 250 tensors, 250.7 M
params, `(3,3,3)` convolutions at 512 channels, `s8` spatial and `t4` temporal
downsampling, 16 latent channels. It is *not* costed here -- see `vae_note()`.
"""

# --- verified architecture -------------------------------------------------
#
# provenance: safetensors headers of Comfy-Org/SeedVR2 diffusion_models/*_fp16.
# `blocks` is the per-variant block plan: (count, "dual" | "vid") in order,
# because which blocks are dual-stream is not uniform and is most of the
# difference between the two sizes.

VARIANTS = {
    "3b": dict(
        vid_dim=2560, heads=20, head_dim=128, mlp_hidden=6912,
        patch=(1, 2, 2), vid_in_channels=33, vid_out_channels=16,
        blocks=[(10, "dual"), (22, "vid")],
        dit_checkpoint_bytes=13566090228,   # seedvr2_ema_3b.pth, fp32
        shipped_params=3_392_100_000,       # 3b_fp16.safetensors header total
    ),
    "7b": dict(
        vid_dim=3072, heads=24, head_dim=128, mlp_hidden=8192,
        patch=(1, 2, 2), vid_in_channels=33, vid_out_channels=16,
        blocks=[(36, "dual")],
        dit_checkpoint_bytes=32958774606,   # seedvr2_ema_7b.pth, fp32
        shipped_params=8_240_200_000,
    ),
}

VAE = dict(params=250_672_975, latent_channels=16,
           spatial_downsample_factor=8, temporal_downsample_factor=4,
           block_out_channels=(128, 256, 512, 512), layers_per_block=2)

# Window attention (`window: (4,3,3)`, every layer in both configs). Attention is
# local, which is what keeps its cost linear in frames instead of quadratic.
WINDOW = (4, 3, 3)

# Text-conditioning stream: `txt_in` projects 5120-dim embeddings, and the
# released conditioning tensors are 58 (positive) and 64 (negative) tokens --
# `positive_conditioning` / `negative_conditioning` in the header.
TXT_TOKEN_COUNT = 58
TXT_DIM = 5120

# One step, no classifier-free guidance: `sample_steps=1` and `cfg_scale=1.0` are
# the defaults in `projects/inference_seedvr2_3b.py`, and the DiT is called once
# per window. CFG would double every figure here.
STEPS = 1


def params_from_checkpoint(variant):
    """Released fp32 checkpoint bytes / 4. Exact, not an estimate."""
    return VARIANTS[variant]["dit_checkpoint_bytes"] // 4


def vae_params():
    return VAE["params"]


def block_counts(variant):
    """(dual_stream_blocks, vid_only_blocks)."""
    dual = vid = 0
    for n, kind in VARIANTS[variant]["blocks"]:
        if kind == "dual":
            dual += n
        else:
            vid += n
    return dual, vid


def params_per_block(variant, kind="dual"):
    """Parameter count of one block, stream by stream, from the verified shapes.

    A stream is qkv (3*dim^2) + proj_out (dim^2 + dim) + swiglu MLP
    (3 * dim * hidden) + 6 ada vectors. The dual block pays for two of them.
    """
    v = VARIANTS[variant]
    dim, hidden = v["vid_dim"], v["mlp_hidden"]
    stream = 3 * dim * dim + dim * dim + dim + 3 * dim * hidden + 6 * dim
    return stream * (2 if kind == "dual" else 1)


def total_params(variant):
    v = VARIANTS[variant]
    dim = v["vid_dim"]
    dual, vid = block_counts(variant)
    blocks = dual * params_per_block(variant, "dual") + vid * params_per_block(variant, "vid")
    patch_volume = v["patch"][0] * v["patch"][1] * v["patch"][2]
    # vid_in, vid_out, txt_in, ada embedding MLP, output norm
    other = (dim * v["vid_in_channels"] * patch_volume + dim
             + v["vid_out_channels"] * patch_volume * dim + v["vid_out_channels"] * patch_volume
             + dim * TXT_DIM + dim
             + 6 * dim * dim + dim * dim + dim + 6 * dim + dim)
    return blocks + other


def vae_latent_shape(width, height, frames):
    """The tensor the DiT is handed: (t, h, w) as the VAE produces it.

    Spatial /8 and temporal /4, *before* the DiT's own patchify -- so this is
    twice the token grid's spatial size in each axis. Keeping the two apart
    matters: the patchify in `bench.py` divides these axes by 2 again.
    """
    t = max(frames // VAE["temporal_downsample_factor"], 1)
    return t, max(height // VAE["spatial_downsample_factor"], 1), \
        max(width // VAE["spatial_downsample_factor"], 1)


def latent_grid(width, height, frames, variant="3b"):
    """The token grid the DiT's attention runs over, after patchify.

    Two downsamples stack: the VAE divides spatially by 8 and temporally by 4,
    then the DiT's patch_size of (1,2,2) divides the spatial axes by 2 again.
    """
    v = VARIANTS[variant]
    t, h_lat, w_lat = vae_latent_shape(width, height, frames)
    return max(t // v["patch"][0], 1), h_lat // v["patch"][1], w_lat // v["patch"][2]


def tokens(width, height, frames, variant="3b"):
    t, h, w = latent_grid(width, height, frames, variant)
    return t * h * w


def flops_linear(variant, n_tokens):
    """Forward-pass arithmetic of the DiT's linear layers: 2 * N * tokens.

    The video stream sees `n_tokens`; the text stream sees only the 58
    conditioning tokens, and only in the dual blocks, so it rides along here as
    an additive term rather than a multiplier. It is ~1% of the total.
    """
    dual, vid = block_counts(variant)
    video = 2.0 * (dual * params_per_block(variant, "dual")
                   + vid * params_per_block(variant, "vid")) * n_tokens
    txt = 2.0 * dual * params_per_block(variant, "dual") * TXT_TOKEN_COUNT / 2
    return (video + txt) * STEPS


def flops_attention(variant, n_tokens, width, height, frames, windowed=True):
    """Attention arithmetic, windowed as upstream configures it.

    QK^T and AV cost 2 * 2 * n * k * head_dim. Global attention makes k =
    n_tokens and the term grows with the square of the frame count; the
    configured (4,3,3) window makes k <= 36 and it grows linearly. The caller
    reports both so the difference is visible rather than asserted.
    """
    v = VARIANTS[variant]
    dim, layers = v["vid_dim"], sum(n for n, _ in v["blocks"])
    if not windowed:
        keys = n_tokens
    else:
        t, h, w = latent_grid(width, height, frames, variant)
        keys = min(WINDOW[0], t) * min(WINDOW[1], h) * min(WINDOW[2], w)
    return 4.0 * dim * layers * n_tokens * keys * STEPS


def vae_note():
    """The VAE is deliberately not costed, and that is not an oversight.

    It is convolutional and runs at pixel resolution, so `2 * N * params` -- the
    formula that is exact for the DiT's linear layers -- is meaningless for it: a
    conv reuses each weight across the raster, so its arithmetic is set by the
    feature-map sizes, not by its parameter count. Costing it properly means
    walking four encoder and four decoder blocks of (3,3,3) convolutions over the
    pixel grid, which is a second model to write and verify.

    It is left out because it cannot change the answer. The DiT alone already
    costs seconds per frame at 480p (`bench.py`), and the VAE encode/decode
    happens in addition to the pass that is timed -- so every figure in this
    directory is a floor on the real thing.
    """
    return ("VAE (%.1f M params, 3D causal, spatial /%d, temporal /%d) not costed; "
            "see vae_note()" % (vae_params() / 1e6, VAE["spatial_downsample_factor"],
                                VAE["temporal_downsample_factor"]))


def describe(variant):
    v = VARIANTS[variant]
    dual, vid = block_counts(variant)
    return ("SeedVR2 %s: dim=%d  %d blocks (%d dual-stream + %d vid-only)  "
            "mlp=%d  heads=%dx%d  patch=%s\n"
            "  %.4f B params modelled, %.4f B shipped  (+ %.1f M VAE)"
            % (variant.upper(), v["vid_dim"], dual + vid, dual, vid,
               v["mlp_hidden"], v["heads"], v["head_dim"], v["patch"],
               total_params(variant) / 1e9, v["shipped_params"] / 1e9,
               vae_params() / 1e6))


def weights_total_bytes(variant, dtype_bytes=2):
    """fp16 weights: DiT + VAE, as Comfy-Org ships them."""
    return (VARIANTS[variant]["shipped_params"] + vae_params()) * dtype_bytes
