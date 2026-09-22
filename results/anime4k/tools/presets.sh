#!/bin/bash
# Shared preset -> shader file list definitions.
# Sourced by other tools. Provides function preset_shaders <preset> <sep>.
#
# Shader sources:
#   Anime4K v3.2 release zip  : single-pass CNN upscalers (byte-identical to v4.0.1)
#   Anime4K v4.0.1 release zip: used for the *official* modeA/B/C chains, because the
#                               v3.2 zip ships neither a README/mpv.conf nor the
#                               Restore_CNN / Clamp_Highlights / AutoDownscalePre_x2
#                               shaders that the official mode definitions reference.
# Chains are copied verbatim from the official v4.0.1 macOS/Linux Low-end template
# (GLSL_Mac_Linux_Low-end.zip -> input.conf, CTRL+1/2/3):
#   Mode A: Clamp_Highlights, Restore_CNN_M,      Upscale_CNN_x2_M, AutoDownscalePre_x2, AutoDownscalePre_x4, Upscale_CNN_x2_S
#   Mode B: Clamp_Highlights, Restore_CNN_Soft_M, Upscale_CNN_x2_M, AutoDownscalePre_x2, AutoDownscalePre_x4, Upscale_CNN_x2_S
#   Mode C: Clamp_Highlights, Upscale_Denoise_CNN_x2_M,              AutoDownscalePre_x2, AutoDownscalePre_x4, Upscale_CNN_x2_S

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
V32="$ROOT/third_party/Anime4K/Anime4K_v3.2"
V4="$ROOT/third_party/Anime4K/Anime4K_v4.0.1"

preset_shaders() {
    local p="$1" sep="${2:-:}"
    case "$p" in
        off)         echo "" ;;
        cnnM)        echo "$V32/Anime4K_Upscale_CNN_x2_M.glsl" ;;
        cnnDenoiseM) echo "$V32/Anime4K_Upscale_Denoise_CNN_x2_M.glsl" ;;
        modeA)       echo "$V4/Anime4K_Clamp_Highlights.glsl$sep$V4/Anime4K_Restore_CNN_M.glsl$sep$V4/Anime4K_Upscale_CNN_x2_M.glsl$sep$V4/Anime4K_AutoDownscalePre_x2.glsl$sep$V4/Anime4K_AutoDownscalePre_x4.glsl$sep$V4/Anime4K_Upscale_CNN_x2_S.glsl" ;;
        modeB)       echo "$V4/Anime4K_Clamp_Highlights.glsl$sep$V4/Anime4K_Restore_CNN_Soft_M.glsl$sep$V4/Anime4K_Upscale_CNN_x2_M.glsl$sep$V4/Anime4K_AutoDownscalePre_x2.glsl$sep$V4/Anime4K_AutoDownscalePre_x4.glsl$sep$V4/Anime4K_Upscale_CNN_x2_S.glsl" ;;
        modeC)       echo "$V4/Anime4K_Clamp_Highlights.glsl$sep$V4/Anime4K_Upscale_Denoise_CNN_x2_M.glsl$sep$V4/Anime4K_AutoDownscalePre_x2.glsl$sep$V4/Anime4K_AutoDownscalePre_x4.glsl$sep$V4/Anime4K_Upscale_CNN_x2_S.glsl" ;;
        *) echo "UNKNOWN_PRESET:$p" >&2; return 1 ;;
    esac
}

# Human-readable preset -> shader file name list (used for the report).
preset_filerefs() {
    local p="$1"
    preset_shaders "$p" "|" | tr '|' '\n' | sed 's|.*/||'
}

ALL_PRESETS="off cnnM cnnDenoiseM modeA modeB modeC"
