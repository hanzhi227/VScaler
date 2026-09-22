# Third-party notices

VScaler is MIT licensed; see [`LICENSE`](LICENSE). This file records the model
weights embedded in the source and the external tools used by the app.

## Embedded in this repository

### Anime4K v3.2 (MIT)

`engine/Anime4KMetal` is a Metal port of bloc97's Anime4K v3.2
`Upscale_CNN_x2_M` network (plus its denoise variant). The weights embedded in
`Sources/Anime4KMetal/CNNWeights.swift` are extracted from upstream's
`Anime4K_Upscale_CNN_x2_M.glsl` by `Scripts/extract_weights.py`.

- Upstream: <https://github.com/bloc97/Anime4K>
- License: MIT License — Copyright (c) 2019 bloc97

> MIT License
>
> Copyright (c) 2019 bloc97
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

### Real-ESRGAN checkpoints (BSD 3-Clause)

The two live modes `--mode realesrgan` and `--mode realesrgan-general` run
Real-ESRGAN's released `realesr-animevideov3` and `realesr-general-x4v3`
checkpoints (v0.2.5.0), exported to fp16 blobs in
`engine/RealESRGANMetal/Sources/RealESRGANMetal/RealESRGANWeights.swift` and
`RealESRGANGeneralWeights.swift` by `scripts/realesrgan/export_weights.py`.
The Metal re-implementation of `SRVGGNetCompact` in that package is this
project's own code, verified against upstream's PyTorch output.

- Upstream: <https://github.com/xinntao/Real-ESRGAN>
- License: BSD 3-Clause — Copyright 2021 Xintao Wang

> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice,
>    this list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice,
>    this list of conditions and the following disclaimer in the documentation
>    and/or other materials provided with the distribution.
> 3. Neither the name of the copyright holder nor the names of its contributors
>    may be used to endorse or promote products derived from this software
>    without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
> ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
> LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
> CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
> SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
> INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
> CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
> ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
> POSSIBILITY OF SUCH DAMAGE.

## External dependencies (invoked, not distributed)

The app calls these tools as separate processes or uses macOS system frameworks.
The app bundle does not include ffmpeg or VLC.

- **ffmpeg / ffprobe** — demuxing, decoding, audio, and the stream path's
  encode. Expected at `/opt/homebrew/bin/ffmpeg` (Homebrew). The Homebrew
  formula is GPL-licensed; you install your own copy.
  <https://ffmpeg.org>
- **VLC** — only the optional stream path (`macvideoupscale --open-vlc`) launches
  it, and only if it is already installed. <https://www.videolan.org>
- **MetalFX, Metal, AVFoundation, CoreText** — Apple system frameworks, used
  through the macOS 14+ SDK.

## Test material (fetched or generated locally, not committed)

The clips under `test-media/` are deliberately outside version control; their
provenance and rebuild commands live in `test-media/liveaction/README.md`,
`test-media/yt/README.md`, and `scripts/gen_synth_anime.py`.

- *Table tennis* — Wikimedia Commons, by Evworo,
  [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
- *Big Buck Bunny* and *Sintel* excerpts — © Blender Foundation,
  CC BY ([bbb](https://peach.blender.org), [sintel](https://durian.blender.org)).
- The synthesized anime-style clips are generated from scratch by
  `scripts/gen_synth_anime.py` and carry no third-party content.
- TV episodes and downloaded promotional footage used during development are
  third-party copyrighted material, are **not** part of this repository, and
  must never be committed or redistributed with it.

## Evaluation-only upstreams (referenced by `scripts/`, not part of the app)

These models and frameworks were benchmarked or considered during design and
are documented under `scripts/` and `docs/`. None of their code or weights is
included in this repository or in the app.

- [RT4KSR](https://github.com/eduardzamfir/RT4KSR) — NTIRE 2023 Real-Time 4K
  Super-Resolution, measured and not adopted (`scripts/rt4ksr/`).
- [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) — diffusion video
  restoration, measured in fp16 and 4-bit MLX and not adopted
  (`scripts/seedvr2/`).
- [Video2X](https://github.com/k4yt3x/video2x) (AGPL-3.0) — upscaling
  framework, evaluated as a vehicle and not used (`scripts/video2x/`).
- [PiperSR](https://github.com/ModelPiper/PiperSR) — checked and rejected
  (`scripts/rt4ksr/README.md`).
- [AnimeJaNai](https://github.com/the-database/AnimeJaNai) — weights
  benchmarked against the shipped checkpoints (`scripts/bench_animejanai.py`);
  the weights themselves are not distributed here.
