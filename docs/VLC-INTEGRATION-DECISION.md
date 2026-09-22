# Decision: integrate into VLC natively, or keep the external stream pipeline?

> **Update — the question was settled a different way.** Rather than integrate
> into VLC (options B/C/D) or keep feeding it a live stream (option A), the app
> now plays the upscaled frames **itself**: `PlayerController` renders to an
> `AVSampleBufferDisplayLayer` and schedules audio on the same
> `AVSampleBufferRenderSynchronizer`, with a 2–3 frame buffer instead of the
> stream path's 30 s ring. The motivation was playback ergonomics, not the
> upscaler: a live MPEG-TS stream is not seekable, so VLC's scrubber could not
> work, every run meant launching a separate owned VLC instance, and the
> read-ahead had to stay large to keep that pull-based player fed. Owning the
> render clock removes all three constraints at once — see `README.md`.
>
> The analysis below is kept because it is what ruled out the VLC plugin routes,
> and because the **stream pipeline is still shipped** (the `macvideoupscale` CLI)
> for anyone who wants to hand the picture to an external player. Item 6 of the
> brief ("decide on native VLC integration") is therefore answered: no native
> integration, because it would have forced per-frame GPU↔CPU round trips
> (option B) or an unreleased, ABI-unstable VLC (option D), and it would not
> have fixed the seek/launch complaints anyway.

**Item 6 of the project brief.** Evidence base: `docs/vlc-integration-research.md`
(web/source research on VLC 3.x and 4.x filter APIs, with citations), plus the
working prototype in `engine/`.

## The options

| option | effort | integration quality | risk / maintenance |
|---|---|---|---|
| **A. Keep the external pipeline (current prototype)** | done | good: VLC plays a local stream; controls live in our app; ~1–2 s extra latency; one encode step | low; no VLC internals, survives VLC updates |
| **B. Drop-in VLC 3 plugin** (`VLC_PLUGIN_PATH`, external plugin dir) | days–weeks | best on paper: native playback surface, no encode, `--video-filter` | VLC 3 filters take **CPU `picture_t`** — a Metal upscaler needs upload/download per frame (≈16 MB/frame at 1080p RGBA). ABI is internal/unstable; empirically unverified |
| **C. Rebuild VLC with the filter compiled in** | weeks | as B, no ABI risk | forking/rebuilding VLC + keeping up with upstream |
| **D. VLC 4 nightly `--pl-user-shader`** (libplacebo vout, user GLSL) | days | potentially the clean native path with **GPU→GPU** processing | VLC 4 is still unreleased (3.0.23 is current); needs porting Anime4K from mpv-GLSL to libplacebo's shader format; tested only on nightlies |
| **E. Switch player to mpv/IINA** (not requested, for completeness) | hours | Anime4K runs natively via `--glsl-shaders` today, no encode, no latency | different player than the brief asks for |

## Verified facts behind the table

- VLC 3's video-filter API is CPU-`picture_t` based; there is no GPU filter hook.
  Its libplacebo use is internal (HDR tone-mapping), not exposed to user shaders.
- The macOS release binary does carry the `disable-library-validation`
  entitlement and scans `VLC_PLUGIN_PATH`, so an *external* drop-in plugin is
  plausible (module ABI `vlc_entry__3_0_0f`), but this was **not** empirically
  loaded — treat as unverified.
- VLC 4 master has an OpenGL filter chain and a libplacebo vout with
  `--pl-user-shader`; macOS VLC output is still OpenGL (no Metal renderer) in
  both 3.x and 4.x.

## Recommendation

**Keep the external pipeline (option A) for now, and re-test option D when
VLC 4 ships.** Rationale:

1. The prototype already meets the acceptance target end-to-end, and its cost is
   measurable and acceptable: one hardware H.264 encode (the encoder runs
   2.4× faster than realtime at 4K on this machine) plus ~1–2 s of added
   latency. VLC remains untouched and therefore stable.
2. Option B is the only way to get *true* native integration on today's stable
   VLC, but it inverts our main advantage: Anime4K is fast **because** it runs on
   the GPU, and a VLC 3 CPU filter would force a per-frame GPU↔CPU round trip
   whose bandwidth alone (4K RGBA ≈ 33 MB/frame) is hostile. It would likely be
   *slower* than the current pipeline, not faster.
3. Option D is the right long-term native answer (GPU→GPU, no encode, no extra
   latency) but depends on an unreleased VLC and on porting the shader chain to
   libplacebo's format. Revisit when VLC 4 is stable; the Metal port we built is
   already numerically validated against the GLSL, which de-risks that port.
4. Option E (mpv/IINA) is the pragmatic escape hatch if native integration is
   wanted today: mpv runs Anime4K's GLSL directly, and our Metal pipeline could
   be exposed to IINA/mpv via a filter or a small patch. It was out of scope here
   because the brief specifies VLC.

**Trigger to revisit:** if VLC 4 becomes the default install, or if the ~1–2 s
latency / extra encode becomes the dominant complaint.
