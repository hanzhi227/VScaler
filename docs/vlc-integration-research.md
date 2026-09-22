# Integrating a custom Metal real-time upscaler into VLC on macOS

Scope: can an Anime4K-style Metal GPU filter run natively inside VLC, vs. keeping the existing
external Metal+stream pipeline. Verified against VLC 3.0.23 (installed app) and VLC 4 `master`.
Cited items are confirmed; anything marked **Inference** is reasoning, not documentation.

## 0. TL;DR

- VLC 3.x filters are CPU `picture_t` filters. A Metal filter is possible but requires a full
  per-frame CPU→GPU upload and GPU→CPU download; no zero-copy.
- A drop-in `.dylib` for the **release** VLC.app is plausible: the app ships
  `com.apple.security.cs.disable-library-validation`, scans `VLC_PLUGIN_PATH`, and plugins are
  `vlc_entry__3_0_0f` dylibs linked to `@rpath/libvlccore.dylib`. There is no official third-party
  plugin API, so it is unsupported territory.
- VLC 4 (master, unreleased as of 2026) has a GPU filter pipeline (`"opengl filter"` modules,
  GPU→GPU) and a libplacebo vout with `--pl-user-shader`; macOS output is **still OpenGL**, not Metal.
- mpv / IINA remain the only proven low-effort path for Anime4K-style shaders on macOS.

## 1. VLC 3.x (Vetinari, 3.0.23)

### 1.1 Video filter API
- A filter is `filter_t` with `picture_t *(*pf_video_filter)(filter_t *, picture_t *)`
  ([include/vlc_filter.h](https://code.videolan.org/videolan/vlc/-/blob/3.0.x/include/vlc_filter.h));
  output comes from `filter_NewPicture()` (CPU-visible, chroma-typed planes). The vout runs them via
  `filter_chain_VideoFilter()` in
  [src/video_output/video_output.c](https://code.videolan.org/videolan/vlc/-/blob/3.0.x/src/video_output/video_output.c).
- Modules register `set_capability("video filter", …)` / `"video converter"`
  ([scale.c](https://code.videolan.org/videolan/vlc/-/blob/3.0.x/modules/video_filter/scale.c)).
  All modules in
  [modules/video_filter/](https://code.videolan.org/videolan/vlc/-/tree/3.0.x/modules/video_filter)
  are CPU; no OpenGL/Metal filter exists in 3.0.x.
- **GPU inside a filter?** Structurally yes: open a `MTLDevice`, wrap input `picture_t` planes into
  `MTLTexture`s, dispatch compute, write a `filter_NewPicture()` output. But `picture_t` is host
  memory, so every frame pays upload+readback (~16 MB/frame at 1080p RGBA). Confirmed structurally;
  the cost is **Inference**. There is no hook to pass a decoder GPU surface, and no Metal↔CPU
  zero-copy bridge.
- 3.0.x macOS vout is OpenGL
  ([macosx.m](https://code.videolan.org/videolan/vlc/-/blob/3.0.x/modules/video_output/macosx.m)).
  libplacebo is used **internally only** for HDR tone-mapping in
  [vout_helper.c](https://code.videolan.org/videolan/vlc/-/blob/3.0.x/modules/video_output/opengl/vout_helper.c)
  (`#ifdef HAVE_LIBPLACEBO`; confirmed in the 3.0.23 tarball). No user-shader option, no Metal vout.

### 1.2 Can the release VLC.app load a third-party drop-in plugin?
Verified locally on `/Applications/VLC.app` (3.0.23, arm64 thin):
- Plugins are `Contents/MacOS/plugins/*.dylib` (+ `plugins.dat` cache). Each exports
  `_vlc_entry__3_0_0f` and links `@rpath/libvlccore.dylib` → `lib/libvlccore.9.dylib` (`nm`/`otool`).
- Entry-suffix ABI is hard-coded for the branch: `MODULE_SUFFIX "__3_0_0f"`
  ([vlc_plugin.h](https://code.videolan.org/videolan/vlc/-/blob/3.0.x/include/vlc_plugin.h)). A plugin
  must be built from 3.0.x headers, for arm64, exporting exactly that symbol.
- Search path: `<libdir>/plugins`, then every `VLC_PLUGIN_PATH` entry
  ([bank.c](https://code.videolan.org/videolan/vlc/-/blob/3.0.x/src/modules/bank.c),
  `AllocateAllPlugins`). A stale/absent per-dir `plugins.dat` triggers a rescan (logs
  `stale plugins cache: modified %s`); `--reset-plugins-cache` forces it.
- Gatekeeper: `codesign -d --verbose=4` shows hardened runtime, and entitlements include
  **`com.apple.security.cs.disable-library-validation = true`** — so dylibs not signed by VideoLAN
  can load. (Packaging uses `extras/package/macosx/codesign.sh`; the sandboxed MAS
  `VLC.entitlements` is a different build.)
- **Verdict: realistically viable**, with caveats: use `VLC_PLUGIN_PATH` pointing at an external
  folder instead of editing the signed bundle (bundle edits invalidate the seal; **Inference:** an
  already-approved/non-quarantined app usually still runs, but it is fragile); no stability promise
  for internal `libvlccore` interfaces, though `3_0_0f` is fixed within 3.0.x. I did **not**
  empirically build/load a test plugin, so treat this as high-confidence inference, not a test result.

### 1.3 Building VLC 3 from source with a filter compiled in
Per [wiki.videolan.org/MacOSCompile](https://wiki.videolan.org/MacOSCompile/): Xcode, tools from
`extras/tools`, contrib via `extras/package/macosx/build.sh -c`, `configure.sh`, `make -j4`;
`make VLC.app` + `codesign.sh`. Multi-hour first build; adding a filter needs edits to
`modules/video_filter/Makefile.am` + `configure.ac`. **Effort: high**, worthwhile only for a
self-contained app; a drop-in plugin is far cheaper.
## 2. VLC 4.x (master / nightlies)

- **Release status:** 4.0 is **not released** — `NEWS`
  ([link](https://code.videolan.org/videolan/vlc/-/blob/master/NEWS)) still says "Changes between
  3.0.x and 4.0.0-dev"; the 4.0.0 release page 404s; Wikipedia lists "4.0.x Otto Chriek" with no date
  while desktop stable is 3.0.23 (2026-01-08)
  ([Wikipedia](https://en.wikipedia.org/wiki/VLC_media_player)). Daily 4.0 macOS arm64 builds exist at
  [artifacts.videolan.org/vlc/nightly-macos-arm64](https://artifacts.videolan.org/vlc/nightly-macos-arm64/)
  (verified a 2026-02-26 build).
- **GPU filter APIs (confirmed):**
  - `"Opengl filter executor"` ([opengl.c](https://code.videolan.org/videolan/vlc/-/blob/master/modules/video_filter/opengl.c)),
    a `video-filter` module (shortcut `opengl`) taking `--opengl-filter` (list of `"opengl filter"`
    modules) + an offscreen GL provider. Its `Filter()` uses `vlc_gl_filters_*`, so filters operate on
    **GL textures, GPU→GPU, no CPU readback**.
  - Providers: [gladjust.c](https://code.videolan.org/videolan/vlc/-/blob/master/modules/video_filter/gladjust.c)
    (registers `video filter` + `opengl filter`), `dither.c`, `pl_scale.c` (libplacebo scaler).
  - Verified shipped in the macOS arm64 nightly: `libopengl_filter_plugin`, `libdither_plugin`,
    `libpl_scale_plugin`, `libplacebo_plugin` (vout), `libvout_macosx_plugin`.
  - **A custom GPU filter is still a compiled C module** registering `"opengl filter"` — no
    file-based GLSL hook in the filter chain — and it is OpenGL, not Metal.
- **libplacebo vout:** `--pl-user-shader <file>`, `--pl-upscaler-preset`, `--pl-debanding`,
  `--pl-dither` ([display.c](https://code.videolan.org/videolan/vlc/-/blob/master/modules/video_output/libplacebo/display.c)).
  NEWS adds "Vulkan/libplacebo video output plugin" and "libplacebo scaling and OpenGL filter pipeline
  support"; libplacebo 5.264.1 is bundled. User shaders use libplacebo's `PL_HOOK_*` format
  ([custom.h](https://code.videolan.org/videolan/libplacebo/-/blob/master/src/include/libplacebo/shaders/custom.h));
  whether Anime4K's mpv-format shaders run unmodified is **unverified** (mpv's hook syntax is related
  but not guaranteed identical → porting may be needed).
- **macOS vout: still OpenGL.** `macosx.m` is "Mac OS X OpenGL video output" (`NSOpenGLView`,
  shortcut `macosx`); `window_macosx.m` is only the window provider. NEWS adds an Apple
  `AVSampleBufferDisplayLayer` vout (`VLCVideoUIView`, mentions `CAMetalLayer`), but the render path
  stays OpenGL and Metal appears only as a `MetalToolchain` build requirement. No native Metal VLC
  renderer. **Confirmed.**
- **ABI changed:** VLC 4 plugins export plain `vlc_entry` + `vlc_entry_api_version`, with
  `VLC_API_VERSION_STRING "4.0.6"` (verified in the nightly SDK headers). VLC 3 plugins are **not**
  loadable, and the 4.x API is explicitly unstable.

## 3. Alternatives (brief)

- **mpv:** `--glsl-shader=<file>` / `--glsl-shaders=<list>` inject fragment shaders "at almost
  arbitrary points in the rendering pipeline, and access all previous intermediate textures"
  ([mpv manual](https://mpv.io/manual/stable/)). Anime4K documents mpv on macOS
  ([README](https://github.com/bloc97/Anime4K),
  [Mac/MPV guide](https://github.com/bloc97/Anime4K/blob/master/md/GLSL_Instructions_Mac_MPV.md)).
  macOS renderer: `vo=gpu`/`gpu-next` with `--gpu-context=cocoa-cb` (`macvk` = Vulkan/MoltenVK,
  experimental). Native seek/pause/fullscreen, no re-encode, lowest latency.
- **IINA** (mpv-based, native macOS, GPLv3, v1.4.4, macOS 12+/10.15+, [iina.io](https://iina.io/)):
  Anime4K ships an
  [IINA guide](https://github.com/bloc97/Anime4K/blob/master/md/GLSL_Instructions_Mac_IINA.md);
  shaders are added via IINA's custom mpv options. Best turnkey native-macOS option.
- **Pointing VLC's pipeline at custom shaders:** VLC 3 has no hook (libplacebo = internal HDR only).
  In VLC 4 the only file-based entry is `--pl-user-shader` on the **libplacebo vout** (not the macOS
  default), and the `--opengl-filter` chain needs compiled modules. libplacebo is bundled in both, but
  VLC 3 does not expose it as a filter/vout module.

## 4. Verdict table

| Option | Effort | Risk | Playback integration | Maintenance |
|---|---|---|---|---|
| mpv + Anime4K/shaders | Low | Low | Native fullscreen/pause/seek, no re-encode, lowest latency | Low (upstream shaders) |
| IINA + shaders (mpv-based) | Low | Low | Native macOS app, mpv-quality playback | Low |
| VLC 4 nightly `--pl-user-shader` (vout) | Low–Med | Med (unstable API, non-default macOS vout, shader-format port) | Native VLC playback; GPU→GPU; OpenGL backend | High (ABI churn) |
| VLC 4 compiled `"opengl filter"` C module | High | High (OpenGL not Metal; unstable ABI; nightly-only) | GPU→GPU in VLC; native playback | High |
| VLC 3 drop-in `.dylib` filter (CPU `picture_t`) | Med | Med–High (unsupported path; per-frame up/download) | Native VLC playback but extra GPU copies; no zero-copy | Med (rebuild per 3.0.x) |
| Fork & rebuild VLC 3 with filter compiled in | High | Med | Same, self-contained app | High (track 3.0.x) |
| Current external Metal+stream pipeline | Done | Low | External player/window; stream/re-encode latency | Low |

## 5. Recommendation for this hobby project

- Native VLC integration pays off **only if you specifically need VLC's playback surface** (format
  coverage, UI, embedding). For pure shader upscaling, mpv/IINA give better integration quality at a
  fraction of the effort — though neither accepts raw **Metal** compute, only GLSL.
- Realistic VLC routes, in order:
  1. **Cheapest experiment first:** install a VLC 4 macOS nightly and try
     `--pl-user-shader=<Anime4K shader>` (+ `--pl-upscaler-preset`, `--video-filter=pl_scale`). No
     build; validates whether libplacebo accepts your shaders. Compare against IINA.
  2. **If you need your own code in VLC:** a VLC 3 drop-in plugin (build against 3.0.x headers, export
     `vlc_entry__3_0_0f`, load via `VLC_PLUGIN_PATH`) doing Metal compute on each `picture_t`, and
     measure the upload/download cost before investing.
  3. Fork/rebuild only if you need one self-contained app — highest effort, least payoff.
- **Bottom line:** prefer IINA/mpv for the player experience; treat native VLC as a stretch goal and
  start with the no-build VLC 4 `--pl-user-shader` experiment, not VLC 3 plugin work.

## Source notes

Confirmed by inspection: installed VLC 3.0.23 (version, entitlements, plugin symbols, `plugins.dat`);
VLC 3.0.23 tarball (`filter_t`, `bank.c`, `vout_helper.c` libplacebo); VLC 4 macOS arm64 nightly
2026-02-26 SDK (shipped plugins, `vlc_entry`/`vlc_entry_api_version`, `VLC_API_VERSION_STRING
"4.0.6"`, `pl_scale`). Not verified: Anime4K mpv shaders loading unmodified in VLC 4 `--pl-user-shader`;
whether VLC 4's macOS libplacebo vout finds a working GPU provider (only `libplacebo_plugin` shipped,
no separate `placebo_gl`/`placebo_vk`); empirical drop-in load on the release app; VLC 4 release date.
