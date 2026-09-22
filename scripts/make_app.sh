#!/usr/bin/env bash
#
# Builds "VScaler.app" — a real double-clickable bundle around the
# SwiftPM executable — and installs it, so it launches from Spotlight, the
# Finder or the Dock instead of a terminal.
#
#   ./scripts/make_app.sh                 # build, sign, install to /Applications
#   ./scripts/make_app.sh --install DIR   # ...to DIR instead
#   ./scripts/make_app.sh --out DIR       # just assemble DIR/VScaler.app
#   ./scripts/make_app.sh --run           # build/install, then launch it
#
# The bundle is self-contained: the Metal kernels are compiled from source
# strings inside the binary and ffmpeg/ffprobe are referenced by absolute path
# (/opt/homebrew/bin), so the executable and Info.plist are all it needs.
#
# It is assembled in a temporary directory and moved into place, so a normal run
# never leaves a second copy of the app lying around. Two bundles with the same
# identifier confuse LaunchServices (Spotlight shows both, "Open With" may pick
# either), so there is deliberately only ever one.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG="$ROOT/engine/MacVideoUpscaler"
BUNDLE_ID="dev.hanzh.macvideoupscaler"
APP_NAME="VScaler"
EXECUTABLE="MacVideoUpscalerApp"
DEST=""                           # empty = install to /Applications
OUT=""
RUN=0
LSREGISTER=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install) DEST="${2:-/Applications}"; [[ $# -gt 1 ]] && shift ;;
        --out) OUT="${2:?--out needs a directory}"; shift ;;
        --run) RUN=1 ;;
        -h|--help) awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

if [[ -n "$OUT" ]]; then
    DEST="$OUT"
elif [[ -z "$DEST" ]]; then
    DEST="/Applications"
fi
TARGET="$DEST/$APP_NAME.app"
STAGE="$(mktemp -d)/$APP_NAME.app"
trap 'rm -rf "$(dirname "$STAGE")"' EXIT

# Only ever remove a bundle that is ours: a same-named app from elsewhere is not
# something this script should delete.
remove_if_ours() {
    local path="$1"
    [[ -e "$path" ]] || return 0
    local found
    found="$(defaults read "$path/Contents/Info" CFBundleIdentifier 2>/dev/null || echo "")"
    if [[ "$found" != "$BUNDLE_ID" ]]; then
        echo "refusing to replace $path (it is '$found', not ours)" >&2
        exit 1
    fi
    rm -rf "$path"
}

echo "==> building (release)"
swift build -c release --package-path "$PKG"

echo "==> assembling $APP_NAME.app"
mkdir -p "$STAGE/Contents/MacOS" "$STAGE/Contents/Resources"
cp "$PKG/.build/release/$EXECUTABLE" "$STAGE/Contents/MacOS/$EXECUTABLE"
cp "$PKG/AppBundle/Info.plist" "$STAGE/Contents/Info.plist"
cp "$ROOT/LICENSE" "$ROOT/THIRD-PARTY-NOTICES.md" "$STAGE/Contents/Resources/"
printf 'APPL????' > "$STAGE/Contents/PkgInfo"

# --- icon: scale the checked-in artwork into the macOS iconset
ICON_SRC="$PKG/AppBundle/AppIcon-1024.png"
CACHE="$PKG/.build/app-bundle"
ICNS="$CACHE/AppIcon.icns"
if [[ ! -f "$ICNS" || "$ICON_SRC" -nt "$ICNS" ]]; then
    echo "==> building icon"
    mkdir -p "$CACHE"
    ICONSET="$CACHE/AppIcon.iconset"
    rm -rf "$ICONSET"; mkdir -p "$ICONSET"
    for s in 16 32 128 256 512; do
        sips -z "$s" "$s" "$ICON_SRC" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
        sips -z "$((s * 2))" "$((s * 2))" "$ICON_SRC" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
    done
    iconutil -c icns "$ICONSET" -o "$ICNS"
    rm -rf "$ICONSET"
fi
cp "$ICNS" "$STAGE/Contents/Resources/AppIcon.icns"

# Ad-hoc signature: enough for local use (Gatekeeper only gates quarantined
# downloads) and it keeps the bundle's identity stable across rebuilds, so
# folder-permission grants and Dock position survive.
echo "==> signing (ad-hoc)"
codesign --force --sign - --identifier "$BUNDLE_ID" "$STAGE" >/dev/null
codesign --verify --strict "$STAGE"
echo "    signature OK"

remove_if_ours "$DEST/MacVideoUpscaler.app"
remove_if_ours "$DEST/FrameRise.app"
remove_if_ours "$TARGET"
mkdir -p "$DEST"
mv "$STAGE" "$TARGET"
# Register this copy and drop any stale record of a bundle that used to live
# elsewhere, so the app the system opens is the one that was just built.
"$LSREGISTER" -f "$TARGET" >/dev/null 2>&1 || true

if [[ $RUN -eq 1 ]]; then
    echo "==> launching"
    open "$TARGET"
fi

echo
if [[ -n "$OUT" ]]; then
    echo "assembled: $TARGET   (not installed)"
else
    echo "installed: $TARGET"
fi
echo "  launch:      open \"$TARGET\"      (or Spotlight: VScaler)"
echo "  open a file: open -a \"$TARGET\" /path/to/episode.mkv   (or drag it onto the icon)"
