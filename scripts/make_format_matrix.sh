#!/usr/bin/env bash
# Builds one small file per container/codec combination, from a single source
# clip, so "does it support format X?" is answered by playing X rather than by
# reading a list of extensions. Used by the format-matrix check in the README.
#
#   scripts/make_format_matrix.sh [outdir] [source]
#
# Defaults: outdir /tmp/format-matrix, source test-media/segments/naruto_t880.mkv
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/tmp/format-matrix}"
SRC="${2:-$ROOT/test-media/segments/naruto_t880.mkv}"
DUR="${DURATION:-12}"

[ -f "$SRC" ] || { echo "no source clip: $SRC" >&2; exit 1; }
mkdir -p "$OUT"
cd "$OUT"

# -t keeps every file short; -c:a copy is not used because the point is to
# exercise each audio decoder too.
V="-t $DUR"

echo "== container/codec matrix from $(basename "$SRC") =="
# name                     ffmpeg arguments (video codec / audio codec / container)
enc() {
    local name="$1"; shift
    if ffmpeg -v error -y $V -i "$SRC" "$@" "$name" 2>/tmp/matrix-err; then
        printf '  built  %-22s %s\n' "$name" "$(ffprobe -v error -show_entries \
            stream=codec_name -of csv=p=0 "$name" 2>/dev/null | paste -sd+ -)"
    else
        printf '  SKIP   %-22s (%s)\n' "$name" "$(head -1 /tmp/matrix-err | cut -c1-70)"
    fi
}

# --- the common case: MP4 and friends
enc h264.mp4          -c:v libx264 -preset ultrafast -crf 28 -c:a aac
enc hevc.mp4          -c:v libx265 -preset ultrafast -crf 30 -c:a aac -tag:v hvc1
enc h264.m4v          -c:v libx264 -preset ultrafast -crf 28 -c:a aac
enc h264.mov          -c:v libx264 -preset ultrafast -crf 28 -c:a aac
enc h264_noaudio.mp4  -c:v libx264 -preset ultrafast -crf 28 -an
enc h264_51.mp4       -c:v libx264 -preset ultrafast -crf 28 -c:a aac -ac 6
# MP4 with a text subtitle track (mov_text) — a real-world case for anime files.
# The second input has to come before the codec options, or they are applied to
# the subtitle file as input options and ffmpeg rejects the command.
enc h264_subs.mp4     -i "$ROOT/test-media/subs_test.srt" -map 0:v:0 -map 0:a:0 -map 1 \
                      -c:v libx264 -preset ultrafast -crf 28 -c:a aac -c:s mov_text

# --- the formats macOS does not ship a UTI for
enc h264.mkv          -c:v libx264 -preset ultrafast -crf 28 -c:a aac
enc vp9_opus.webm     -c:v libvpx-vp9 -b:v 400k -c:a libopus
enc vp8_vorbis.webm   -c:v libvpx -b:v 400k -c:a vorbis -strict -2
enc h264.ts           -c:v libx264 -preset ultrafast -crf 28 -c:a aac -f mpegts
enc h264.m2ts         -c:v libx264 -preset ultrafast -crf 28 -c:a aac -f mpegts
enc h264.mts          -c:v libx264 -preset ultrafast -crf 28 -c:a aac -f mpegts
enc h264.flv          -c:v libx264 -preset ultrafast -crf 28 -c:a libmp3lame -f flv

# --- older / software-codec containers
enc mpeg4.avi         -c:v mpeg4 -q:v 6 -c:a libmp3lame
enc mpeg2.mpg         -c:v mpeg2video -q:v 6 -c:a mp2
enc mpeg1.mpg         -c:v mpeg1video -q:v 6 -c:a mp2
enc mpeg2.vob         -c:v mpeg2video -q:v 6 -c:a mp2 -f vob
enc wmv2.wmv          -c:v wmv2 -q:v 6 -c:a wmav2
enc h264.3gp          -c:v libx264 -preset ultrafast -crf 28 -c:a aac -f 3gp
enc mjpeg.mov         -c:v mjpeg -q:v 6 -c:a pcm_s16le
# Formats that exist mainly to prove the declared document types work: a .divx
# extension over AVI, RealVideo, ASF and MXF.
enc divx.divx         -c:v mpeg4 -q:v 6 -c:a libmp3lame -f avi
enc rv20.rm           -c:v rv20 -q:v 6 -c:a ac3 -f rm
enc wmv2.asf          -c:v wmv2 -q:v 6 -c:a wmav2 -f asf
enc mpeg2.mxf         -c:v mpeg2video -q:v 6 -c:a pcm_s16le -f mxf
# Not built: .ogv. This ffmpeg has no Theora encoder (only libtheora would do;
# the Ogg muxer takes nothing else for video), so the type is declared for the
# Finder but has no sample here. RealVideo and MXF used to be in the same
# position until rv10/rv20 and the mxf muxer turned out to be present.

# --- metadata the player has to cope with
# 10-bit sources: Hi10P H.264 and 10-bit HEVC are common for anime, and the
# decoder has to convert them down to 8-bit BGRA for the scaler.
enc h264_10bit.mkv    -c:v libx264 -preset ultrafast -crf 28 -pix_fmt yuv420p10le -c:a aac
enc hevc_10bit.mp4    -c:v libx265 -preset ultrafast -crf 30 -pix_fmt yuv420p10le -c:a aac -tag:v hvc1
# Odd dimensions: a source that is not a multiple of 2 exercises the scaler's
# 2x output sizing (1282x962 rather than a tidy 1280x960). 4:2:0 cannot represent
# an odd width, so this one is 4:4:4.
enc odd_size.mp4      -vf scale=641:481 -c:v libx264 -pix_fmt yuv444p \
                      -preset ultrafast -crf 28 -c:a aac

# Rotation: a phone-shot portrait clip carries a display matrix, not rotated
# pixels. `-display_rotation` as an *input* option with `-c copy` is the way to
# write one (the old `-metadata rotate=90` is silently dropped by modern ffmpeg),
# and `-map 0` is needed or ffmpeg's default stream selection drops subtitles.
for r in 90 180 270; do
    ffmpeg -v error -y $V -display_rotation $r -i "$SRC" -c copy "rotate$r.mp4" 2>/dev/null \
        && printf '  built  %-22s %s\n' "rotate$r.mp4" \
           "copy + display matrix $(ffprobe -v error -select_streams v -show_entries stream_side_data=rotation -of csv=p=0 "rotate$r.mp4")" \
        || printf '  SKIP   %-22s\n' "rotate$r.mp4"
done
# The awkward combination: rotation *and* subtitles, where bitmap and text
# subtitles take different paths through the filter graph.
ffmpeg -v error -y -display_rotation 90 -i "$ROOT/test-media/segments/subs_bitmap.mkv" \
       -map 0 -c copy rot_subs_bitmap.mkv 2>/dev/null \
    && printf '  built  %-22s %s\n' rot_subs_bitmap.mkv "rotated + dvd_subtitle" || true
ffmpeg -v error -y -display_rotation 90 -i "$ROOT/test-media/segments/subs_text.mkv" \
       -map 0 -c copy rot_subs_text.mkv 2>/dev/null \
    && printf '  built  %-22s %s\n' rot_subs_text.mkv "rotated + subrip" || true

# Cover art, the layout that broke: an attached picture that is video stream 0
# makes a naive "first video stream" mapping show the poster instead of the film.
ffmpeg -v error -y -ss 3 -i "$SRC" -frames:v 1 /tmp/matrix-cover.jpg 2>/dev/null
ffmpeg -v error -y $V -i "$SRC" -i /tmp/matrix-cover.jpg -map 0 -map 1 \
       -c copy -disposition:v:1 attached_pic cover.mp4 2>/dev/null \
    && printf '  built  %-22s %s\n' cover.mp4 "film first, poster attached" \
    || printf '  SKIP   cover.mp4\n'
# Poster *first*, and flagged, so stream 0 really is the cover.
ffmpeg -v error -y $V -i /tmp/matrix-cover.jpg -i "$SRC" -map 0 -map 1 -c copy \
       -disposition:v:0 attached_pic cover_first.mp4 2>/dev/null \
    && printf '  built  %-22s %s\n' cover_first.mp4 "poster is video stream 0" \
    || printf '  SKIP   cover_first.mp4\n'
# Poster first but *unflagged* (some muxers omit the disposition) — the case that
# needs the longest-stream rule rather than the flag alone.
ffmpeg -v error -y $V -i /tmp/matrix-cover.jpg -i "$SRC" -map 0 -map 1 -c copy \
       cover_unflagged.mp4 2>/dev/null \
    && printf '  built  %-22s %s\n' cover_unflagged.mp4 "poster first, no disposition" \
    || printf '  SKIP   cover_unflagged.mp4\n'
# Not a video at all: audio with a cover image. The player has to say so rather
# than "play" the artwork. Codec options go after every input, before the output.
ffmpeg -v error -y $V -i "$SRC" -i /tmp/matrix-cover.jpg \
       -map 0:a:0 -map 1 -c:a aac -c:v copy -disposition:v:0 attached_pic \
       audio_with_cover.m4a 2>/dev/null \
    && printf '  built  %-22s %s\n' audio_with_cover.m4a "audio + cover, no video" \
    || printf '  SKIP   %-22s\n' audio_with_cover.m4a

echo "== files in $OUT =="
ls -1 "$OUT" | sed 's/^/  /'
