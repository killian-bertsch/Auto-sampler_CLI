#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# convert_samples.sh
#
# Converts a folder of numbered FLAC sample layers into per-instrument folders.
#
# INPUT naming convention (files must be in <input_dir>):
#   {N} {InstrumentName}.flac    (N = 1, 2, 3 …)   — sustain layers
#   R {InstrumentName}.flac                          — release  (single or double space ok)
#
# OUTPUT structure:
#   <output_dir>/
#     {InstrumentName}/
#       sustain.flac   ← layers concatenated in order, peak-normalised
#       release.flac   ← peak-normalised
#
# Instruments listed in STEREO_INSTRUMENTS are kept stereo.
# All others are downmixed to mono.
# Both passes (analysis + encode) use identical filter chains so the
# measured peak and applied gain are always consistent.
#
# Usage:
#   ./convert_samples.sh <input_dir> [output_dir]
#
#   output_dir defaults to input_dir (instrument folders are created inside it).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_PEAK_DB=-6.0              # dBFS peak target for all outputs
STEREO_INSTRUMENTS=("THEGRANDEUR")   # Keep these stereo; all others → mono
NUM_LAYERS=(1 2 3)               # Sustain layer numbers, in order
# ─────────────────────────────────────────────────────────────────────────────

INPUT_DIR="${1:?Usage: $0 <input_dir> [output_dir]}"
OUTPUT_DIR="${2:-$INPUT_DIR}"
INPUT_DIR="$(realpath "$INPUT_DIR")"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Helpers ───────────────────────────────────────────────────────────────────

is_stereo() {
    local name="$1" s
    (( ${#STEREO_INSTRUMENTS[@]} == 0 )) && return 1
    for s in "${STEREO_INSTRUMENTS[@]}"; do
        [[ "$name" == "$s" ]] && return 0
    done
    return 1
}

# get_peak <"file"|"concat"> <path> <af_chain>
# Runs a null-output pass and returns the max peak in dBFS.
# af_chain is passed directly to -af (e.g. "aformat=channel_layouts=mono,volumedetect")
get_peak() {
    local mode="$1" src="$2" af_chain="$3"
    local out
    if [[ "$mode" == "concat" ]]; then
        out=$(ffmpeg -f concat -safe 0 -i "$src" -af "$af_chain" -f null /dev/null 2>&1)
    else
        out=$(ffmpeg -i "$src" -af "$af_chain" -f null /dev/null 2>&1)
    fi
    echo "$out" | awk '/max_volume:/{print $5}'
}

# calc_gain <peak_dBFS>  →  prints gain in dB needed to reach TARGET_PEAK_DB
calc_gain() {
    python3 -c "peak=float('$1'); print(f'{$TARGET_PEAK_DB - peak:.4f}')"
}

# ── Discover instruments ───────────────────────────────────────────────────────
INSTRUMENTS=()
while IFS= read -r f; do
    INSTRUMENTS+=("$(basename "$f" | sed 's/^1 //;s/\.flac$//')")
done < <(find "$INPUT_DIR" -maxdepth 1 -name "1 *.flac" | sort)

if (( ${#INSTRUMENTS[@]} == 0 )); then
    log "ERROR: No instruments found — expected files named '1 <InstrumentName>.flac' in:"
    log "       $INPUT_DIR"
    exit 1
fi

log "Input dir  : $INPUT_DIR"
log "Output dir : $OUTPUT_DIR"
log "Target peak: ${TARGET_PEAK_DB} dBFS"
log "Instruments: ${INSTRUMENTS[*]}"
log ""

# ── Main loop ─────────────────────────────────────────────────────────────────
for INSTRUMENT in "${INSTRUMENTS[@]}"; do
    log "══════════════════════════════════════════"
    log "  Instrument : $INSTRUMENT"
    mkdir -p "$OUTPUT_DIR/$INSTRUMENT"

    # Build the audio filter prefix (channel conversion) for this instrument.
    # Both the analysis pass and the encode pass use the same filter chain so
    # the measured peak exactly matches what will be encoded.
    if is_stereo "$INSTRUMENT"; then
        CHAN_FILTER=""      # no channel conversion; stays stereo
        log "  Channels   : stereo"
    else
        CHAN_FILTER="aformat=channel_layouts=mono"
        log "  Channels   : mono"
    fi

    # Helper: build full filter chain for analysis or encoding
    # analysis:  <chan_filter>,volumedetect   (or just volumedetect if stereo)
    # encoding:  <chan_filter>,volume=<gain>dB
    make_af() {
        local suffix="$1"
        if [[ -n "$CHAN_FILTER" ]]; then
            echo "${CHAN_FILTER},${suffix}"
        else
            echo "$suffix"
        fi
    }

    # ── Sustain ───────────────────────────────────────────────────────────────
    CONCAT_FILE="/tmp/concat_${INSTRUMENT// /_}.txt"
    > "$CONCAT_FILE"
    LAYER_COUNT=0
    for NUM in "${NUM_LAYERS[@]}"; do
        SRC="$INPUT_DIR/$NUM $INSTRUMENT.flac"
        if [[ -f "$SRC" ]]; then
            echo "file '$SRC'" >> "$CONCAT_FILE"
            (( LAYER_COUNT++ ))
        else
            log "  WARNING: layer $NUM missing — $SRC"
        fi
    done

    if (( LAYER_COUNT == 0 )); then
        log "  ERROR: no sustain layers found for $INSTRUMENT — skipping"
        continue
    fi

    SUSTAIN_OUT="$OUTPUT_DIR/$INSTRUMENT/sustain.flac"

    log "  [sustain] Pass 1/2 — analysing peak ($LAYER_COUNT layers)..."
    ANALYSIS_AF=$(make_af "volumedetect")
    PEAK=$(get_peak concat "$CONCAT_FILE" "$ANALYSIS_AF")
    GAIN=$(calc_gain "$PEAK")
    log "  [sustain] Peak: ${PEAK} dB  |  Gain: ${GAIN} dB"

    log "  [sustain] Pass 2/2 — encoding..."
    ENCODE_AF=$(make_af "volume=${GAIN}dB")
    ffmpeg -y -f concat -safe 0 -i "$CONCAT_FILE" \
        -af "$ENCODE_AF" \
        -c:a flac "$SUSTAIN_OUT"
    log "  [sustain] ✓  $(du -sh "$SUSTAIN_OUT" | cut -f1)"

    # ── Release ───────────────────────────────────────────────────────────────
    # Accept either single or double space between "R" and the instrument name
    RELEASE_SRC=""
    for candidate in \
        "$INPUT_DIR/R $INSTRUMENT.flac" \
        "$INPUT_DIR/R  $INSTRUMENT.flac"
    do
        [[ -f "$candidate" ]] && { RELEASE_SRC="$candidate"; break; }
    done

    if [[ -z "$RELEASE_SRC" ]]; then
        log "  WARNING: no release file found for $INSTRUMENT — skipping release"
        continue
    fi

    RELEASE_OUT="$OUTPUT_DIR/$INSTRUMENT/release.flac"

    log "  [release] Pass 1/2 — analysing peak..."
    PEAK=$(get_peak file "$RELEASE_SRC" "$ANALYSIS_AF")
    GAIN=$(calc_gain "$PEAK")
    log "  [release] Peak: ${PEAK} dB  |  Gain: ${GAIN} dB"

    log "  [release] Pass 2/2 — encoding..."
    ENCODE_AF=$(make_af "volume=${GAIN}dB")
    ffmpeg -y -i "$RELEASE_SRC" \
        -af "$ENCODE_AF" \
        -c:a flac "$RELEASE_OUT"
    log "  [release] ✓  $(du -sh "$RELEASE_OUT" | cut -f1)"

done

log ""
log "══════════════════════════════════════════"
log "Done. Output: $OUTPUT_DIR"
