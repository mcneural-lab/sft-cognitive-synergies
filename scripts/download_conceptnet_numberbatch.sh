#!/usr/bin/env bash
# Download the English ConceptNet Numberbatch vectors used by Figure 2D.
#
# ConceptNet Numberbatch is by Luminoso Technologies, Inc. and is released
# under CC BY-SA 4.0. See LICENSE-DATA for the attribution this repository
# carries, and https://github.com/commonsense/conceptnet-numberbatch for the
# upstream terms.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=repro_common.sh
source "${SCRIPT_DIR}/repro_common.sh"

VERSION="19.08"
DEFAULT_URL="https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-${VERSION}.txt.gz"
DEFAULT_OUTPUT="data/embeddings/conceptnet"
DEFAULT_ARCHIVE="data/embeddings/numberbatch-en-${VERSION}.txt.gz"
EXPECTED_HEADER="516782 300"

URL="${CONCEPTNET_NUMBERBATCH_URL:-$DEFAULT_URL}"
OUTPUT="${CONCEPTNET_OUTPUT:-$DEFAULT_OUTPUT}"
ARCHIVE="${CONCEPTNET_ARCHIVE:-$DEFAULT_ARCHIVE}"
FORCE=0
KEEP_ARCHIVE=0

usage() {
  cat <<EOF
Usage:
  scripts/download_conceptnet_numberbatch.sh [options]

Downloads ConceptNet Numberbatch ${VERSION} English-only vectors and
decompresses them to:

  ${DEFAULT_OUTPUT}

Options:
  --force             Overwrite an existing output file and restart archive download.
  --keep-archive      Keep the downloaded .txt.gz archive after decompression.
  --url URL           Override the download URL.
  --output PATH       Override the decompressed output path.
  --archive PATH      Override the temporary archive path.
  -h, --help          Show this help.

Environment overrides:
  CONCEPTNET_NUMBERBATCH_URL
  CONCEPTNET_OUTPUT
  CONCEPTNET_ARCHIVE

Source:
  ${DEFAULT_URL}

License and attribution:
  ConceptNet Numberbatch is distributed under CC-BY-SA 4.0.
  Suggested attribution: semantic vectors from ConceptNet Numberbatch,
  by Luminoso Technologies, Inc.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --keep-archive)
      KEEP_ARCHIVE=1
      shift
      ;;
    --url)
      URL="${2:?Missing value for --url}"
      shift 2
      ;;
    --output)
      OUTPUT="${2:?Missing value for --output}"
      shift 2
      ;;
    --archive)
      ARCHIVE="${2:?Missing value for --archive}"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

cd_repo_root
require_cmd curl
require_cmd gzip

ensure_dir "$(dirname "$OUTPUT")"
ensure_dir "$(dirname "$ARCHIVE")"

if [[ -f "$OUTPUT" && "$FORCE" -eq 0 ]]; then
  header="$(sed -n '1p' "$OUTPUT")"
  if [[ "$header" != "$EXPECTED_HEADER" ]]; then
    die "Existing $OUTPUT does not look like Numberbatch ${VERSION} English-only data; rerun with --force to replace it."
  fi
  info "ConceptNet Numberbatch already exists at $OUTPUT"
  exit 0
fi

if [[ "$FORCE" -eq 1 ]]; then
  rm -f "$OUTPUT" "$ARCHIVE"
fi

if [[ -f "$ARCHIVE" ]] && gzip -t "$ARCHIVE" >/dev/null 2>&1; then
  info "Using existing validated archive $ARCHIVE"
else
  info "Downloading ConceptNet Numberbatch ${VERSION} English-only vectors"
  info "$URL"
  curl \
    --fail \
    --location \
    --retry 3 \
    --retry-delay 5 \
    --continue-at - \
    --output "$ARCHIVE" \
    "$URL"
fi

info "Validating archive $ARCHIVE"
gzip -t "$ARCHIVE"

tmp_output="${OUTPUT}.tmp.$$"
trap 'rm -f "$tmp_output"' EXIT

info "Decompressing to $OUTPUT"
gzip -dc "$ARCHIVE" >"$tmp_output"

header="$(sed -n '1p' "$tmp_output")"
if [[ "$header" != "$EXPECTED_HEADER" ]]; then
  die "Downloaded file header was '$header', expected '$EXPECTED_HEADER'."
fi

mv "$tmp_output" "$OUTPUT"
trap - EXIT

if [[ "$KEEP_ARCHIVE" -eq 0 ]]; then
  rm -f "$ARCHIVE"
fi

bytes="$(wc -c <"$OUTPUT" | tr -d ' ')"
info "Wrote $OUTPUT ($bytes bytes)"
