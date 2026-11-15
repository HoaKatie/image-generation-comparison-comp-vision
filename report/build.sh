#!/usr/bin/env bash
# Build the report:
#   report.pdf   pandoc -> HTML -> PDF with headless Google Chrome (no LaTeX needed)
#   report.docx  for editing in Word / Google Docs
#
#   bash report/build.sh                 # set CHROME=... if Chrome is somewhere else
set -euo pipefail
cd "$(dirname "$0")"
CHROME=${CHROME:-"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}

if [ -f ../results/main_results.md ]; then
  EMBED=(--self-contained)   # figures embedded in the HTML
else
  echo "Warning: results/ not found, so the figures will be empty boxes." \
       "Run the experiments first (docs/RUNNING_EXPERIMENTS.md)." >&2
  EMBED=()
fi
pandoc report.md -s --mathml "${EMBED[@]+"${EMBED[@]}"}" -c report.css -o report.html
"$CHROME" --headless --disable-gpu --no-pdf-header-footer --allow-file-access-from-files \
  --print-to-pdf="$PWD/report.pdf" "file://$PWD/report.html" 2>/dev/null
pandoc report.md -o report.docx 2>/dev/null   # missing figures only cause warnings
echo "Wrote report/report.pdf and report/report.docx"
