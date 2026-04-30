#!/bin/bash
#
# run_ocr.sh - OCR Pipeline for FinanceBench PDFs
#

set -e

# [환경 설정]
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

. "$SCRIPT_DIR/scripts/lib.sh"

# Default values
N_COMPANIES=""
PDF_DIR="data/finance_pdfs"
OCR_OUTPUT="data/finance_corpus_ocr"
SAMPLE=""
TABLE_CONVERSION_FLAG="--no_convert_tables"

# Parse arguments
while [ $# -gt 0 ]; do
    case $1 in
        --n) N_COMPANIES="--n $2"; shift 2 ;;
        --pdf-dir) PDF_DIR="$2"; shift 2 ;;
        --output) OCR_OUTPUT="$2"; shift 2 ;;
        --sample) SAMPLE="--sample"; shift 1 ;;
        --convert_tables) TABLE_CONVERSION_FLAG="--convert_tables"; shift 1 ;;
        --no_convert_tables) TABLE_CONVERSION_FLAG="--no_convert_tables"; shift 1 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Treat --n as sample mode as well (same semantics as main.py).
if { [ "$SAMPLE" == "--sample" ] || [ -n "$N_COMPANIES" ]; } && [ "$OCR_OUTPUT" == "data/finance_corpus_ocr" ]; then
    OCR_OUTPUT="data/finance_corpus_sample_ocr"
fi

echo "========================================="
echo "     OCR Pipeline Pre-flight Check       "
echo "========================================="

echo "Step 1: Checking/Starting model services..."

# Start Generation Server
./run_servers.sh gen
if ! wait_for_server "http://localhost:28000/v1/models" "Generation Model" "200"; then
    echo "Fatal error: Generation server failed." >&2
    exit 1
fi

# Start OCR Server
./run_servers.sh ocr
if ! wait_for_server "http://localhost:28001/v1/models" "OCR Model" "200"; then
    echo "Fatal error: OCR server failed." >&2
    exit 1
fi

# [2] Run OCR
echo ""
echo "[Step] Running OCR pipeline..."
python main.py --mode ocr --pdf_dir "$PDF_DIR" --ocr_output "$OCR_OUTPUT" $TABLE_CONVERSION_FLAG $N_COMPANIES $SAMPLE
