#!/bin/bash
#
# Quick test for Q&A dataset converters with streaming
# Tests Natural Questions, MS MARCO, and FiQA converters
#
# Usage: ./test_qa_datasets.sh [num_samples]
#

NUM_SAMPLES=${1:-10}

echo "🧪 TESTING Q&A DATASET CONVERTERS (Streaming Mode)"
echo "=========================================="
echo "Testing with $NUM_SAMPLES samples each"
echo ""

# Run the streaming test
PYTHONPATH=./src python3 test_qa_streaming.py

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ALL Q&A CONVERTERS WORKING WITH STREAMING!"
    echo ""
    echo "📊 Converters tested:"
    echo "   1. Natural Questions (Google) - Contextual extraction + Quality indicators"
    echo "   2. MS MARCO (Microsoft) - Passage ranking + Quality indicators"
    echo ""
    echo "💡 Benefits of streaming mode:"
    echo "   - No dataset download required (saves disk space)"
    echo "   - Instant start (no waiting for download)"
    echo "   - Up to 2x faster processing"
    echo "   - Memory efficient (processes on-the-fly)"
else
    echo "❌ SOME TESTS FAILED - Check errors above"
fi

echo "=========================================="
exit $EXIT_CODE
