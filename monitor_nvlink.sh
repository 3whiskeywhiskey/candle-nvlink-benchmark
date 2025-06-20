#!/bin/bash

# NVLink Bandwidth Monitor Script
# Usage: ./monitor_nvlink.sh [duration_seconds] [output_file]

DURATION=${1:-300}  # Default 5 minutes
OUTPUT_FILE=${2:-"nvlink_bandwidth.log"}

echo "🚀 Starting NVLink bandwidth monitoring..."
echo "📊 Duration: ${DURATION} seconds"
echo "📝 Output file: ${OUTPUT_FILE}"
echo "💡 Monitoring NVLink Total RX/TX (metrics 60,61) + individual links"
echo ""

# Monitor NVLink Total RX/TX + some individual links with timestamps
nvidia-smi dmon \
    --gpm-metrics 60,61,62,63,64,65,66,67 \
    --options T \
    --delay 1 \
    --count ${DURATION} \
    --filename ${OUTPUT_FILE} &

DMON_PID=$!

echo "🔍 Monitoring process PID: ${DMON_PID}"
echo "📈 Real-time display (first 50 lines):"
echo ""

# Show real-time output for the first part
timeout 50 nvidia-smi dmon \
    --gpm-metrics 60,61 \
    --options T \
    --delay 1

echo ""
echo "💾 Full monitoring continues in background..."
echo "📄 Tail log file: tail -f ${OUTPUT_FILE}"
echo "🛑 Stop monitoring: kill ${DMON_PID}"

# Wait for background process
wait ${DMON_PID}
echo "✅ Monitoring completed! Data saved to ${OUTPUT_FILE}" 