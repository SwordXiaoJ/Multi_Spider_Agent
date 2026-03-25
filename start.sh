#!/bin/bash
# Start PiCrawler patrol agent

set -e

echo "============================================"
echo "  PiCrawler Agent Startup"
echo "============================================"

# Start camera stream
cd /home/pi/picrawler
python3 examples/remote_stream.py > /tmp/remote_stream.log 2>&1 &
STREAM_PID=$!
echo "[1/2] remote_stream.py started (PID=$STREAM_PID)"

# Wait for camera to be ready
echo "      Waiting for camera..."
for i in $(seq 1 15); do
    sleep 1
    if curl -s --max-time 2 http://localhost:9000/status > /dev/null 2>&1; then
        echo "      Camera ready after ${i}s"
        break
    fi
done

# Start agent
cd /home/pi
/home/pi/agent_picrawler/venv/bin/python3 -m agent_picrawler.main > /tmp/agent.log 2>&1 &
AGENT_PID=$!
echo "[2/2] Agent starting (PID=$AGENT_PID)"

# Wait for agent to be ready
for i in $(seq 1 15); do
    sleep 1
    if curl -s --max-time 2 http://localhost:9004/health > /dev/null 2>&1; then
        echo "      Agent ready after ${i}s"
        break
    fi
done

echo ""
echo "============================================"
echo "  All services running"
echo "  Camera:  PID=$STREAM_PID  port 9000"
echo "  Agent:   PID=$AGENT_PID  port 9004"
echo "  Health:  curl http://localhost:9004/health"
echo "  Stop:    ./stop.sh"
echo "============================================"

# Save PIDs for stop.sh
echo "$STREAM_PID" > /tmp/picrawler_stream.pid
echo "$AGENT_PID" > /tmp/picrawler_agent.pid

# Wait for either to exit
trap "kill $STREAM_PID $AGENT_PID 2>/dev/null" EXIT
wait
