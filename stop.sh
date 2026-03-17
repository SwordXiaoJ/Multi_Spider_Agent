#!/bin/bash
# Safely stop PiCrawler agent
# Sits robot down before killing processes

echo "Stopping PiCrawler Agent..."

# 1. Let robot sit down safely (only if not mock mode)
if pgrep -f "agent_picrawler.main" > /dev/null 2>&1; then
    echo "[1/3] Sitting robot down..."
    /home/pi/agent_picrawler/venv/bin/python3 -c "
try:
    from picrawler import Picrawler
    Picrawler().do_step('sit', 50)
    print('      Robot sat down.')
except:
    print('      Skip (mock mode or hardware not available)')
" 2>/dev/null
else
    echo "[1/3] Agent not running, skip sit"
fi

# 2. Kill agent
echo "[2/3] Stopping agent..."
pkill -f "agent_picrawler.main" 2>/dev/null && echo "      Agent stopped" || echo "      Agent was not running"
sleep 1

# 3. Kill camera stream
echo "[3/3] Stopping camera..."
pkill -f "remote_stream.py" 2>/dev/null && echo "      Camera stopped" || echo "      Camera was not running"

# Clean PID files
rm -f /tmp/picrawler_stream.pid /tmp/picrawler_agent.pid

echo ""
echo "All services stopped."
