#!/bin/bash
# Check PiCrawler agent status

echo "============================================"
echo "  PiCrawler Agent Status"
echo "============================================"

# Check camera
if pgrep -f "remote_stream.py" > /dev/null 2>&1; then
    PID=$(pgrep -f "remote_stream.py" | head -1)
    echo "  Camera:  RUNNING (PID=$PID)"
    curl -s --max-time 2 http://localhost:9000/status > /dev/null 2>&1 \
        && echo "           Port 9000 OK" \
        || echo "           Port 9000 NOT responding"
else
    echo "  Camera:  STOPPED"
fi

# Check agent
if pgrep -f "agent_picrawler.main" > /dev/null 2>&1; then
    PID=$(pgrep -f "agent_picrawler.main" | head -1)
    echo "  Agent:   RUNNING (PID=$PID)"
    HEALTH=$(curl -s --max-time 2 http://localhost:9004/health 2>/dev/null)
    if [ -n "$HEALTH" ]; then
        echo "           Port 9004 OK"
        echo "           $HEALTH"
    else
        echo "           Port 9004 NOT responding (still starting?)"
    fi
else
    echo "  Agent:   STOPPED"
fi

# Memory usage
echo ""
echo "  Memory:"
free -m | grep Mem | awk '{printf "           Total: %dMB  Used: %dMB  Free: %dMB\n", $2, $3, $7}'

echo "============================================"
