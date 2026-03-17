# Central Computer Plan - PiCrawler Multi-Agent System

## Overview

Central computer orchestrates 4 PiCrawler robots (each running autonomous A2A agent on port 9004).
Reference codebase: `AgenticNetwork/` (already cloned on this machine).

## Pi Agent Status (already done)

Each Pi runs:
- A2A Server on port `9004` (JSON-RPC, method `message/send`)
- Health check: `GET /health` returns `{"status":"ok","agent_id":"...","uptime_s":...,"mock_mode":...}`
- Agent card: `GET /.well-known/agent-card.json`
- LangGraph workflow: plan_mission -> execute_mission -> analyze_results -> finalize
- Camera detection (face/color/QR via Vilib)
- LLM decision-making (cloud API)

## Architecture

```
User (browser/curl)
    |
    v
Gateway (FastAPI, port 8080)
    |
    v
Planner (LangGraph, port 8083)
    |--- Discovery: find online Pi agents
    |--- Route: select which agent(s) for the task
    |--- Execute: send A2A message to Pi(s)
    |--- Collect: gather results
    |--- Summarize: combine multi-agent results
    v
Pi Agents (A2A, port 9004 each)
```

## Step-by-Step Implementation

### Step 1: Project Structure

Create in the central computer project root:

```
central_picrawler/
  config.py           # Pi IPs, ports, LLM config
  discovery.py         # Find and health-check Pi agents
  planner.py           # LangGraph workflow to orchestrate agents
  a2a_client.py        # Send A2A messages to Pi agents
  gateway.py           # FastAPI HTTP API for users
  main.py              # Entry point, starts gateway + planner
  requirements.txt     # Dependencies
```

### Step 2: config.py

```python
import os

# Pi agent addresses - add/change IPs as needed
PI_AGENTS = [
    {"id": "crawler-alpha", "host": os.getenv("PI_ALPHA_HOST", "192.168.1.101"), "port": 9004},
    {"id": "crawler-beta",  "host": os.getenv("PI_BETA_HOST",  "192.168.1.102"), "port": 9004},
    {"id": "crawler-gamma", "host": os.getenv("PI_GAMMA_HOST", "192.168.1.103"), "port": 9004},
    {"id": "crawler-delta", "host": os.getenv("PI_DELTA_HOST", "192.168.1.104"), "port": 9004},
]

GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8080"))
PLANNER_PORT = int(os.getenv("PLANNER_PORT", "8083"))

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
```

### Step 3: discovery.py

Reference: `AgenticNetwork/shared/discovery/static_discovery.py`

Functionality:
- Maintain list of Pi agents from config
- Periodically health-check each Pi via `GET http://<ip>:9004/health`
- Return list of online agents with their status
- Fetch agent cards via `GET http://<ip>:9004/.well-known/agent-card.json`

Key methods:
```python
class AgentDiscovery:
    async def get_online_agents() -> list[dict]
        # For each PI_AGENTS entry, call /health with timeout=3s
        # Return those that respond with status=ok

    async def get_agent_card(host, port) -> dict
        # GET /.well-known/agent-card.json

    async def health_check(host, port) -> dict | None
        # GET /health, return response or None if unreachable
```

### Step 4: a2a_client.py

Reference: `AgenticNetwork/services/planner/tools.py` (send_message_to_agent function)

Functionality:
- Send A2A JSON-RPC `message/send` to a specific Pi agent
- Wait for response (synchronous, the Pi executes the full patrol and returns)
- Parse the response text

The A2A message format that Pi agents expect:
```json
{
    "jsonrpc": "2.0",
    "method": "message/send",
    "id": "<unique-id>",
    "params": {
        "message": {
            "messageId": "<unique-id>",
            "role": "user",
            "parts": [{"kind": "text", "text": "<user prompt or task>"}],
            "metadata": {
                "target_type": "person",
                "search_pattern": "lawnmower"
            }
        }
    }
}
```

Can also embed structured payload in text:
```
Search for red objects in zone A\n\nTask: {"target_type": "red_object", "prompt": "Search for red objects"}
```

Pi agent response format:
```json
{
    "jsonrpc": "2.0",
    "id": "...",
    "result": {
        "kind": "message",
        "parts": [{"kind": "text", "text": "Patrol Result:\nAgent: ...\nDetections: ...\n..."}],
        "role": "agent"
    }
}
```

Key methods:
```python
class A2AClient:
    async def send_task(host, port, prompt, metadata=None) -> dict
        # POST JSON-RPC to http://<host>:<port>/
        # timeout should be long (120s+) since patrol takes time

    async def send_task_to_multiple(agents, prompt) -> list[dict]
        # Send same task to multiple agents in parallel (asyncio.gather)
```

IMPORTANT: Use plain `httpx` or `requests` to POST JSON-RPC. Do NOT use the a2a-sdk client
(it requires SLIM/transport setup). Just POST the JSON-RPC directly - this is what works
and what we tested on the Pi side.

### Step 5: planner.py

Reference: `AgenticNetwork/services/planner/agent_langgraph.py`

LangGraph workflow with nodes:

```
START -> parse_intent -> discover_agents -> route_task -> execute_task -> summarize -> END
```

Nodes:

1. **parse_intent**: Use LLM to understand user request
   - Input: user prompt ("search the building for people")
   - Output: intent dict with target_type, urgency, area description
   - Can also determine if single-agent or multi-agent needed

2. **discover_agents**: Call discovery.get_online_agents()
   - Output: list of available Pi agents

3. **route_task**: Decide which agent(s) to assign
   - Strategies (reference AgenticNetwork's ExecutionStrategy):
     - SINGLE: pick one agent (e.g., closest to target area)
     - PARALLEL: send to all agents simultaneously
     - SEQUENTIAL: try one, if fails try next
   - For now, default to PARALLEL (send to all online agents)

4. **execute_task**: Send A2A messages to selected agents
   - Use a2a_client.send_task_to_multiple()
   - Collect all responses
   - Timeout handling for unresponsive agents

5. **summarize**: Combine results from all agents
   - Use LLM to merge detection results
   - Deduplicate detections
   - Generate overall mission report

### Step 6: gateway.py

Reference: `AgenticNetwork/services/gateway/gateway.py`

FastAPI server with endpoints:

```
POST /api/search          # Submit a search/patrol task
GET  /api/agents          # List all agents and their status
GET  /api/agents/{id}/health  # Health check specific agent
GET  /api/results/{task_id}   # Get task result (if async)
```

POST /api/search request body:
```json
{
    "prompt": "Search for people in the building",
    "target_type": "person",        // optional
    "strategy": "parallel",          // optional: single/parallel
    "agents": ["crawler-alpha"]      // optional: specific agents
}
```

Response:
```json
{
    "task_id": "...",
    "status": "completed",
    "agents_used": ["crawler-alpha", "crawler-beta"],
    "results": [
        {
            "agent_id": "crawler-alpha",
            "detections": 2,
            "area_covered_pct": 100.0,
            "raw_response": "..."
        }
    ],
    "summary": "2 agents searched the area. Agent alpha found 2 faces..."
}
```

### Step 7: main.py

Start gateway (which internally calls planner when requests come in).

```python
import uvicorn
from central_picrawler.gateway import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Step 8: requirements.txt

```
fastapi
uvicorn
httpx
langgraph
langchain-core
requests
```

## Testing Plan

1. Start with discovery only - verify can reach Pi agents via /health
2. Test a2a_client - send a mock patrol task to one Pi
3. Test planner workflow - full LangGraph pipeline
4. Test gateway - HTTP API end-to-end
5. Multi-agent test - send parallel tasks to multiple Pis

## Test Commands

```bash
# Test discovery
curl http://localhost:8080/api/agents

# Test search
curl -X POST http://localhost:8080/api/search \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Search for faces in the area"}'

# Direct test to Pi (bypass central)
curl -X POST http://<pi-ip>:9004/ \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"message/send","id":"test-1","params":{"message":{"messageId":"msg-1","role":"user","parts":[{"kind":"text","text":"Patrol for faces"}]}}}'
```

## Key Notes

- Pi agents run MOCK_MODE=true by default (no physical movement, safe to test)
- Pi patrol takes ~15-30 seconds per mission, set HTTP timeout to 120s+
- remote_stream.py must be running on Pi before agent can use camera
- Start Pi agent: `cd /home/pi/agent_picrawler && ./start.sh`
- Pi agent listens on 0.0.0.0:9004, accessible from central computer
- No SLIM needed - plain HTTP JSON-RPC works
- Agent card path: `/.well-known/agent-card.json` (not `/agent-card`)

## Reference Files in AgenticNetwork

| Component | Reference File |
|-----------|---------------|
| Discovery | `shared/discovery/static_discovery.py` |
| Planner LangGraph | `services/planner/agent_langgraph.py` |
| A2A message sending | `services/planner/tools.py` (send_message_to_agent) |
| Gateway | `services/gateway/gateway.py` |
| Schemas | `shared/schemas/request.py`, `result.py` |
| Config | `config/llm_config.py` |
| Agent card format | `agents/medical_agent/card.py` |
