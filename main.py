"""
Server entry point for PiCrawler patrol agent.

Starts A2A server + optional HTTP server.
"""

import asyncio
import logging
import os
import time

from uvicorn import Config, Server
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from a2a.server.apps import A2AStarletteApplication
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.request_handlers import DefaultRequestHandler

from agent_picrawler.agent_executor_a2a import CrawlerAgentExecutor
from agent_picrawler.card import AGENT_CARD, AGENT_ID
from agent_picrawler.config import AGENT_PORT, MOCK_MODE

_start_time = time.time()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

HTTP_HOST = "0.0.0.0"


async def health(request: Request):
    uptime = int(time.time() - _start_time)
    return JSONResponse({
        "status": "ok",
        "agent_id": AGENT_ID,
        "uptime_s": uptime,
        "mock_mode": MOCK_MODE,
    })


async def run_http_server(server):
    try:
        app = server.build()
        app.add_route("/health", health, methods=["GET"])
        config = Config(
            app=app,
            host=HTTP_HOST,
            port=AGENT_PORT,
            loop="asyncio",
        )
        userver = Server(config)
        await userver.serve()
    except Exception as e:
        logger.error(f"HTTP server error: {e}")


async def main():
    print("=" * 60)
    print(f"PiCrawler Patrol Agent")
    print(f"Agent: {AGENT_ID}")
    print(f"HTTP: port {AGENT_PORT}")
    print(f"Mock Mode: {os.getenv('MOCK_MODE', 'true')}")
    print("=" * 60)

    request_handler = DefaultRequestHandler(
        agent_executor=CrawlerAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=AGENT_CARD,
        http_handler=request_handler,
    )

    await run_http_server(server)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down.")
