"""A2A Agent Card for PiCrawler patrol agent."""

from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from agent_picrawler.config import AGENT_ID, AGENT_PORT

AGENT_SKILL = AgentSkill(
    id="ground_patrol",
    name="Ground Patrol",
    description="Ground-based search and patrol using PiCrawler robot with camera, "
                "face/color/QR detection, and obstacle avoidance",
    tags=["crawler", "search", "ground", "patrol", "detection", "surveillance", "indoor"],
    examples=[
        "Patrol this area for people",
        "Search for red objects",
        "Scan the zone for QR codes",
        "Look for faces in the area",
    ],
)

AGENT_CARD = AgentCard(
    name=f"Crawler Agent - {AGENT_ID}",
    id=AGENT_ID,
    description="Ground patrol robot with camera and detection capabilities.",
    url=f"http://0.0.0.0:{AGENT_PORT}",
    version="1.0.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text", "application/json"],
    capabilities=AgentCapabilities(streaming=False),
    skills=[AGENT_SKILL],
)
