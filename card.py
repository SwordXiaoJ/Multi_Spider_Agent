"""
PiCrawler Agent Card — identity and capabilities declaration.

Used by:
- A2A server (/.well-known/agent-card.json)
- ADS self-registration on startup
"""

import hashlib
import json
import logging
from pathlib import Path

from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from agent_picrawler.config import AGENT_ID, AGENT_PORT, LOCAL_IP

logger = logging.getLogger(__name__)

# ── OASF metadata (for ADS registration) ────────────────────
OASF_SKILLS = [{"name": "advanced_reasoning_planning/strategic_planning", "id": 1501}]
OASF_DOMAINS = [{"name": "technology/software_engineering", "id": 102}]

# ── Agent Card ───────────────────────────────────────────────
AGENT_SKILL = AgentSkill(
    id="ground_patrol",
    name="Ground Patrol",
    description=(
        "Autonomous ground patrol robot. Two modes: "
        "(1) Agent mode — receives observations from Central, LLM decides all actions; "
        "(2) Execute mode — runs direct_control steps sequentially. "
        "Supports: stand, sit, walk, turn, wave, dance, push-up, look around. "
        "GET /capabilities for full action list and endpoints."
    ),
    tags=["crawler", "search", "ground", "patrol", "agent", "autonomous", "indoor", "robot"],
    examples=[
        "Detect faces, if found wave, otherwise sit",
        "Patrol and find Alice",
        "Stand up and dance",
        "Wave",
    ],
)

AGENT_CARD = AgentCard(
    name=f"Crawler Agent - {AGENT_ID}",
    id=AGENT_ID,
    description=(
        "Autonomous PiCrawler robot agent. "
        "Agent mode: LLM-driven decision making from Central's observations. "
        "Execute mode: sequential direct_control actions. "
        "GET /capabilities for supported actions and endpoints."
    ),
    url=f"http://{LOCAL_IP}:{AGENT_PORT}",
    version="3.0.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text", "application/json"],
    capabilities=AgentCapabilities(streaming=False),
    skills=[AGENT_SKILL],
)


# ── ADS Self-Registration ────────────────────────────────

_ADS_CACHE_FILE = Path(__file__).parent / ".ads_cache.json"


def _card_hash() -> str:
    """Hash of card + OASF metadata. Changes when card content changes."""
    payload = json.dumps({
        "card": json.loads(AGENT_CARD.model_dump_json()),
        "oasf_skills": OASF_SKILLS,
        "oasf_domains": OASF_DOMAINS,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def register_to_ads(ads_address: str, oasf_address: str) -> bool:
    """Register this agent's card to ADS. Skips if card unchanged since last registration.

    Args:
        ads_address: ADS gRPC address, e.g. "10.229.117.154:8888"
        oasf_address: OASF translation service, e.g. "10.229.117.154:31234"

    Returns:
        True if registration succeeded or was skipped (already registered).
    """
    current_hash = _card_hash()

    # Check cache — skip if card unchanged
    if _ADS_CACHE_FILE.exists():
        try:
            cache = json.loads(_ADS_CACHE_FILE.read_text())
            if cache.get("hash") == current_hash:
                logger.info(f"ADS: card unchanged, skipping (CID={cache.get('cid', '?')})")
                return True
        except (json.JSONDecodeError, KeyError):
            pass  # corrupt cache, re-register

    try:
        import grpc
        from google.protobuf.json_format import ParseDict, MessageToJson
        from google.protobuf.struct_pb2 import Struct
        from agntcy.dir_sdk.client import Client, Config
        from agntcy.dir_sdk.models import core_v1, routing_v1
        from agntcy.oasfsdk.translation.v1.translation_service_pb2 import A2AToRecordRequest
        from agntcy.oasfsdk.translation.v1.translation_service_pb2_grpc import TranslationServiceStub
    except ImportError as e:
        logger.warning(f"ADS SDK not installed, skipping registration: {e}")
        return False

    try:
        # Step 1: Translate AgentCard → OASF Record via translation service
        channel = grpc.insecure_channel(oasf_address)
        stub = TranslationServiceStub(channel)

        card_dict = json.loads(AGENT_CARD.model_dump_json())
        card_struct = Struct()
        card_struct.update({"a2aCard": card_dict})

        response = stub.A2AToRecord(A2AToRecordRequest(data=card_struct))
        channel.close()

        # Parse OASF record and enrich with skills/domains
        record_dict = json.loads(MessageToJson(response.record))
        # Override schema_version to match ADS server's supported schemas
        record_dict["schema_version"] = "0.8.0"
        if not record_dict.get("skills"):
            record_dict["skills"] = OASF_SKILLS
        if not record_dict.get("domains"):
            record_dict["domains"] = OASF_DOMAINS
        for module in record_dict.get("modules", []):
            data = module.get("data", {})
            if "card_schema_version" in data:
                data["protocol_version"] = data.pop("card_schema_version")

        # Step 2: Push to ADS
        data_struct = Struct()
        ParseDict(record_dict, data_struct)
        record = core_v1.Record(data=data_struct)

        config = Config(server_address=ads_address)
        client = Client(config)

        refs = client.push([record])
        cid = refs[0].cid

        record_refs = routing_v1.RecordRefs(refs=[core_v1.RecordRef(cid=cid)])
        client.publish(routing_v1.PublishRequest(record_refs=record_refs))

        # Save cache
        _ADS_CACHE_FILE.write_text(json.dumps({"hash": current_hash, "cid": cid}))

        logger.info(f"Registered to ADS: CID={cid}")
        return True

    except Exception as e:
        logger.warning(f"ADS registration failed (non-fatal): {e}")
        return False
