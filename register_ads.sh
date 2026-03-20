#!/bin/bash
# Register PiCrawler agent to ADS (Agent Discovery Service)
# Usage: ./register_ads.sh [ADS_ADDRESS] [OASF_ADDRESS]

ADS_ADDRESS="${1:-10.229.117.154:8888}"
OASF_ADDRESS="${2:-10.229.117.154:31234}"

cd /home/pi
./agent_picrawler/venv/bin/python -c "
import sys; sys.path.insert(0, '/home/pi')
from agent_picrawler.card import register_to_ads
result = register_to_ads('${ADS_ADDRESS}', '${OASF_ADDRESS}')
print('Success' if result else 'Failed')
"
