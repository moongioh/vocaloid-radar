"""Runtime configuration, read from the environment.

DATABASE_URL is injected by docker-compose.dev.yml (points at the `db` service).
The localhost:5433 default is only for the rare case of running against the dev
DB from outside the compose network.
"""
import os

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://vocaloid:vocaloid@localhost:5433/vocaloid",
)

# llm_gateway endpoint (plan 0001: this project is a consumer). The gateway is an
# OpenAI-compatible LiteLLM proxy; its real address is injected via GATEWAY_URL
# (a private, non-public host — never hardcoded here). The localhost default is for
# running the gateway's own docker-compose alongside this project locally.
# GATEWAY_API_KEY is the LiteLLM key; it lives in the environment, never in git.
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:4000")
GATEWAY_API_KEY = os.environ.get("GATEWAY_API_KEY", "")

# V3.2 embedding lane. gw-embed = voyage-4-large (1024d) on the gateway; the
# schema's vector(1024) is tied to this — swapping models means re-embedding.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "gw-embed")
