"""Constants for the TypeSafe integration."""

from typing import Final

DOMAIN: Final = "typesafe"

# Configuration options
CONF_API_KEY: Final = "api_key"
CONF_BASE_URL: Final = "base_url"
CONF_MODEL: Final = "model"
CONF_CONFIDENCE_THRESHOLD: Final = "confidence_threshold"
CONF_COMPOUND_THRESHOLD: Final = "compound_threshold"
CONF_DOMAIN_FILTER_MODE: Final = "domain_filter_mode"
CONF_RETRIEVER_TYPE: Final = "retriever_type"
CONF_FALLBACK_AGENT: Final = "fallback_agent"

# Defaults
DEFAULT_MODEL: Final = "jev-latest"
DEFAULT_CONFIDENCE_THRESHOLD: Final = 0.40
DEFAULT_COMPOUND_THRESHOLD: Final = 0.70
DEFAULT_DOMAIN_FILTER_MODE: Final = "boost"
DEFAULT_RETRIEVER_TYPE: Final = "lexical"
DEFAULT_NAME: Final = "TypeSafe"


# API endpoints
API_BASE_URL: Final = "https://api.typesafe.ai"
DEFAULT_TIMEOUT: Final = 10.0
