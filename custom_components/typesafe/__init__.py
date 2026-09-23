"""TypeSafe custom component."""

from __future__ import annotations

import logging

from collections.abc import Mapping
from typing import Any

from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import httpx_client

from .client import TypeSafeClient
from .const import (
    API_BASE_URL,
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_COMPOUND_THRESHOLD,
    CONF_CONFIDENCE_THRESHOLD,
    CONF_DOMAIN_FILTER_MODE,
    CONF_MODEL,
    CONF_RETRIEVER_TYPE,
    DEFAULT_COMPOUND_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DOMAIN_FILTER_MODE,
    DEFAULT_MODEL,
    DEFAULT_RETRIEVER_TYPE,
)
from .engine import TypeSafeDecisionEngine
from .models import TypeSafeConfigEntry, TypeSafeData
from .speculative.flow import (
    DecisionFlow,
    FlowConfig,
    create_decision_flow,
)

_LOGGER = logging.getLogger(__name__)

PLATFORMS: tuple[Platform, ...] = (Platform.CONVERSATION,)


def create_flow_from_options(options: Mapping[str, Any]) -> DecisionFlow:
    """Create a DecisionFlow configured from config entry options."""
    config = FlowConfig(
        confidence_threshold=float(
            options.get(CONF_CONFIDENCE_THRESHOLD, DEFAULT_CONFIDENCE_THRESHOLD)
        ),
        compound_threshold=float(
            options.get(CONF_COMPOUND_THRESHOLD, DEFAULT_COMPOUND_THRESHOLD)
        ),
        retriever_type=options.get(CONF_RETRIEVER_TYPE, DEFAULT_RETRIEVER_TYPE),
        domain_filter_mode=options.get(
            CONF_DOMAIN_FILTER_MODE, DEFAULT_DOMAIN_FILTER_MODE
        ),
    )
    return create_decision_flow(config)


async def async_setup_entry(hass: HomeAssistant, entry: TypeSafeConfigEntry) -> bool:
    """Set up a config entry."""
    api_key = entry.data[CONF_API_KEY]
    base_url = entry.data.get(CONF_BASE_URL, API_BASE_URL)
    model = entry.data.get(CONF_MODEL, DEFAULT_MODEL)

    http_client = httpx_client.get_async_client(hass)
    client = TypeSafeClient(
        api_key=api_key,
        http_client=http_client,
        model=model,
        base_url=base_url,
    )
    engine = TypeSafeDecisionEngine(client=client)
    flow = create_flow_from_options(entry.options)

    entry.runtime_data = TypeSafeData(client=client, engine=engine, flow=flow)

    await hass.config_entries.async_forward_entry_setups(
        entry,
        platforms=PLATFORMS,
    )

    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: TypeSafeConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(
        entry,
        PLATFORMS,
    )


async def async_reload_entry(hass: HomeAssistant, entry: TypeSafeConfigEntry) -> None:
    """Reload config entry when options change."""
    await hass.config_entries.async_reload(entry.entry_id)
