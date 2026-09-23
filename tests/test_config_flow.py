"""Hermetic unit tests for TypeSafe config flow and options flow."""

from __future__ import annotations

from unittest.mock import patch

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.typesafe.client import TypeSafeAuthError, TypeSafeError
from custom_components.typesafe.const import (
    API_BASE_URL,
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_COMPOUND_THRESHOLD,
    CONF_CONFIDENCE_THRESHOLD,
    CONF_DOMAIN_FILTER_MODE,
    CONF_FALLBACK_AGENT,
    CONF_MODEL,
    CONF_RETRIEVER_TYPE,
    DEFAULT_NAME,
    DOMAIN,
)
from tests.conftest import MockTypeSafeClient


async def test_user_flow_success(
    hass: HomeAssistant,
) -> None:
    """Test successful user step creating config entry."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result.get("type") is FlowResultType.FORM
    assert result.get("step_id") == "user"
    assert not result.get("errors")

    with patch(
        "custom_components.typesafe.async_setup_entry", return_value=True
    ) as mock_setup:
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_API_KEY: "valid-api-key",
                CONF_MODEL: "jev-latest",
            },
        )
        await hass.async_block_till_done()

    assert result2.get("type") is FlowResultType.CREATE_ENTRY
    assert result2.get("title") == DEFAULT_NAME
    assert result2.get("data") == {
        CONF_API_KEY: "valid-api-key",
        CONF_BASE_URL: API_BASE_URL,
        CONF_MODEL: "jev-latest",
    }
    assert len(mock_setup.mock_calls) == 1


async def test_user_flow_custom_base_url(
    hass: HomeAssistant,
) -> None:
    """Test the user flow stores a custom (e.g. self-hosted) base URL."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result.get("type") is FlowResultType.FORM

    with patch(
        "custom_components.typesafe.async_setup_entry", return_value=True
    ) as mock_setup:
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_API_KEY: "valid-api-key",
                CONF_BASE_URL: "https://laya.example.com",
                CONF_MODEL: "laya-latest",
            },
        )
        await hass.async_block_till_done()

    assert result2.get("type") is FlowResultType.CREATE_ENTRY
    assert result2.get("data") == {
        CONF_API_KEY: "valid-api-key",
        CONF_BASE_URL: "https://laya.example.com",
        CONF_MODEL: "laya-latest",
    }
    assert len(mock_setup.mock_calls) == 1


async def test_user_flow_invalid_auth(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test user step with invalid API key."""
    mock_client.validate_error = TypeSafeAuthError("Invalid API key")

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result.get("type") is FlowResultType.FORM

    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_API_KEY: "bad-api-key",
            CONF_MODEL: "jev-latest",
        },
    )
    await hass.async_block_till_done()

    assert result2.get("type") is FlowResultType.FORM
    assert result2.get("errors") == {"base": "invalid_auth"}


async def test_user_flow_cannot_connect(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test user step with network connection failure."""
    mock_client.validate_error = TypeSafeError("Connection timeout")

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result.get("type") is FlowResultType.FORM

    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_API_KEY: "any-api-key",
            CONF_MODEL: "jev-latest",
        },
    )
    await hass.async_block_till_done()

    assert result2.get("type") is FlowResultType.FORM
    assert result2.get("errors") == {"base": "cannot_connect"}


async def test_options_flow_update(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """Test options flow updating confidence threshold and fallback agent."""
    result = await hass.config_entries.options.async_init(config_entry.entry_id)
    assert result.get("type") is FlowResultType.FORM
    assert result.get("step_id") == "init"

    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            CONF_CONFIDENCE_THRESHOLD: 0.85,
            CONF_COMPOUND_THRESHOLD: 0.4,
            CONF_RETRIEVER_TYPE: "lexical",
            CONF_DOMAIN_FILTER_MODE: "boost",
            CONF_FALLBACK_AGENT: "conversation.home_assistant",
        },
    )
    await hass.async_block_till_done()

    assert result2.get("type") is FlowResultType.CREATE_ENTRY
    assert config_entry.options == {
        CONF_CONFIDENCE_THRESHOLD: 0.85,
        CONF_COMPOUND_THRESHOLD: 0.4,
        CONF_RETRIEVER_TYPE: "lexical",
        CONF_DOMAIN_FILTER_MODE: "boost",
        CONF_FALLBACK_AGENT: "conversation.home_assistant",
    }
