"""Config flow for TypeSafe integration."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.helpers import httpx_client, selector

from .client import TypeSafeAuthError, TypeSafeClient, TypeSafeError
from .const import (
    API_BASE_URL,
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_COMPOUND_THRESHOLD,
    CONF_CONFIDENCE_THRESHOLD,
    CONF_DOMAIN_FILTER_MODE,
    CONF_FALLBACK_AGENT,
    CONF_MODEL,
    CONF_RETRIEVER_TYPE,
    DEFAULT_COMPOUND_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DOMAIN_FILTER_MODE,
    DEFAULT_MODEL,
    DEFAULT_NAME,
    DEFAULT_RETRIEVER_TYPE,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


class TypeSafeConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for TypeSafe."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial user step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            http_client = httpx_client.get_async_client(self.hass)
            client = TypeSafeClient(
                api_key=user_input[CONF_API_KEY],
                http_client=http_client,
                model=user_input.get(CONF_MODEL, DEFAULT_MODEL),
                base_url=user_input.get(CONF_BASE_URL, API_BASE_URL),
            )
            try:
                await client.async_validate_key()
            except TypeSafeAuthError:
                errors["base"] = "invalid_auth"
            except (TypeSafeError, Exception):
                errors["base"] = "cannot_connect"
            else:
                return self.async_create_entry(
                    title=DEFAULT_NAME,
                    data=user_input,
                )

        schema = vol.Schema(
            {
                vol.Required(CONF_API_KEY): selector.TextSelector(
                    selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
                ),
                vol.Optional(
                    CONF_BASE_URL, default=API_BASE_URL
                ): selector.TextSelector(),
                vol.Optional(
                    CONF_MODEL, default=DEFAULT_MODEL
                ): selector.TextSelector(),
            }
        )
        return self.async_show_form(
            step_id="user",
            data_schema=schema,
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return TypeSafeOptionsFlowHandler()


class TypeSafeOptionsFlowHandler(OptionsFlow):
    """Handle options flow for TypeSafe."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_CONFIDENCE_THRESHOLD,
                    default=self.config_entry.options.get(
                        CONF_CONFIDENCE_THRESHOLD, DEFAULT_CONFIDENCE_THRESHOLD
                    ),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        mode=selector.NumberSelectorMode.SLIDER,
                    )
                ),
                vol.Optional(
                    CONF_COMPOUND_THRESHOLD,
                    default=self.config_entry.options.get(
                        CONF_COMPOUND_THRESHOLD, DEFAULT_COMPOUND_THRESHOLD
                    ),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        mode=selector.NumberSelectorMode.SLIDER,
                    )
                ),
                vol.Optional(
                    CONF_RETRIEVER_TYPE,
                    default=self.config_entry.options.get(
                        CONF_RETRIEVER_TYPE, DEFAULT_RETRIEVER_TYPE
                    ),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=["lexical", "exhaustive"],
                        translation_key="retriever_type",
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(
                    CONF_DOMAIN_FILTER_MODE,
                    default=self.config_entry.options.get(
                        CONF_DOMAIN_FILTER_MODE, DEFAULT_DOMAIN_FILTER_MODE
                    ),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=["none", "strict", "boost"],
                        translation_key="domain_filter_mode",
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(
                    CONF_FALLBACK_AGENT,
                    description={
                        "suggested_value": self.config_entry.options.get(
                            CONF_FALLBACK_AGENT
                        )
                    },
                ): selector.ConversationAgentSelector(),
            }
        )
        return self.async_show_form(step_id="init", data_schema=schema)
