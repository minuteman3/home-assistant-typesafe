"""Conversation entity for TypeSafe integration."""

from __future__ import annotations

import logging
from typing import Any, Literal

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import (
    async_should_expose,
)
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    intent,
)
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import CONF_FALLBACK_AGENT, DOMAIN
from .models import TypeSafeConfigEntry
from .speculative.context import DecisionContext
from .speculative.flow import DecisionFlow
from .speculative.scoring.engine import DecisionEngine

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: TypeSafeConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up TypeSafe conversation entity."""
    entity = TypeSafeConversationEntity(
        entry=entry,
        engine=entry.runtime_data.engine,
        flow=entry.runtime_data.flow,
    )
    async_add_entities([entity])


class TypeSafeConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
):
    """TypeSafe conversation agent and entity."""

    _attr_has_entity_name = True

    def __init__(
        self,
        entry: TypeSafeConfigEntry,
        engine: DecisionEngine,
        flow: DecisionFlow,
    ) -> None:
        """Initialize the conversation entity."""
        self._entry = entry
        self._engine = engine
        self._flow = flow
        self._attr_unique_id = entry.entry_id
        self._attr_name = entry.title

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """Register conversation agent when added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self._entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """Unregister conversation agent when removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self._entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process an incoming conversation utterance."""
        area_reg = ar.async_get(self.hass)
        entity_reg = er.async_get(self.hass)

        all_states = self.hass.states.async_all()
        exposed_states = [
            state
            for state in all_states
            if async_should_expose(self.hass, conversation.DOMAIN, state.entity_id)
        ]
        active_states = exposed_states if exposed_states else all_states

        originating_area_id: str | None = None
        if user_input.device_id:
            dev_reg = dr.async_get(self.hass)
            dev_entry = dev_reg.async_get(user_input.device_id)
            if dev_entry and dev_entry.area_id:
                originating_area_id = dev_entry.area_id

        context = DecisionContext(
            hass=self.hass,
            area_registry=area_reg,
            entity_registry=entity_reg,
            states=active_states,
            language=user_input.language,
            device_id=user_input.device_id,
            originating_area_id=originating_area_id,
        )

        decision = await self._flow.async_run(
            text=user_input.text, context=context, engine=self._engine
        )

        if decision.should_escalate or not decision.intent_name:
            fallback_agent = self._entry.options.get(CONF_FALLBACK_AGENT)

            if fallback_agent:
                _LOGGER.debug(
                    "Escalating utterance %r to fallback agent %s (reason: %s)",
                    user_input.text,
                    fallback_agent,
                    decision.escalation_reason,
                )
                try:
                    return await conversation.async_converse(
                        hass=self.hass,
                        text=user_input.text,
                        conversation_id=user_input.conversation_id,
                        context=user_input.context,
                        language=user_input.language,
                        agent_id=fallback_agent,
                        device_id=user_input.device_id,
                    )
                except (ValueError, Exception) as err:
                    _LOGGER.warning(
                        "Fallback conversation agent %s failed: %s",
                        fallback_agent,
                        err,
                    )

            # Fallback failed or no fallback agent configured -> return NO_INTENT_MATCH
            intent_response = intent.IntentResponse(language=user_input.language)
            if decision.is_compound:
                msg = (
                    "I heard multiple requests. Please give one command at a time "
                    "or configure a fallback conversation agent."
                )
            else:
                msg = "Sorry, I could not understand that request."
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.NO_INTENT_MATCH, msg
            )
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
            )

        # Format slots for Home Assistant intent handling
        formatted_slots: dict[str, Any] = {}
        for slot_key, slot_val in decision.slots.items():
            if slot_key == "entity_id":
                state = self.hass.states.get(slot_val)
                name = (
                    state.attributes.get("friendly_name") if state else None
                ) or slot_val
                formatted_slots["name"] = {"value": name}
                if (
                    state
                    and "domain" not in formatted_slots
                    and "domain" not in decision.slots
                ):
                    formatted_slots["domain"] = {"value": state.domain}
            elif slot_key == "domain":
                formatted_slots["domain"] = {"value": slot_val}
            else:
                formatted_slots[slot_key] = {"value": slot_val}

        if "domain" not in formatted_slots and decision.domain:
            formatted_slots["domain"] = {"value": decision.domain}

        try:
            intent_response = await intent.async_handle(
                self.hass,
                DOMAIN,
                decision.intent_name,
                formatted_slots,
                user_input.text,
                user_input.context,
                user_input.language,
            )
        except intent.IntentError as err:
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                str(err),
            )

        # Direct intent dispatch bypasses the built-in agent's response templates.
        # Give Assist a confirmation without overwriting speech or query answers.
        if (
            not intent_response.speech
            and intent_response.response_type is intent.IntentResponseType.ACTION_DONE
        ):
            if intent_response.failed_results:
                speech = (
                    "Done, but some devices could not be controlled."
                    if intent_response.success_results
                    else "Sorry, the devices could not be controlled."
                )
            else:
                speech = "Done."
            intent_response.async_set_speech(speech)

        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=user_input.conversation_id,
        )
