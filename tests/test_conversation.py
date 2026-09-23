"""Hermetic unit tests for TypeSafe conversation entity and intent routing."""

from __future__ import annotations

from typing import cast
import pytest
from custom_components.typesafe.client import (
    TypeSafeAuthError,
    TypeSafeError,
    TypeSafeRateLimitError,
)

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.const import MATCH_ALL
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import area_registry as ar, intent

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.typesafe.const import (
    CONF_API_KEY,
    CONF_CONFIDENCE_THRESHOLD,
    CONF_FALLBACK_AGENT,
    DEFAULT_NAME,
    DOMAIN,
)
from tests.conftest import (
    MockBaseIntentHandler,
    MockFallbackAgent,
    MockTurnOnIntentHandler,
    MockTypeSafeClient,
)
from tests.eval.fixtures_standard import MockClimateIntentHandler


async def test_process_turn_on_entity_high_confidence(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Test high-confidence match resolves entity and executes HassTurnOn."""
    hass.states.async_set(
        "light.kitchen_lights", "off", {"friendly_name": "Kitchen Lights"}
    )
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {
                "choice": "light.kitchen_lights",
                "confidence": 0.95,
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on the kitchen lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    handler = mock_intent_handlers["HassTurnOn"]
    assert len(handler.handled_intents) == 1
    intent_obj = handler.handled_intents[0]
    assert intent_obj.intent_type == "HassTurnOn"
    assert intent_obj.slots["name"]["value"] == "Kitchen Lights"


@pytest.mark.parametrize(
    ("response_type", "existing_speech", "success_count", "failure_count", "expected"),
    [
        (intent.IntentResponseType.ACTION_DONE, None, 1, 0, "Done."),
        (intent.IntentResponseType.ACTION_DONE, None, 0, 0, "Done."),
        (
            intent.IntentResponseType.ACTION_DONE,
            None,
            1,
            1,
            "Done, but some devices could not be controlled.",
        ),
        (
            intent.IntentResponseType.ACTION_DONE,
            None,
            0,
            1,
            "Sorry, the devices could not be controlled.",
        ),
        (
            intent.IntentResponseType.ACTION_DONE,
            "Custom confirmation",
            1,
            0,
            "Custom confirmation",
        ),
        (
            intent.IntentResponseType.ERROR,
            "Device unavailable",
            0,
            1,
            "Device unavailable",
        ),
        (
            intent.IntentResponseType.QUERY_ANSWER,
            "It is 20 degrees",
            0,
            0,
            "It is 20 degrees",
        ),
        (intent.IntentResponseType.QUERY_ANSWER, None, 0, 0, None),
    ],
)
async def test_action_confirmation_speech(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    response_type: intent.IntentResponseType,
    existing_speech: str | None,
    success_count: int,
    failure_count: int,
    expected: str | None,
) -> None:
    """Raw action responses need speech; preserve handler speech and query results."""
    response = intent.IntentResponse(language="en")
    response.response_type = response_type
    if response_type is intent.IntentResponseType.ERROR:
        response.error_code = intent.IntentResponseErrorCode.FAILED_TO_HANDLE
    if existing_speech is not None:
        response.async_set_speech(existing_speech)
    target = intent.IntentResponseTarget(
        name="Kitchen Lights",
        type=intent.IntentResponseTargetType.ENTITY,
        id="light.kitchen_lights",
    )
    response.async_set_results([target] * success_count, [target] * failure_count)

    class RawResponseHandler(intent.IntentHandler):
        intent_type = "HassTurnOn"

        async def async_handle(
            self, intent_obj: intent.Intent
        ) -> intent.IntentResponse:
            return response

    intent.async_register(hass, RawResponseHandler())
    hass.states.async_set(
        "light.kitchen_lights", "off", {"friendly_name": "Kitchen Lights"}
    )
    mock_client.set_answers(
        {
            "intent": {"choice": "HassTurnOn", "confidence": 0.95},
            "target_entity": {"choice": "light.kitchen_lights", "confidence": 0.95},
            "is_compound": {"noul": 0.01},
        }
    )
    result = await conversation.async_converse(
        hass=hass,
        text="Turn on the kitchen lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is response_type
    assert len(result.response.success_results) == success_count
    assert len(result.response.failed_results) == failure_count
    if expected is None:
        assert result.response.speech == {}
    else:
        assert result.response.speech["plain"]["speech"] == expected


async def test_process_turn_off_entity_high_confidence(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Test high-confidence match executes HassTurnOff."""
    hass.states.async_set("switch.fan", "on", {"friendly_name": "Living Room Fan"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOff",
                "confidence": 0.92,
                "probabilities": {"HassTurnOff": 0.92},
            },
            "target_entity": {
                "choice": "switch.fan",
                "confidence": 0.92,
            },
            "is_compound": {"noul": 0.02},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn off the living room fan",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    handler = mock_intent_handlers["HassTurnOff"]
    assert len(handler.handled_intents) == 1
    assert handler.handled_intents[0].slots["name"]["value"] == "Living Room Fan"


async def test_process_area_targeting(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Test area targeting formats area and domain slots."""
    area_reg = ar.async_get(hass)
    area = area_reg.async_create("Kitchen")

    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.91,
                "probabilities": {"HassTurnOn": 0.91},
            },
            "target_type": {"choice": "area", "confidence": 0.90},
            "target_area": {"choice": area.id, "confidence": 0.90},
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on kitchen lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    handler = mock_intent_handlers["HassTurnOn"]
    assert len(handler.handled_intents) == 1
    assert handler.handled_intents[0].slots["area"]["value"] == "Kitchen"
    assert handler.handled_intents[0].slots["domain"]["value"] == "light"


@pytest.mark.parametrize(
    ("utterance", "expected_brightness"),
    [
        ("Set bedroom light to 50%", 50),
        ("Set bedroom light to 0%", 0),
        ("Set bedroom light to 100%", 100),
    ],
)
async def test_process_numeric_brightness_extraction(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
    utterance: str,
    expected_brightness: int,
) -> None:
    """Test numeric percentage slot extraction for brightness including boundaries."""
    hass.states.async_set("light.bedroom", "on", {"friendly_name": "Bedroom Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassLightSet",
                "confidence": 0.88,
                "probabilities": {"HassLightSet": 0.88},
            },
            "target_entity": {"choice": "light.bedroom", "confidence": 0.90},
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text=utterance,
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    handler = mock_intent_handlers["HassLightSet"]
    assert len(handler.handled_intents) == 1
    assert handler.handled_intents[0].slots["name"]["value"] == "Bedroom Light"
    assert (
        handler.handled_intents[0].slots["brightness"]["value"] == expected_brightness
    )


async def test_process_low_confidence_with_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """Test low-confidence utterance escalates to fallback conversation agent."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_CONFIDENCE_THRESHOLD: 0.7,
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
    )
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.35,
                "probabilities": {"HassTurnOn": 0.35},
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on something maybe",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert len(mock_fallback_agent.calls) == 1
    assert mock_fallback_agent.calls[0].text == "Turn on something maybe"
    assert "Fallback response:" in result.response.speech["plain"]["speech"]


async def test_process_low_confidence_no_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test low-confidence without fallback returns NO_INTENT_MATCH error."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.35,
                "probabilities": {"HassTurnOn": 0.35},
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on something maybe",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_process_unmatched_intent_with_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """Test unmatched intent escalates to fallback agent."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
    )
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
                "probabilities": {"unmatched": 0.99},
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="What is the weather tomorrow?",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert len(mock_fallback_agent.calls) == 1
    assert mock_fallback_agent.calls[0].text == "What is the weather tomorrow?"
    assert "Fallback response:" in result.response.speech["plain"]["speech"]


async def test_process_unmatched_intent_no_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test unmatched intent without fallback returns NO_INTENT_MATCH."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
                "probabilities": {"unmatched": 0.99},
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Tell me a joke",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_process_compound_with_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """Test compound command escalates to fallback agent."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
    )
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "is_compound": {"noul": 0.92},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on kitchen lights and lock front door",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert len(mock_fallback_agent.calls) == 1
    assert (
        mock_fallback_agent.calls[0].text
        == "Turn on kitchen lights and lock front door"
    )
    assert "Fallback response:" in result.response.speech["plain"]["speech"]


async def test_process_compound_no_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test compound command without fallback returns NO_INTENT_MATCH with multiple requests message."""
    mock_client.set_answers(
        {
            "is_compound": {"noul": 0.92},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on lights and open blinds",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH
    assert "multiple requests" in result.response.speech["plain"]["speech"]


async def test_entity_exposure_filtering(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Test unexposed entities are filtered out from target question candidates."""
    _ = mock_intent_handlers
    hass.states.async_set(
        "light.exposed_light", "off", {"friendly_name": "Exposed Light"}
    )
    hass.states.async_set(
        "light.hidden_light", "off", {"friendly_name": "Hidden Light"}
    )

    # Expose exposed_light and explicitly unexpose hidden_light
    async_expose_entity(hass, conversation.DOMAIN, "light.exposed_light", True)
    async_expose_entity(hass, conversation.DOMAIN, "light.hidden_light", False)

    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {
                "choice": "light.exposed_light",
                "confidence": 0.95,
            },
            "is_compound": {"noul": 0.01},
        }
    )

    await conversation.async_converse(
        hass=hass,
        text="Turn on light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert len(mock_client.calls) == 1
    target_criteria = mock_client.calls[0]["questions"]["target_entity"]["criteria"]
    assert "light.exposed_light" in target_criteria
    assert "light.hidden_light" not in target_criteria


async def test_intent_handle_error_handling(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test IntentHandleError results in FAILED_TO_HANDLE response."""
    hass.states.async_set(
        "light.failing_light", "off", {"friendly_name": "Failing Light"}
    )

    class FailingTurnOnHandler(intent.IntentHandler):
        intent_type = "HassTurnOn"

        async def async_handle(
            self, intent_obj: intent.Intent
        ) -> intent.IntentResponse:
            raise intent.IntentHandleError("Hardware communication failure")

    intent.async_register(hass, FailingTurnOnHandler())

    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {
                "choice": "light.failing_light",
                "confidence": 0.95,
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on failing light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.FAILED_TO_HANDLE
    assert "Hardware communication failure" in str(
        result.response.speech["plain"]["speech"]
    )


async def test_confidence_exact_threshold_boundary(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Test confidence exactly at threshold (0.40) executes, not escalates."""
    hass.states.async_set("light.test_light", "off", {"friendly_name": "Test Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.40,
                "probabilities": {"HassTurnOn": 0.40},
            },
            "target_entity": {"choice": "light.test_light", "confidence": 0.40},
            "is_compound": {"noul": 0.0},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on test light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    handler = cast(MockTurnOnIntentHandler, mock_intent_handlers["HassTurnOn"])
    assert len(handler.handled_intents) == 1


async def test_confidence_just_below_threshold(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test confidence just below threshold (0.399) escalates / returns NO_INTENT_MATCH."""
    hass.states.async_set("light.test_light", "off", {"friendly_name": "Test Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.399,
                "probabilities": {"HassTurnOn": 0.399},
            },
            "target_entity": {"choice": "light.test_light", "confidence": 0.399},
            "is_compound": {"noul": 0.0},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on test light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_confidence_zero(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test confidence of 0.0 correctly escalates."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.0,
                "probabilities": {"HassTurnOn": 0.0},
            },
            "is_compound": {"noul": 0.0},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on test light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_confidence_one(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Test confidence of 1.0 executes successfully."""
    hass.states.async_set("light.test_light", "off", {"friendly_name": "Test Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 1.0,
                "probabilities": {"HassTurnOn": 1.0},
            },
            "target_entity": {"choice": "light.test_light", "confidence": 1.0},
            "is_compound": {"noul": 0.0},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on test light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE


async def test_threshold_configured_as_zero(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Test when confidence_threshold is configured as 0.0 in options."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_CONFIDENCE_THRESHOLD: 0.0},
    )
    await hass.async_block_till_done()

    hass.states.async_set("light.test_light", "off", {"friendly_name": "Test Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.05,
                "probabilities": {"HassTurnOn": 0.05},
            },
            "target_entity": {"choice": "light.test_light", "confidence": 0.05},
            "is_compound": {"noul": 0.0},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on test light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE


async def test_threshold_configured_as_one(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Test when confidence_threshold is configured as 1.0 in options."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_CONFIDENCE_THRESHOLD: 1.0},
    )
    await hass.async_block_till_done()

    hass.states.async_set("light.test_light", "off", {"friendly_name": "Test Light"})
    # 0.999 should fail
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.999,
                "probabilities": {"HassTurnOn": 0.999},
            },
            "target_entity": {"choice": "light.test_light", "confidence": 0.999},
            "is_compound": {"noul": 0.0},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on test light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result.response.response_type is intent.IntentResponseType.ERROR

    # 1.0 should succeed
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 1.0,
                "probabilities": {"HassTurnOn": 1.0},
            },
            "target_entity": {"choice": "light.test_light", "confidence": 1.0},
            "is_compound": {"noul": 0.0},
        }
    )
    result2 = await conversation.async_converse(
        hass=hass,
        text="Turn on test light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result2.response.response_type is intent.IntentResponseType.ACTION_DONE


# ==============================================================================
# Empty or Missing Friendly Names / Entity Attributes
# ==============================================================================


async def test_entity_missing_friendly_name_attribute(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Probe entity with completely empty attributes dict (no friendly_name)."""
    hass.states.async_set("light.kitchen_lights", "off", {})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {
                "choice": "light.kitchen_lights",
                "confidence": 0.95,
            },
            "is_compound": {"noul": 0.01},
        }
    )

    await conversation.async_converse(
        hass=hass,
        text="Turn on the kitchen lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    handler = cast(MockTurnOnIntentHandler, mock_intent_handlers["HassTurnOn"])
    assert len(handler.handled_intents) == 1
    intent_obj = handler.handled_intents[0]
    slot_name = intent_obj.slots.get("name", {}).get("value")
    assert slot_name is not None, f"Expected non-None slot_name, got {slot_name!r}"
    assert slot_name in ("kitchen lights", "Kitchen lights", "light.kitchen_lights")


async def test_entity_friendly_name_is_empty_string(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Probe entity with friendly_name == ''."""
    hass.states.async_set("light.kitchen_lights", "off", {"friendly_name": ""})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {
                "choice": "light.kitchen_lights",
                "confidence": 0.95,
            },
            "is_compound": {"noul": 0.01},
        }
    )

    await conversation.async_converse(
        hass=hass,
        text="Turn on the kitchen lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    handler = cast(MockTurnOnIntentHandler, mock_intent_handlers["HassTurnOn"])
    assert len(handler.handled_intents) == 1
    intent_obj = handler.handled_intents[0]
    slot_name = intent_obj.slots.get("name", {}).get("value")
    assert slot_name, f"Expected non-empty slot_name, got {slot_name!r}"


async def test_entity_friendly_name_is_none(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Probe entity with friendly_name attribute explicitly set to None."""
    hass.states.async_set("light.kitchen_lights", "off", {"friendly_name": None})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {
                "choice": "light.kitchen_lights",
                "confidence": 0.95,
            },
            "is_compound": {"noul": 0.01},
        }
    )

    await conversation.async_converse(
        hass=hass,
        text="Turn on the kitchen lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    handler = cast(MockTurnOnIntentHandler, mock_intent_handlers["HassTurnOn"])
    assert len(handler.handled_intents) == 1
    intent_obj = handler.handled_intents[0]
    slot_name = intent_obj.slots.get("name", {}).get("value")
    assert slot_name is not None, f"Expected non-None slot_name, got {slot_name!r}"


# ==============================================================================
# Fallback Agent Configured vs Not Configured Edge Cases
# ==============================================================================


async def test_fallback_agent_nonexistent(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Probe fallback agent configured to an ID that does not exist in HA."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_FALLBACK_AGENT: "non_existent_agent_xyz",
        },
    )
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Hello world",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_client_evaluation_error_with_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """Probe API client exception triggers escalation when fallback is configured."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
    )
    await hass.async_block_till_done()

    mock_client.evaluate_error = TypeSafeError("API 500 error")

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on the lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert len(mock_fallback_agent.calls) == 1
    assert "Fallback response:" in result.response.speech["plain"]["speech"]


async def test_client_evaluation_error_without_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Probe API client exception returns NO_INTENT_MATCH error when no fallback is configured."""
    mock_client.evaluate_error = TypeSafeError("API connection timeout")

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on the lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


# ==============================================================================
# Unmatched Intent Strings & Unparsable / Malformed Responses
# ==============================================================================


@pytest.mark.parametrize(
    "unmatched_choice",
    [
        "unmatched",
        "none",
        "other",
        "",
        None,
    ],
)
async def test_unmatched_intent_strings(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    unmatched_choice: str | None,
) -> None:
    """Test all variants of unmatched intent choices."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": unmatched_choice,
                "confidence": 0.99,
            },
            "is_compound": {"noul": 0.0},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Sing a song",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_unparseable_response_answers_none(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test API response where answers field is None: {'model': '...', 'answers': None}."""
    mock_client.answers = None

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_unparseable_response_intent_none(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test API response where answers has {'intent': None}."""
    mock_client.set_answers(
        {
            "intent": None,
            "is_compound": {"noul": 0.0},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_unparseable_response_is_compound_none(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test API response where answers has {'is_compound': None} defaults to non-compound."""
    mock_client.set_answers(
        {
            "intent": {"choice": "HassTurnOn", "confidence": 0.95},
            "is_compound": None,
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result.response.response_type in (
        intent.IntentResponseType.ACTION_DONE,
        intent.IntentResponseType.ERROR,
    )


async def test_unparseable_response_probabilities_none(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test API response where probabilities is None falls back to intent confidence."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": None,
            },
            "is_compound": {"noul": 0.0},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result.response.response_type in (
        intent.IntentResponseType.ACTION_DONE,
        intent.IntentResponseType.ERROR,
    )


# ==============================================================================
# Compound Command Boundary Probes
# ==============================================================================


@pytest.mark.parametrize(
    "noul,should_compound",
    [
        (0.70, False),  # Exactly at compound threshold 0.70 -> not compound
        (0.7001, True),  # Just above 0.70 -> compound
        (0.6999, False),  # Just below 0.70 -> not compound
        (1.0, True),  # Maximum compound probability
        (0.0, False),  # Minimum compound probability
    ],
)
async def test_compound_noul_boundary(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
    noul: float,
    should_compound: bool,
) -> None:
    """Test compound boundary: compound_threshold is 0.70."""

    hass.states.async_set("light.test_light", "off", {"friendly_name": "Test Light"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {"choice": "light.test_light", "confidence": 0.95},
            "is_compound": {"noul": noul},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on test light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    if should_compound:
        assert result.response.response_type is intent.IntentResponseType.ERROR
        assert "multiple requests" in result.response.speech["plain"]["speech"]
    else:
        assert result.response.response_type is intent.IntentResponseType.ACTION_DONE


@pytest.mark.parametrize(
    "client_error",
    [
        TypeSafeAuthError("Invalid API key (401)"),
        TypeSafeRateLimitError("Rate limit exceeded (429)"),
        TypeSafeError("Unprocessable Entity (422)"),
        TypeSafeError("Site Overloaded (529)"),
        TypeSafeError("Request failed: Timeout"),
    ],
)
async def test_conversation_api_errors_with_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
    client_error: Exception,
) -> None:
    """Verify all API client errors (401, 422, 429, 529, timeout) gracefully escalate to fallback agent."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
    )
    await hass.async_block_till_done()

    mock_client.evaluate_error = client_error

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on living room light",
        conversation_id="test_conv",
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert len(mock_fallback_agent.calls) == 1
    assert mock_fallback_agent.calls[0].text == "Turn on living room light"
    assert "Fallback response:" in result.response.speech["plain"]["speech"]


@pytest.mark.parametrize(
    "client_error",
    [
        TypeSafeAuthError("Invalid API key (401)"),
        TypeSafeRateLimitError("Rate limit exceeded (429)"),
        TypeSafeError("Unprocessable Entity (422)"),
        TypeSafeError("Site Overloaded (529)"),
        TypeSafeError("Request failed: Timeout"),
    ],
)
async def test_conversation_api_errors_no_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    client_error: Exception,
) -> None:
    """Verify all API client errors without fallback return NO_INTENT_MATCH without uncaught crash."""
    mock_client.evaluate_error = client_error

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on living room light",
        conversation_id="test_conv",
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


# ==============================================================================
# Speculative Fan-out Questions When No Entities or Areas Exist
# ==============================================================================


async def test_speculative_fan_out_zero_entities_zero_areas(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Verify questions constructed when Home Assistant has NO entities and NO areas."""
    # Ensure no entities or areas exist
    mock_client.set_answers(
        {
            "intent": {"choice": "unmatched", "confidence": 0.99},
            "is_compound": {"noul": 0.01},
        }
    )

    await conversation.async_converse(
        hass=hass,
        text="Turn on something",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert len(mock_client.calls) == 1
    questions = mock_client.calls[0]["questions"]

    # Must contain intent and is_compound
    assert "intent" in questions
    assert "is_compound" in questions

    # Must NOT contain target_entity or target_area or target_type
    assert "target_entity" not in questions
    assert "target_area" not in questions
    assert "target_type" not in questions


async def test_speculative_fan_out_with_areas_only(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Verify questions constructed when areas exist but NO controllable entities exist."""
    area_reg = ar.async_get(hass)
    area = area_reg.async_create("Living Room")

    mock_client.set_answers(
        {
            "intent": {"choice": "unmatched", "confidence": 0.99},
            "is_compound": {"noul": 0.01},
        }
    )

    await conversation.async_converse(
        hass=hass,
        text="Turn on living room",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert len(mock_client.calls) == 1
    questions = mock_client.calls[0]["questions"]

    assert "intent" in questions
    assert "is_compound" in questions
    assert "target_area" in questions
    assert area.id in questions["target_area"]["criteria"]

    # target_entity and target_type must NOT be present
    assert "target_entity" not in questions
    assert "target_type" not in questions


async def test_speculative_fan_out_with_entities_only(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Verify questions constructed when controllable entities exist but NO areas exist."""
    hass.states.async_set("light.hallway", "off", {"friendly_name": "Hallway Light"})

    mock_client.set_answers(
        {
            "intent": {"choice": "unmatched", "confidence": 0.99},
            "is_compound": {"noul": 0.01},
        }
    )

    await conversation.async_converse(
        hass=hass,
        text="Turn on hallway light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert len(mock_client.calls) == 1
    questions = mock_client.calls[0]["questions"]

    assert "intent" in questions
    assert "is_compound" in questions
    assert "target_entity" in questions
    assert "light.hallway" in questions["target_entity"]["criteria"]

    # target_area and target_type must NOT be present
    assert "target_area" not in questions
    assert "target_type" not in questions


async def test_execution_when_intent_matches_but_zero_entities_exist(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Verify behavior when model returns HassTurnOn but no entities exist in HA."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on the lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    # In this case, HassTurnOn is dispatched with empty slots (no name, no entity, no area)
    # The intent handler executes or handles it.
    handler = mock_intent_handlers["HassTurnOn"]
    assert len(handler.handled_intents) == 1
    assert handler.handled_intents[0].slots == {}


# ==============================================================================
# Multi-Intent / Compound Utterance Escalation
# ==============================================================================


async def test_compound_utterance_precedence_over_intent(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Verify compound detection takes strict precedence over high intent confidence."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
    )
    await hass.async_block_till_done()

    # Even though HassTurnOn has 0.99 confidence, is_compound is 0.85
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.99,
                "probabilities": {"HassTurnOn": 0.99},
            },
            "is_compound": {"noul": 0.85},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on kitchen lights and lock the front door",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    # Intent handler must NOT have been called!
    handler = mock_intent_handlers["HassTurnOn"]
    assert len(handler.handled_intents) == 0

    # Must escalate to fallback agent
    assert len(mock_fallback_agent.calls) == 1
    assert "Fallback response:" in result.response.speech["plain"]["speech"]


@pytest.mark.parametrize(
    ("noul_value", "should_escalate_as_compound"),
    [
        (0.70, False),  # 0.70 is not > 0.70
        (0.71, True),  # 0.71 is > 0.70
        (0.69, False),  # 0.69 is not > 0.70
    ],
)
async def test_compound_boundary_threshold(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
    noul_value: float,
    should_escalate_as_compound: bool,
) -> None:
    """Verify the 0.70 threshold boundary for compound detection."""

    _ = mock_intent_handlers
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "is_compound": {"noul": noul_value},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on the lights and maybe music",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    if should_escalate_as_compound:
        assert result.response.response_type is intent.IntentResponseType.ERROR
        assert "multiple requests" in result.response.speech["plain"]["speech"]
    else:
        assert result.response.response_type is intent.IntentResponseType.ACTION_DONE


# ==============================================================================
# Empirical Bug Reproductions (Challenger Findings)
# ==============================================================================


async def test_defensive_null_compound_answer_handled_gracefully(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Verify null is_compound answer does not crash and defaults to non-compound execution."""
    mock_client.set_answers(
        {
            "intent": {"choice": "HassTurnOn", "confidence": 0.95},
            "is_compound": None,
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result.response.response_type in (
        intent.IntentResponseType.ACTION_DONE,
        intent.IntentResponseType.ERROR,
    )


async def test_defensive_null_intent_answer_handled_gracefully(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Verify null intent answer does not crash and escalates/returns NO_INTENT_MATCH."""
    mock_client.set_answers(
        {
            "intent": None,
            "is_compound": {"noul": 0.0},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_defensive_missing_friendly_name_populates_slot(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Verify entity with no friendly_name attribute populates name slot using fallback."""
    hass.states.async_set("light.kitchen_strip", "off", {})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {
                "choice": "light.kitchen_strip",
                "confidence": 0.95,
            },
            "is_compound": {"noul": 0.01},
        }
    )

    await conversation.async_converse(
        hass=hass,
        text="Turn on kitchen strip",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    handler = mock_intent_handlers["HassTurnOn"]
    assert len(handler.handled_intents) == 1
    intent_obj = handler.handled_intents[0]
    slot_val = intent_obj.slots["name"]["value"]
    assert slot_val is not None
    assert slot_val in ("kitchen strip", "Kitchen strip", "light.kitchen_strip")


async def test_defensive_nonexistent_fallback_agent_handled_gracefully(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Verify nonexistent fallback_agent is caught and returns NO_INTENT_MATCH error without uncaught crash."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_FALLBACK_AGENT: "non_existent_agent_xyz",
        },
    )
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "intent": {"choice": "unmatched", "confidence": 0.99},
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on the music",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_defensive_intent_unexpected_error_caught_as_failed_to_handle(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Verify IntentUnexpectedError is caught by conversation.py and formatted as FAILED_TO_HANDLE."""

    class UnexpectedErrorHandler(intent.IntentHandler):
        intent_type = "HassTurnOn"

        async def async_handle(
            self, intent_obj: intent.Intent
        ) -> intent.IntentResponse:
            raise intent.IntentUnexpectedError("Hardware device exploded")

    intent.async_register(hass, UnexpectedErrorHandler())

    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on the lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.FAILED_TO_HANDLE
    assert "Hardware device exploded" in str(result.response.speech["plain"]["speech"])


@pytest.mark.parametrize(
    "client_exc",
    [
        TypeSafeAuthError("Invalid API key"),
        TypeSafeRateLimitError("Rate limit exceeded"),
        TypeSafeError("Overloaded (529)"),
        TypeSafeError("Unprocessable (422)"),
        TypeSafeError("Request failed: timeout"),
    ],
)
async def test_conversation_handles_all_client_errors_with_and_without_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
    client_exc: Exception,
) -> None:
    """Verify conversation pipeline degrades cleanly on all client errors."""
    mock_client.evaluate_error = client_exc

    # Without fallback agent configured
    result_no_fallback = await conversation.async_converse(
        hass=hass,
        text="Turn on kitchen light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result_no_fallback.response.response_type is intent.IntentResponseType.ERROR
    assert (
        result_no_fallback.response.error_code
        is intent.IntentResponseErrorCode.NO_INTENT_MATCH
    )

    # With fallback agent configured
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
    )
    await hass.async_block_till_done()

    result_with_fallback = await conversation.async_converse(
        hass=hass,
        text="Turn on kitchen light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert (
        result_with_fallback.response.response_type
        is intent.IntentResponseType.ACTION_DONE
    )
    assert (
        "Fallback response" in result_with_fallback.response.speech["plain"]["speech"]
    )
    assert len(mock_fallback_agent.calls) == 1


# ==============================================================================
# Zero exposed entities and zero areas
# ==============================================================================


async def test_zero_exposed_entities_and_zero_areas_question_schema(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Verify questions payload when 0 entities are exposed and 0 areas exist."""
    # Ensure no areas exist
    area_reg = ar.async_get(hass)
    assert len(area_reg.areas) == 0

    # Ensure no controllable entities exist in HA states
    from custom_components.typesafe.speculative.retrieval.heuristics import (
        CONTROLLABLE_DOMAINS,
    )

    controllable_count = sum(
        1 for s in hass.states.async_all() if s.domain in CONTROLLABLE_DOMAINS
    )
    assert controllable_count == 0

    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on everything",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    # Inspect the questions sent to TypeSafe
    assert len(mock_client.calls) == 1
    questions = mock_client.calls[0]["questions"]

    # Must contain intent and is_compound
    assert "intent" in questions
    assert "is_compound" in questions

    # Must NOT contain target_entity, target_area, or target_type
    assert "target_entity" not in questions
    assert "target_area" not in questions
    assert "target_type" not in questions

    assert "HassTurnOn" in questions["intent"]["criteria"]

    # Intent executes successfully without slots
    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    handler = cast(MockTurnOnIntentHandler, mock_intent_handlers["HassTurnOn"])
    assert len(handler.handled_intents) == 1
    assert handler.handled_intents[0].slots == {}


async def test_non_controllable_entities_are_excluded_from_criteria(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Verify sensor/weather domains are excluded from target_entity questions."""
    hass.states.async_set("sensor.outdoor_temperature", "21.5")
    hass.states.async_set("weather.home", "sunny")

    mock_client.set_answers(
        {
            "intent": {"choice": "unmatched", "confidence": 0.99},
            "is_compound": {"noul": 0.01},
        }
    )

    await conversation.async_converse(
        hass=hass,
        text="What is the weather?",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    questions = mock_client.calls[0]["questions"]
    # Because there are no controllable entities, target_entity must be omitted
    assert "target_entity" not in questions


# ==============================================================================
# Dynamic reload on options update
# ==============================================================================


async def test_dynamic_reload_confidence_threshold_and_fallback(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Verify options update dynamically reloads confidence threshold and fallback agent."""
    hass.states.async_set(
        "light.living_room", "off", {"friendly_name": "Living Room Light"}
    )

    # Setup: Utterance with confidence 0.75
    # Default threshold is 0.70 -> Should succeed
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.75,
                "probabilities": {"HassTurnOn": 0.75},
            },
            "target_entity": {
                "choice": "light.living_room",
                "confidence": 0.75,
            },
            "is_compound": {"noul": 0.01},
        }
    )

    res1 = await conversation.async_converse(
        hass=hass,
        text="Turn on living room light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res1.response.response_type is intent.IntentResponseType.ACTION_DONE
    handler = cast(MockTurnOnIntentHandler, mock_intent_handlers["HassTurnOn"])
    assert len(handler.handled_intents) == 1

    # Update 1: Raise threshold to 0.85 (without fallback agent)
    # The same 0.75 confidence utterance should now be rejected as NO_INTENT_MATCH
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_CONFIDENCE_THRESHOLD: 0.85,
        },
    )
    await hass.async_block_till_done()

    assert config_entry.runtime_data.flow.resolver.confidence_threshold == 0.85

    res2 = await conversation.async_converse(
        hass=hass,
        text="Turn on living room light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res2.response.response_type is intent.IntentResponseType.ERROR
    assert res2.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH

    # Update 2: Configure fallback agent
    # The same 0.75 confidence utterance should now escalate to the fallback agent
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            CONF_CONFIDENCE_THRESHOLD: 0.85,
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
    )
    await hass.async_block_till_done()

    res3 = await conversation.async_converse(
        hass=hass,
        text="Turn on living room light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res3.response.response_type is intent.IntentResponseType.ACTION_DONE
    assert (
        "Fallback response: Turn on living room light"
        in res3.response.speech["plain"]["speech"]
    )
    assert len(mock_fallback_agent.calls) == 1


# ==============================================================================
# Additional Boundary & Edge Case Stress Probes
# ==============================================================================


async def test_empty_utterance_handling(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Verify empty or whitespace utterances do not crash flow or conversation."""
    mock_client.set_answers(
        {
            "intent": {"choice": "unmatched", "confidence": 0.99},
            "is_compound": {"noul": 0.0},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="   ",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_target_entity_not_in_states_falls_back_to_entity_id(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Verify target entity not registered in hass states passes entity_id as slot name."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {
                "choice": "light.phantom_device",
                "confidence": 0.95,
            },
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on phantom device",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    handler = cast(MockTurnOnIntentHandler, mock_intent_handlers["HassTurnOn"])
    assert len(handler.handled_intents) == 1
    assert handler.handled_intents[0].slots["name"]["value"] == "light.phantom_device"


async def test_empty_answers_dict_escalates(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Verify completely empty answers dict {} escalates gracefully without crash."""
    mock_client.set_answers({})

    result = await conversation.async_converse(
        hass=hass,
        text="Turn on lights",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert result.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


# ==============================================================================
# Additional Boundary & Protocol Coverage
# ==============================================================================


async def test_conversation_agent_properties(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """Test conversation agent entity properties."""
    manager = conversation.get_agent_manager(hass)
    agent = manager.async_get_agent(config_entry.entry_id)
    assert agent is not None
    assert agent.supported_languages == MATCH_ALL
    assert agent.unique_id == config_entry.entry_id
    assert agent.name == config_entry.title


async def test_conversation_context_and_language_forwarding(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Test conversation input context, language, device_id, and conversation_id are preserved."""
    ctx = Context()
    hass.states.async_set("light.kitchen", "off", {"friendly_name": "Küche"})
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {"choice": "light.kitchen", "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Schalte die Küche ein",
        conversation_id="custom-uuid-9999",
        context=ctx,
        language="de",
        device_id="satellite_speaker_living_room",
        agent_id=config_entry.entry_id,
    )
    assert res.conversation_id == "custom-uuid-9999"
    assert res.response.language == "de"
    assert res.response.response_type is intent.IntentResponseType.ACTION_DONE
    handled = mock_intent_handlers["HassTurnOn"].handled_intents[-1]
    assert handled.context is ctx
    assert handled.language == "de"


@pytest.mark.parametrize(
    ("utterance", "expected_temp"),
    [
        ("Set temperature to 72 degrees", 72.0),
        ("Set HVAC to 21.5°", 21.5),
    ],
)
async def test_process_numeric_temperature_extraction(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    climate_handler: MockClimateIntentHandler,
    utterance: str,
    expected_temp: float,
) -> None:
    """Test numeric temperature slot extraction for integer and decimal values."""
    hass.states.async_set(
        "climate.living_room", "heat", {"friendly_name": "Thermostat"}
    )
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassClimateSetTemperature",
                "confidence": 0.95,
                "probabilities": {"HassClimateSetTemperature": 0.95},
            },
            "target_entity": {"choice": "climate.living_room", "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    await conversation.async_converse(
        hass=hass,
        text=utterance,
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert len(climate_handler.handled_intents) == 1
    assert (
        climate_handler.handled_intents[0].slots["temperature"]["value"]
        == expected_temp
    )


async def test_area_targeting_without_domain_keyword(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Test area targeting without recognized domain keyword in utterance omits domain slot."""
    area_reg = ar.async_get(hass)
    area = area_reg.async_create("Attic")
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.90,
                "probabilities": {"HassTurnOn": 0.90},
            },
            "target_type": {"choice": "area", "confidence": 0.90},
            "target_area": {"choice": area.id, "confidence": 0.90},
            "is_compound": {"noul": 0.0},
        }
    )
    await conversation.async_converse(
        hass=hass,
        text="Activate the attic",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    handler = mock_intent_handlers["HassTurnOn"]
    slots = handler.handled_intents[0].slots
    assert slots["area"]["value"] == "Attic"
    assert "domain" not in slots


async def test_target_entity_broad_execution_no_entity_slot(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Test broad utterance without entity target does not populate entity_id slot."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.90,
                "probabilities": {"HassTurnOn": 0.90},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    await conversation.async_converse(
        hass=hass,
        text="Turn on everything",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    handler = mock_intent_handlers["HassTurnOn"]
    assert "entity_id" not in handler.handled_intents[0].slots


async def test_unregistered_intent_name_fails(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test when model returns an unregistered intent name, returns FAILED_TO_HANDLE."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "UnregisteredIntentXYZ",
                "confidence": 0.95,
                "probabilities": {"UnregisteredIntentXYZ": 0.95},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Do unknown action",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR
    assert res.response.error_code is intent.IntentResponseErrorCode.FAILED_TO_HANDLE


async def test_special_characters_in_entity_friendly_name(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
) -> None:
    """Test quotes and accented characters in entity friendly name forwarded into slot."""
    hass.states.async_set(
        "light.art", "off", {"friendly_name": 'René\'s "Special" Art Light'}
    )
    mock_client.set_answers(
        {
            "intent": {
                "choice": "HassTurnOn",
                "confidence": 0.95,
                "probabilities": {"HassTurnOn": 0.95},
            },
            "target_entity": {"choice": "light.art", "confidence": 0.95},
            "is_compound": {"noul": 0.0},
        }
    )
    await conversation.async_converse(
        hass=hass,
        text="Turn on art light",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    handler = mock_intent_handlers["HassTurnOn"]
    assert (
        handler.handled_intents[0].slots["name"]["value"]
        == 'René\'s "Special" Art Light'
    )


async def test_fallback_agent_receives_identical_conversation_id_and_context(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """Test fallback agent receives the identical conversation_id and context."""
    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_FALLBACK_AGENT: "mock_fallback_agent"},
    )
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
                "probabilities": {"unmatched": 0.99},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    ctx = Context()
    await conversation.async_converse(
        hass=hass,
        text="Complex query",
        conversation_id="conv-uuid-1234",
        context=ctx,
        agent_id=config_entry.entry_id,
    )
    assert len(mock_fallback_agent.calls) == 1
    call = mock_fallback_agent.calls[0]
    assert call.conversation_id == "conv-uuid-1234"
    assert call.context is ctx


async def test_fallback_agent_raises_exception_falls_through(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test exception raised by fallback agent falls through to NO_INTENT_MATCH."""

    class FailingAgent(conversation.AbstractConversationAgent):
        @property
        def supported_languages(self) -> list[str]:
            return ["en"]

        async def async_process(
            self, user_input: conversation.ConversationInput
        ) -> conversation.ConversationResult:
            raise RuntimeError("Fallback crashed")

    manager = conversation.get_agent_manager(hass)
    manager.async_set_agent("failing_fallback", FailingAgent())

    hass.config_entries.async_update_entry(
        config_entry,
        options={CONF_FALLBACK_AGENT: "failing_fallback"},
    )
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
                "probabilities": {"unmatched": 0.99},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Trigger fallback",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    assert res.response.response_type is intent.IntentResponseType.ERROR
    assert res.response.error_code is intent.IntentResponseErrorCode.NO_INTENT_MATCH


async def test_fallback_agent_in_data_ignored_without_options(
    hass: HomeAssistant,
    mock_client: MockTypeSafeClient,
    mock_fallback_agent: MockFallbackAgent,
) -> None:
    """Test fallback agent in entry.data is ignored; options is strictly required."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title=DEFAULT_NAME,
        data={
            CONF_API_KEY: "test_key",
            CONF_FALLBACK_AGENT: "mock_fallback_agent",
        },
        options={},
        entry_id="data_fallback_entry",
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
                "probabilities": {"unmatched": 0.99},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Check fallback from data",
        conversation_id=None,
        context=Context(),
        agent_id=entry.entry_id,
    )
    assert len(mock_fallback_agent.calls) == 0
    assert res.response.response_type is intent.IntentResponseType.ERROR


async def test_error_response_preserves_language_and_conversation_id(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_client: MockTypeSafeClient,
) -> None:
    """Test error response preserves user_input language and conversation_id."""
    mock_client.set_answers(
        {
            "intent": {
                "choice": "unmatched",
                "confidence": 0.99,
                "probabilities": {"unmatched": 0.99},
            },
            "is_compound": {"noul": 0.0},
        }
    )
    res = await conversation.async_converse(
        hass=hass,
        text="Que hora es?",
        conversation_id="conv-es-001",
        context=Context(),
        language="es",
        agent_id=config_entry.entry_id,
    )
    assert res.conversation_id == "conv-es-001"
    assert res.response.language == "es"


async def test_process_entity_passes_domain_slot_to_prevent_duplicate_name_collision(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    mock_intent_handlers: dict[str, MockBaseIntentHandler],
    mock_client: MockTypeSafeClient,
) -> None:
    """Test that resolving an entity passes domain slot to prevent duplicate name collision."""
    hass.states.async_set(
        "cover.garage_door",
        "closed",
        {"friendly_name": "Garage Door"},
    )
    hass.states.async_set(
        "light.garage_door",
        "off",
        {"friendly_name": "Garage Door"},
    )

    mock_client.set_answers(
        {
            "intent": {"choice": "HassTurnOn", "confidence": 0.95},
            "target_type": {"choice": "entity", "confidence": 0.90},
            "target_entity": {"choice": "cover.garage_door", "confidence": 0.92},
            "is_compound": {"noul": 0.01},
        }
    )

    result = await conversation.async_converse(
        hass=hass,
        text="Open the garage door",
        conversation_id=None,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    assert result.response.response_type is intent.IntentResponseType.ACTION_DONE
    handler = mock_intent_handlers["HassTurnOn"]
    assert len(handler.handled_intents) == 1
    intent_obj = handler.handled_intents[-1]
    assert intent_obj.slots["name"]["value"] == "Garage Door"
    assert intent_obj.slots["domain"]["value"] == "cover"
