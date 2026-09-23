# Home Assistant TypeSafe Integration

A Home Assistant custom conversation integration powered by **TypeSafe AI System One** (Jev model family) for fast, structured intent routing, entity resolution, and device control.

Developed alongside its sibling project [**home-assistant-laya**](https://github.com/allenporter/home-assistant-laya) (a 100% local, open-weight implementation running directly on local hardware), this integration provides cloud-accelerated System One decision routing with zero local GPU/CPU overhead.

## Overview

The TypeSafe integration acts as a high-speed, deterministic voice and intent decision engine for Home Assistant Assist. Built around Daniel Kahneman's "System One" (fast, intuitive decision-making) framework, it classifies natural language utterances and binds device targets in sub-200ms response windows without token-by-token text generation.

### Key Capabilities

- **Sub-200ms Decision Routing**: Evaluates natural language commands against registered Home Assistant intent schemas and exposed entities using structured TypeSafe primitives (`Choice`, `Noul`).
- **5-Stage Speculative Pipeline**: Orchestrates request tokenization, candidate retrieval, hierarchical hydration, engine scoring, and contextual target binding.
- **Dual-Process Architecture**: Resolves high-confidence routine actions directly and seamlessly escalates low-confidence or compound queries to a configured System 2 fallback agent (e.g., an LLM or Home Assistant's built-in conversation agent).
- **Contextual Disambiguation**: Uses voice satellite originating area context (`device_id` -> area) and preserves entity domains to prevent collisions across identically named devices.
- **Continuous Parameter Extraction**: Parses and binds continuous numerical values (e.g., temperature setpoints in degrees, brightness percentages) directly to intent slots.
- **Broad Intent Support**: Dynamically inspects registered Home Assistant intent handlers across lights, switches, climate devices, covers, media players, and more.

## Architecture: The 5-Stage Decision Pipeline

When an utterance is processed through Assist, the `DecisionFlow` pipeline evaluates the request across five stages:

1. **Request Processing & Tokenization**
   Extracts tokens, identifies numerical parameter slots (such as target brightness percentage or temperature degrees), and resolves originating area context.

2. **Candidate Retrieval**
   Discovers controllable devices and registered intent handlers in Home Assistant. Uses configurable retrieval strategies (lexical BM25 ranking or exhaustive evaluation) with domain filtering to scale efficiently in homes with hundreds of entities.

3. **Hierarchical Hydration**
   Constructs structured TypeSafe System One questions (`Choice` and `Noul` primitives) encompassing intent selection, target type (entity vs. area), entity/area candidates, and compound command detection in a single unified payload.

4. **Decision Engine Scoring**
   Queries the TypeSafe System One API (`jev-latest`) in a single network roundtrip, obtaining calibrated probabilities across all questions without hallucination or token streaming latency.

5. **Target Binding & Resolution**
   Binds the selected entity or area to the resolved intent. Applies confidence gating, validates domain matching to prevent cross-domain collisions, and dispatches the execution to Home Assistant's native `intent.async_handle`.

## Configuration

Install the custom component into `custom_components/typesafe` and configure it via **Settings > Devices & Services > Add Integration > TypeSafe**:

- **API Key** (`api_key`): Your TypeSafe API key.
- **Base URL** (`base_url`): API endpoint to query. Defaults to the hosted TypeSafe API (`https://api.typesafe.ai`); point it at a self-hosted, JEV-compatible endpoint (e.g. a `laya.cpp` server) to keep inference local.
- **Model** (`model`): TypeSafe model to query (defaults to `jev-latest`).

### Voice Assistant Setup

To use TypeSafe for voice or text commands:

1. Navigate to **Settings > Voice assistants**.
2. Select your desired Assist pipeline.
3. Set **Conversation agent** to **TypeSafe**.

When an action handler returns no speech, TypeSafe supplies an English confirmation
("Done."), or a warning if it reports failed targets. Existing handler speech is
preserved. This fallback applies only to action responses; it does not render
Home Assistant's localized response templates or supply missing query answers.

### Options

Tune runtime thresholds and pipeline behavior in **Settings > Devices & Services > TypeSafe > Configure**:

| Option                   | Key                    | Default   | Description                                                                                                                                                              |
| :----------------------- | :--------------------- | :-------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Confidence Threshold** | `confidence_threshold` | `0.40`    | Minimum confidence required to execute an intent. Utterances below this threshold escalate to the fallback agent.                                                        |
| **Compound Threshold**   | `compound_threshold`   | `0.70`    | Sensitivity threshold for detecting multi-action utterances (e.g., _"Turn on the lights and play music"_). Utterances scoring above this escalate to the fallback agent. |
| **Retriever Type**       | `retriever_type`       | `lexical` | Candidate retrieval strategy: `lexical` (BM25 token-scored candidate pruning for large homes) or `exhaustive` (evaluates all controllable entities).                     |
| **Domain Filter Mode**   | `domain_filter_mode`   | `boost`   | Controls how intent domains filter candidate entities: `none` (unfiltered), `strict` (strict domain matching), or `boost` (lexical score boost for matching domains).    |
| **Fallback Agent**       | `fallback_agent`       | —         | Optional secondary conversation agent (e.g., an LLM or Home Assistant Cloud) to handle complex queries, conversational dialogue, or low-confidence requests.             |

## Related Projects

- **[home-assistant-laya](https://github.com/allenporter/home-assistant-laya)**: A sibling integration that provides a 100% local, offline-capable voice decision engine running open-weight models directly on your hardware (CPU, CUDA, or Apple Silicon) using the same 5-stage speculative pipeline architecture.

## Development & Testing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [.agent/skills/developer/SKILL.md](.agent/skills/developer/SKILL.md) for local environment setup, running tests (`./script/test`), and linting (`./script/lint`).
