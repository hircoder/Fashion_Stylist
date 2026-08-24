# ADR-0010: Two LLM providers behind one async method, no framework

## Context
The service needs an LLM for planning and reranking. The reviewer may have an Anthropic
or an OpenAI key, or neither.

## Decision
`complete_json(system, user, schema) -> model` is the whole interface. Adapters for the
Anthropic and OpenAI SDKs use each provider's structured output; every SDK exception is
mapped to a small typed error family. `LLM_PROVIDER=none` runs the service without a key.

## Why
* About 80 lines per adapter; nothing to learn for a reader; provider contract tests run
  against a recorded client.
* No LangChain/LlamaIndex: fewer moving parts than the problem needs.

## Consequences
* Features beyond "json that matches a schema" (tools, streaming) are not exposed.
* Defaults per provider are explicit in `.env.example` with cheaper options listed.
