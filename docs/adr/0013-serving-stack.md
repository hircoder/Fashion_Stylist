# ADR-0013: FastAPI + React page + CLI; Docker on Railway

Status: accepted

## Context
The brief asks for a function, CLI or API, optionally a front end. The user's existing
demos use FastAPI serving a single page, deployed on Railway.

## Decision
FastAPI for the API (pydantic models, OpenAPI at `/docs`, `/health` and `/ready`), a small
React + Vite page served by the API (built bundle committed), an argparse CLI sharing the
same request model, a cpu-only Dockerfile and `railway.toml`.

## Why
* One process, one deployable, the page and the API share the same origin and models.
* React with plain CSS was requested; the bundle is 150 KB and needs no node at runtime.

## Consequences
* Node is needed only to rebuild the page.
