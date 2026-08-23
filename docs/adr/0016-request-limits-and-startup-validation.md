# ADR-0016: Request limits in the app, and an index that fails closed

Status: accepted

## Context
The service is meant to sit behind a proxy on a container host with one worker per
replica. Each request can cost several LLM calls, the index is fetched from a url at
boot, and the same code serves a public page.

## Decision
* Per client rate limit (token bucket: RATE_LIMIT_PER_MINUTE, burst of a sixth of it),
  an in-flight cap (MAX_INFLIGHT_REQUESTS, 503 with Retry-After when full), a request
  body cap (MAX_BODY_BYTES, 413), one deadline per request, a global cap on LLM calls in
  flight. The client ip comes from x-forwarded-for only with TRUST_PROXY_HEADERS=1 (the
  container sets it; a laptop does not).
* Browser headers on every response (nosniff, frame deny, referrer policy) and a content
  security policy on the html pages that allows images only from the catalog's hosts.
* INDEX_URL must be http(s) on a public host; loopback, private and link-local addresses
  are refused (INDEX_ALLOW_PRIVATE_URL for an internal mirror), redirects are checked the
  same way, archives are capped in members, depth, size and file types, permissions are
  ours, and a downloaded index is validated by the real loader before it is installed.
* The loader refuses anything it cannot vouch for: pipeline version, model name and the
  pinned model revision (a missing value is a mismatch), row counts and order, every
  serving column, checksums of every file, finite embeddings.
* Startup failures keep /health alive with a curated message (paths and anything key-like
  scrubbed) and make /ready and /recommend answer 503, or exit with STARTUP_FAIL_FAST=1.

## Why
The limits are the second line behind the edge, cheap, and they make the cost of a burst
bounded and visible. Fail-closed index loading is the one invariant a retrieval system
cannot recover from at request time: an embedding row next to the wrong product row is a
silent wrong answer, not an error.

## Consequences
* A client behind a shared NAT shares one bucket; the limits are deliberately loose
  (60 a minute) and can be switched off.
* A locally built index must carry the pinned revision; `make index` records it.
* The lock file next to the index directory stays forever (removing it would let a late
  worker lock a different inode).

## Alternatives considered
Leaving limits to the edge only (fine for Railway, not for `make serve` on a laptop),
an API key on /recommend (out of scope for a demo, one middleware away), a database for
the rate limiter (Redis is in the production plan for multi-replica deployments).
