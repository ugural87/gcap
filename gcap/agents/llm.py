"""Model-agnostic LLM client abstraction.

Two names matter and are deliberately kept apart (Section 2 / the user's
explicit instruction): "ChatGPT 5.6 Luna" is the *product-facing* name the
research proposal and any human-readable report uses; the *real API model
id* is whatever string the provider actually expects on the wire
(``OPENAI_MODEL_PRIMARY`` / ``OPENAI_MODEL_LIGHTWEIGHT`` in the
environment). ``resolve_model_identity`` is the one place that maps
between them, and it fails loudly (``ModelIdentityMismatchError``) rather
than silently substituting one for the other — a RunRecord's
``model_snapshot`` field must always be traceable to an unambiguous real
model id, never a marketing name alone.

No real API key is available or used in this build (the user has one but
declined to share it into this sandbox). ``OpenAIAdapter`` is written to be
production-correct; ``tests/agents/test_llm.py`` exercises its request
construction and response parsing thoroughly (including full
``complete()`` round-trips via an injected ``httpx.Client`` — P2-01, see
``OpenAIAdapter``'s own docstring), but never with a real API key and
never against a real network socket — every graph/harness-level test in
this repository still runs exclusively against ``FakeLLMAdapter``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from gcap.errors import ModelIdentityMismatchError, SchemaConformanceError, TokenBudgetExceededError

# The product-facing name kept separate from any real API model id. This is
# the *only* place this literal string appears; everywhere else in the
# codebase refers to "the primary model" or "the lightweight model" by role.
PRODUCT_NAME_PRIMARY = "ChatGPT 5.6 Luna"


def approximate_token_count(text: str) -> int:
    """Conservative chars/4 heuristic. A placeholder for exact
    provider-native tokenisation (e.g. tiktoken) — deliberately not adding
    that dependency in a sandbox build with no network access to fetch
    encoding files; swapping this function's body is the only change a
    real deployment needs. Documented as a to-decide-adjacent stand-in,
    consistent with how config/registry.py flags placeholders elsewhere.
    """
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class ModelIdentity:
    """Binds the product-facing name to a real, wire-level model id for one
    role ("primary" or "lightweight")."""

    role: str
    product_name: str
    real_model_id: str


def resolve_model_identity(role: str) -> ModelIdentity:
    """Reads OPENAI_MODEL_PRIMARY / OPENAI_MODEL_LIGHTWEIGHT from the
    environment. Raises ModelIdentityMismatchError if the role is unknown
    or the environment variable is unset/blank — a run must never proceed
    with an ambiguous or silently-defaulted model identity."""
    env_var = {"primary": "OPENAI_MODEL_PRIMARY", "lightweight": "OPENAI_MODEL_LIGHTWEIGHT"}.get(role)
    if env_var is None:
        raise ModelIdentityMismatchError(f"Unknown model role {role!r}; expected 'primary' or 'lightweight'.")
    real_model_id = os.environ.get(env_var, "").strip()
    if not real_model_id:
        raise ModelIdentityMismatchError(
            f"{env_var} is unset. The product-facing name {PRODUCT_NAME_PRIMARY!r} must resolve to an "
            f"explicit real API model id via the environment before any LLM call can be made; refusing "
            "to guess or silently fall back to a hard-coded model string."
        )
    return ModelIdentity(
        role=role,
        product_name=PRODUCT_NAME_PRIMARY if role == "primary" else f"{PRODUCT_NAME_PRIMARY} (lightweight)",
        real_model_id=real_model_id,
    )


@dataclass(frozen=True)
class LLMRequest:
    """One structured-output chat request. ``metadata`` carries hints a
    *test* adapter (FakeLLMAdapter) may use to synthesize a deterministic
    response; the real adapter (OpenAIAdapter) ignores it entirely — it
    only ever sees system_prompt/user_prompt, exactly like production."""

    node_name: str
    system_prompt: str
    user_prompt: str
    response_schema_name: str
    max_output_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def prompt_hash(self) -> str:
        payload = json.dumps(
            {"node": self.node_name, "system": self.system_prompt, "user": self.user_prompt}, sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def input_token_estimate(self) -> int:
        return approximate_token_count(self.system_prompt) + approximate_token_count(self.user_prompt)


@dataclass(frozen=True)
class LLMResponse:
    raw_text: str
    parsed: dict[str, Any]
    tokens_in: int
    tokens_out: int
    latency_ms: int
    model_snapshot: str


class LLMClient(Protocol):
    """Every adapter (fake or real) implements this. ``complete`` is
    synchronous by design — the harness's concurrency (Section 8,
    ConcurrencyConfig) is expressed at the task-scheduling layer (many
    tasks, each making blocking/async-wrapped calls under a semaphore),
    not inside the client itself; keeping the client interface synchronous
    keeps FakeLLMAdapter trivial to reason about and keeps the async
    concurrency logic in exactly one place (harness/)."""

    def complete(self, request: LLMRequest) -> LLMResponse: ...


def enforce_token_budget(request: LLMRequest, *, input_cap: int, output_cap: int) -> None:
    estimate = request.input_token_estimate()
    if estimate > input_cap:
        raise TokenBudgetExceededError(
            f"{request.node_name}: estimated input tokens {estimate} exceeds cap {input_cap}."
        )
    if request.max_output_tokens > output_cap:
        raise TokenBudgetExceededError(
            f"{request.node_name}: requested max_output_tokens {request.max_output_tokens} exceeds cap {output_cap}."
        )


class OpenAIAdapter:
    """Real OpenAI-compatible adapter (httpx, chat completions with a JSON
    response format). No test in this repository ever gives it a real API
    key or lets a request reach a real network socket — no real key is
    available in this sandbox session. Included because the codebase must
    be complete and swappable-in without further work once a key is
    provisioned; ``complete()`` will raise immediately if
    ``OPENAI_API_KEY`` is unset, which is also what happens if it is
    accidentally invoked in this sandbox.

    P2-01 (v2.0.0 code review): the ``httpx.Client`` this adapter's one
    real network call goes through is now both INJECTABLE and explicit
    about environment trust, rather than a bare ``httpx.Client(base_url=...,
    timeout=...)`` constructed fresh (with httpx's own implicit default,
    ``trust_env=True``) on every single ``complete()`` call:

    - **Injectable** (``client=``): a caller — chiefly a test — can supply
      its own ``httpx.Client``, e.g. one built with ``httpx.MockTransport``,
      so ``complete()``'s full request-shaping / response-parsing logic
      can be exercised hermetically, with zero real sockets touched and no
      global ``monkeypatch.setattr(httpx.Client, "post", ...)`` reaching
      into every ``httpx.Client`` instance in the process. When omitted,
      this adapter builds and owns exactly one client lazily, on first
      use, reused across every subsequent call instead of the previous
      "new client per call" (no connection-pool reuse, nothing to inject).
      This adapter never closes a client it did not construct itself —
      ``close()``/the context-manager protocol below only ever release a
      client this adapter owns.
    - **Explicit ``trust_env``** (default ``False``): httpx's own default
      (``trust_env=True``) means a client silently reads
      ``HTTP_PROXY``/``HTTPS_PROXY``/``NO_PROXY``/``~/.netrc`` from the
      process environment. This sandbox's own outbound network already
      goes through a pre-configured agent proxy meant for the *session's*
      own tool calls — an adapter that implicitly trusted that same
      environment could silently route a real OpenAI API call (bearer
      token in the ``Authorization`` header included) through a proxy
      never intended to see it, in a way that would vary by deployment
      environment instead of being an explicit, reviewed decision. A
      deployment that genuinely needs an outbound proxy passes
      ``trust_env=True`` explicitly, or constructs and injects its own
      pre-configured ``client=httpx.Client(...)``.
    """

    def __init__(
        self,
        identity: ModelIdentity,
        *,
        base_url: str | None = None,
        timeout_s: float = 30.0,
        client: httpx.Client | None = None,
        trust_env: bool = False,
    ) -> None:
        self.identity = identity
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.timeout_s = timeout_s
        self._api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self._client = client
        # Only a client THIS adapter constructed is ever closed by it (see
        # close()'s own docstring) -- an injected client's lifecycle
        # belongs to whoever constructed and passed it in.
        self._owns_client = client is None
        self._trust_env = trust_env

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout_s, trust_env=self._trust_env)
        return self._client

    def close(self) -> None:
        """Closes the client this adapter constructed itself; a no-op both
        when no client has been built yet and when the client in use was
        injected by the caller (not this adapter's to close)."""
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> OpenAIAdapter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def complete(self, request: LLMRequest) -> LLMResponse:
        if not self._api_key:
            raise ModelIdentityMismatchError(
                "OPENAI_API_KEY is unset; refusing to make a real LLM call. This sandbox session "
                "intentionally never has a real key configured."
            )
        started = time.monotonic()
        client = self._get_client()
        resp = client.post(
            "/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self.identity.real_model_id,
                "messages": [
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt},
                ],
                "max_tokens": request.max_output_tokens,
                "response_format": {"type": "json_object"},
            },
        )
        resp.raise_for_status()
        body = resp.json()
        latency_ms = int((time.monotonic() - started) * 1000)
        raw_text = body["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise SchemaConformanceError(
                f"{request.node_name}: model response was not valid JSON: {raw_text[:200]!r}"
            ) from exc
        usage = body.get("usage", {})
        return LLMResponse(
            raw_text=raw_text,
            parsed=parsed,
            tokens_in=usage.get("prompt_tokens", request.input_token_estimate()),
            tokens_out=usage.get("completion_tokens", approximate_token_count(raw_text)),
            latency_ms=latency_ms,
            model_snapshot=self.identity.real_model_id,
        )


class FakeLLMAdapter:
    """Deterministic, network-free adapter. Every test and every M2
    integration run in this repository uses this adapter — see the module
    docstring for why. A ``responder`` callable maps an LLMRequest to a
    dict payload (the structured "model output"); the default responder
    (``default_responder``) is seeded per (node_name, prompt_hash) so the
    exact same request always produces the exact same response, matching
    the deterministic-seed discipline the rest of the codebase follows.
    """

    def __init__(self, model_snapshot: str = "fake-luna-sandbox-1.0", *, responder: Any = None) -> None:
        self.model_snapshot = model_snapshot
        self._responder = responder or default_responder
        self.call_count = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        parsed = self._responder(request)
        raw_text = json.dumps(parsed, sort_keys=True, default=str)
        return LLMResponse(
            raw_text=raw_text,
            parsed=parsed,
            tokens_in=request.input_token_estimate(),
            tokens_out=approximate_token_count(raw_text),
            latency_ms=1,
            model_snapshot=self.model_snapshot,
        )


def default_responder(request: LLMRequest) -> dict[str, Any]:
    """Synthesizes a plausible structured response purely from
    ``request.metadata`` (never from parsing ``user_prompt`` text — the
    fake adapter is a stand-in for a model's *decision*, not an NLU
    engine). Each node in agents/nodes.py documents exactly which metadata
    keys it relies on."""
    schema = request.response_schema_name
    meta = request.metadata

    if schema == "claim_review":
        # Heed a deterministic tool's flag if one was run; otherwise fall
        # back to a seeded coin flip standing in for the model's own
        # (imperfect) natural-language judgement over the visible text.
        tool_flag = meta.get("tool_flagged")
        if tool_flag is not None:
            flagged = bool(tool_flag)
        else:
            flagged = _seeded_bool(request.prompt_hash(), probability=0.15)
        return {"claim_id": meta.get("claim_id", ""), "flagged": flagged, "rationale": "fake-adapter synthesized"}

    if schema == "trade_proposal":
        return {
            "proposal_id": meta.get("proposal_id", ""),
            "trade_alpha_horizon": str(meta.get("trade_alpha_horizon", "0.02")),
            "tau_days": str(meta.get("tau_days", "180")),
            "rationale": "fake-adapter synthesized",
        }

    if schema == "pm_recommendation":
        return {
            "proposal_id": meta.get("proposal_id", ""),
            "recommend": meta.get("recommend", True),
            "rationale": "fake-adapter synthesized",
        }

    if schema == "treasury_decision":
        default_action = meta.get("suggested_action", "APPROVE")
        return {
            "proposal_id": meta.get("proposal_id", ""),
            "treasury_action": default_action,
            "rationale": "fake-adapter synthesized",
        }

    if schema == "ic_vote":
        return {
            "proposal_id": meta.get("proposal_id", ""),
            "business_approval": meta.get("business_approval", True),
            "rationale": "fake-adapter synthesized",
        }

    raise SchemaConformanceError(f"FakeLLMAdapter has no synthesis rule for schema {schema!r}.")


def _seeded_bool(prompt_hash: str, *, probability: float) -> bool:
    """A deterministic pseudo-random draw derived purely from the prompt
    hash (never from wall-clock or a mutable global RNG), so re-running
    the identical request always yields the identical draw."""
    bucket = int(prompt_hash[:8], 16) / 0xFFFFFFFF
    return bucket < probability
