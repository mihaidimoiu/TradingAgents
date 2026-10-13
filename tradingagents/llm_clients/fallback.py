"""A chat model that hands a failed call to the next provider's model in its chain.

What a failure means is `classify`'s: an account the provider refuses, or an outage
its SDK's own retries did not outlast, takes that provider out of the rest of the
run (``"account"``, ``"down"``); any other failure moves that one call on
(``"call"``); None is never handed on. The last link's failure raises: alone, as
itself; after others, as ChainExhausted naming each provider's.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from typing import Any, Literal

from langchain_core.runnables import Runnable, RunnableConfig

logger = logging.getLogger(__name__)

# How each provider says the account is empty. Matched in the error's text: the
# SDKs raise it as an ordinary rate limit (OpenAI's 429) or bad request
# (Anthropic's 400).
NO_CREDIT = (
    "insufficient_quota",
    "exceeded your current quota",
    "credit balance is too low",
    "credit_balance",
    "no credits remaining",
)

# Methods that build a new model from this one; every link must be built the same way.
DERIVING = frozenset({"bind_tools", "with_structured_output"})

Why = Literal["account", "down", "call"]
Classify = Callable[[BaseException], "Why | None"]
# (provider, its error, the next link's "provider/model" or None, whether the provider is out of the run).
Told = Callable[[str, BaseException, "str | None", bool], None]


def out_of_credit(error: BaseException) -> bool:
    text = str(error).lower()
    return any(sign in text for sign in NO_CREDIT)


def credit_only(error: BaseException) -> Why | None:
    """The default reading: an empty account moves on, any other error raises."""
    return "account" if out_of_credit(error) else None


class ChainExhausted(Exception):
    """Every link of a chain failed: each provider with its error, in the chain's order."""

    def __init__(self, failures: Sequence[tuple[str, BaseException]]):
        self.failures = list(failures)
        super().__init__("; ".join(f"{provider}: {error}" for provider, error in self.failures))


class Exhausted:
    """The providers out of one run, shared by all its clients, each with the error that took it out.

    Once a provider is out, its links are passed over: every call would fail the
    same way, after the SDK's own retries. `told` hears each provider taken out
    once, and each call moved on.
    """

    def __init__(self, told: Told | None = None):
        self._errors: dict[str, BaseException] = {}
        self._told = told
        self._lock = threading.Lock()

    @property
    def errors(self) -> dict[str, BaseException]:
        """Each provider out of the run, with its error: a copy."""
        with self._lock:
            return dict(self._errors)

    @property
    def providers(self) -> set[str]:
        with self._lock:
            return set(self._errors)

    def mark(self, provider: str, error: BaseException, next_name: str | None) -> None:
        with self._lock:
            if provider in self._errors:
                return
            self._errors[provider] = error
        logger.warning("%s is out of this run (%s); its calls go to %s", provider, error, next_name or "nobody")
        if self._told is not None:
            self._told(provider, error, next_name, True)

    def moved(self, provider: str, error: BaseException, next_name: str | None) -> None:
        logger.warning("a call on %s failed (%s); it goes to %s", provider, error, next_name or "nobody")
        if self._told is not None:
            self._told(provider, error, next_name, False)


class ProviderChain(Runnable):
    """`links`, (provider, "provider/model", model), asked in order; see the module docstring."""

    def __init__(self, links: Sequence[tuple[str, str, Runnable]], exhausted: Exhausted,
                 classify: Classify = credit_only):
        self.links = list(links)
        self.exhausted = exhausted
        self.classify = classify

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        out = self.exhausted.errors
        failures = [(provider, out[provider]) for provider, _, _ in self.links if provider in out]
        live = [link for link in self.links if link[0] not in out]
        # The cause is this call's own last failure, never an error another call left in `exhausted`.
        tried: Exception | None = None
        for position, (provider, _, model) in enumerate(live):
            try:
                return model.invoke(input, config, **kwargs)
            except Exception as error:
                why = self.classify(error)
                if why is None or len(self.links) == 1:
                    raise
                failures.append((provider, error))
                tried = error
                next_name = live[position + 1][1] if position + 1 < len(live) else None
                if why == "call":
                    self.exhausted.moved(provider, error, next_name)
                else:
                    self.exhausted.mark(provider, error, next_name)
        order = [provider for provider, _, _ in self.links]
        failures.sort(key=lambda failure: order.index(failure[0]))
        raise ChainExhausted(failures) from tried

    def __getattr__(self, name: str) -> Any:
        # Only reached for what Runnable itself lacks: the first model's own attributes.
        # Read from __dict__: on an instance not yet initialised (copy, pickle)
        # `self.links` would re-enter here without end.
        links = self.__dict__.get("links")
        if not links:
            raise AttributeError(name)
        attr = getattr(links[0][2], name)
        if name not in DERIVING:
            return attr

        def every(*args: Any, **kwargs: Any) -> ProviderChain:
            return ProviderChain([(provider, label, getattr(model, name)(*args, **kwargs))
                                  for provider, label, model in links], self.exhausted, self.classify)

        return every
