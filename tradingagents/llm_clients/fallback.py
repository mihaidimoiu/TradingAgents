"""A chat model that hands its calls to another provider's once its own has no credit left."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig

logger = logging.getLogger(__name__)

# How each provider says the account is empty. Matched in the error's text: the
# SDKs raise it as an ordinary rate limit (OpenAI's 429) or bad request
# (Anthropic's 400), and those must not switch providers.
NO_CREDIT = (
    "insufficient_quota",
    "exceeded your current quota",
    "credit balance is too low",
    "credit_balance",
    "no credits remaining",
)

# Methods that build a new model from this one; the backup must be built the same way.
DERIVING = frozenset({"bind_tools", "with_structured_output"})


def out_of_credit(error: BaseException) -> bool:
    text = str(error).lower()
    return any(sign in text for sign in NO_CREDIT)


class Exhausted:
    """The providers found out of credit during one run, shared by all its clients.

    Once a provider is marked, its clients go straight to their backups: every
    call would fail the same way, after the SDK's own retries.
    """

    def __init__(self, notify: Callable[[str], None] | None = None):
        self.providers: set[str] = set()
        self._notify = notify
        self._lock = threading.Lock()

    def mark(self, provider: str, backup: str) -> None:
        with self._lock:
            if provider in self.providers:
                return
            self.providers.add(provider)
        message = f"{provider} has no credit left; its roles answer on {backup}"
        logger.warning(message)
        if self._notify is not None:
            self._notify(message)


class CreditFallback(Runnable):
    """`primary`, or `backup` once `provider` is out of credit; any other error still raises."""

    def __init__(self, primary: Runnable, backup: Runnable, provider: str, backup_name: str,
                 exhausted: Exhausted):
        self.primary, self.backup = primary, backup
        self.provider, self.backup_name = provider, backup_name
        self.exhausted = exhausted

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        if self.provider not in self.exhausted.providers:
            try:
                return self.primary.invoke(input, config, **kwargs)
            except Exception as error:
                if not out_of_credit(error):
                    raise
                self.exhausted.mark(self.provider, self.backup_name)
        return self.backup.invoke(input, config, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Only reached for what Runnable itself lacks: the model's own attributes.
        # Read from __dict__: on an instance not yet initialised (copy, pickle)
        # `self.primary` would re-enter here without end.
        primary = self.__dict__.get("primary")
        if primary is None:
            raise AttributeError(name)
        attr = getattr(primary, name)
        if name not in DERIVING:
            return attr

        def both(*args: Any, **kwargs: Any) -> CreditFallback:
            return CreditFallback(attr(*args, **kwargs), getattr(self.backup, name)(*args, **kwargs),
                                  self.provider, self.backup_name, self.exhausted)

        return both
