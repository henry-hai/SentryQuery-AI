"""Shared cost controls: a bounded answer cache and a per-client rate limiter.

Every uncached run of the pipeline costs a paid OpenAI embedding plus a Pinecone
query, and both of the long-running interfaces over that pipeline (the MCP
server on stdio and the HTTP API) are meant to be left up. So both put the same
two ceilings in front of the pipeline, in the same order: the cache first, so a
repeat costs nothing and spends no rate-limit budget, then the limiter, and only
then a paid run.

These live here rather than inside either interface so neither one has to import
the other to reach them. The mechanism is shared. The tuning is not: each
interface reads its own environment variable and passes the value in, because a
sensible ceiling for one client on a local stdio pipe is not a sensible ceiling
for a public HTTP endpoint.

Both are in-process only. They reset when the process restarts and are not
shared between processes. There is no Redis, no external cache, and no
distributed quota.
"""
import os
import threading
import time
from collections import OrderedDict, deque
from copy import deepcopy
from typing import Any, Optional

# Neutral fallbacks, used when an interface passes nothing. Each interface has
# its own environment variable and overrides these explicitly.
DEFAULT_CACHE_MAX_ENTRIES = 128
DEFAULT_RATE_LIMIT_PER_MINUTE = 10


def env_int(name: str, default: int) -> int:
    """Read a positive integer setting from the environment, falling back to
    `default` when it is unset, blank, or not a usable number."""
    try:
        value = int(os.getenv(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


class RateLimitExceeded(RuntimeError):
    """Raised instead of firing a paid call when a client is over its limit."""


def normalize_key(text: str) -> str:
    """Cache key for a piece of query text: trimmed, lowercased, whitespace
    collapsed.

    So "  What were TOTAL   net sales? " and "what were total net sales?" hit
    one cache entry instead of costing two paid runs.
    """
    return " ".join(text.split()).lower()


class AnswerCache:
    """A bounded, thread-safe cache of normalized key -> response.

    Insertion-ordered, so at the cap the oldest entry is evicted first. A hit
    does not refresh an entry's position: entries age out by when they were
    first stored, not by when they were last read. Values are deep-copied in and
    out, so a caller holding a returned response cannot mutate the cached copy
    (a shallow copy would still share the sources list).
    """

    def __init__(self, max_entries: int = DEFAULT_CACHE_MAX_ENTRIES):
        self.max_entries = max_entries
        self._entries: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[dict[str, Any]]:
        with self._lock:
            hit = self._entries.get(key)
            return deepcopy(hit) if hit is not None else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._entries[key] = deepcopy(value)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class RateLimiter:
    """Per-client sliding-window limiter over a rolling 60-second window.

    Each client key keeps the timestamps of its recent admitted calls. A call is
    admitted only if fewer than `max_per_minute` of them still fall inside the
    window, so an over-limit client is turned away before any paid call starts.

    `call_label` and `override_env` only shape the rejection message, so each
    interface can name the call the caller actually made and the variable that
    raises its own ceiling.
    """

    WINDOW_SECONDS = 60.0

    def __init__(
        self,
        max_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE,
        call_label: str = "calls",
        override_env: Optional[str] = None,
    ):
        self.max_per_minute = max_per_minute
        self.call_label = call_label
        self.override_env = override_env
        self._calls: dict[str, deque] = {}
        self._lock = threading.Lock()

    def _message(self, retry_in: float) -> str:
        raise_it = f", or raise {self.override_env}" if self.override_env else ""
        return (
            f"Rate limit reached: {self.max_per_minute} {self.call_label} per "
            f"minute per client. Retry in {retry_in:.0f}s{raise_it}. "
            f"No paid model or Pinecone call was made."
        )

    def check(self, client_id: str, now: Optional[float] = None) -> None:
        """Record one call for `client_id`, or raise RateLimitExceeded.

        `now` is injectable so tests can drive the window without sleeping.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            recent = self._calls.setdefault(client_id, deque())
            while recent and now - recent[0] >= self.WINDOW_SECONDS:
                recent.popleft()
            if len(recent) >= self.max_per_minute:
                raise RateLimitExceeded(
                    self._message(self.WINDOW_SECONDS - (now - recent[0]))
                )
            recent.append(now)

    def retry_after(self, client_id: str, now: Optional[float] = None) -> int:
        """Whole seconds until `client_id` has budget again, for a Retry-After
        header. Returns 0 when the client is not currently at its limit."""
        now = time.monotonic() if now is None else now
        with self._lock:
            recent = self._calls.get(client_id)
            if not recent or len(recent) < self.max_per_minute:
                return 0
            return max(1, int(self.WINDOW_SECONDS - (now - recent[0]) + 0.999))

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()
