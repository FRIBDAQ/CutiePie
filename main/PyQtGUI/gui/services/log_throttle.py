"""Rate limiter for log calls made from a hot path.

Mouse-motion and render-tick handlers can raise thousands of times a second,
so a handler that logs an unexpected error unconditionally trades a silent
failure for an unusable log. `LogThrottle` lets the first occurrence through,
swallows the rest for a while, and then reports how many it swallowed, so the
message stays cheap without hiding how often the thing happens.

Qt-free and clock-injectable, so the behaviour is testable headlessly.
"""

import time


class LogThrottle:
    """Allow one log call per interval, counting the ones held back.

    >>> t = LogThrottle(interval_secs=30.0)
    >>> t.allow()
    (True, 0)
    """

    def __init__(self, interval_secs=30.0, clock=time.monotonic):
        self._interval = float(interval_secs)
        self._clock = clock
        self._last_emit = None
        self._suppressed = 0

    def allow(self):
        """Return `(should_log, suppressed_since_last_log)`.

        The count is the number of calls held back since the last permitted
        one, so a caller can say "42 similar suppressed" and reset it. It is
        always 0 when `should_log` is False.
        """
        now = self._clock()
        if self._last_emit is not None and (now - self._last_emit) < self._interval:
            self._suppressed += 1
            return False, 0
        suppressed = self._suppressed
        self._last_emit = now
        self._suppressed = 0
        return True, suppressed

    def reset(self):
        """Forget the interval and the count — the next `allow()` returns True."""
        self._last_emit = None
        self._suppressed = 0
