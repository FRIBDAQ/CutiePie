"""LogThrottle — the rate limiter histoHover uses for unexpected errors.

The point of the class is that a hot handler can report a real defect without
flooding the log. These tests pin both halves of that: the first occurrence
always gets through, and the ones held back are counted rather than lost.

A fake clock drives every timing case; nothing here sleeps.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

from services.log_throttle import LogThrottle


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_first_call_is_always_allowed():
    assert LogThrottle(interval_secs=30.0, clock=FakeClock()).allow() == (True, 0)


def test_second_call_inside_the_interval_is_suppressed():
    clk = FakeClock()
    t = LogThrottle(interval_secs=30.0, clock=clk)
    t.allow()
    clk.advance(29.9)
    assert t.allow() == (False, 0)


def test_call_after_the_interval_is_allowed_again():
    clk = FakeClock()
    t = LogThrottle(interval_secs=30.0, clock=clk)
    t.allow()
    clk.advance(30.1)
    assert t.allow()[0] is True


def test_suppressed_calls_are_counted_and_reported_once():
    clk = FakeClock()
    t = LogThrottle(interval_secs=30.0, clock=clk)
    t.allow()
    for _ in range(41):
        assert t.allow() == (False, 0)
    clk.advance(31.0)
    assert t.allow() == (True, 41)


def test_the_count_resets_after_it_is_reported():
    clk = FakeClock()
    t = LogThrottle(interval_secs=30.0, clock=clk)
    t.allow()
    t.allow(); t.allow()
    clk.advance(31.0)
    t.allow()                      # reports 2
    clk.advance(31.0)
    assert t.allow() == (True, 0)  # and does not report them twice


def test_a_hover_storm_yields_one_log_per_interval():
    """5,000 failures over 100 s of hovering must not be 5,000 log lines."""
    clk = FakeClock()
    t = LogThrottle(interval_secs=30.0, clock=clk)
    emitted = 0
    for i in range(5000):
        clk.advance(0.02)          # 50 mouse-motion events a second
        if t.allow()[0]:
            emitted += 1
    assert emitted == 4            # t=0, then once per 30 s across 100 s


def test_reset_lets_the_next_call_through():
    clk = FakeClock()
    t = LogThrottle(interval_secs=30.0, clock=clk)
    t.allow()
    t.allow()
    t.reset()
    assert t.allow() == (True, 0)


def test_a_zero_interval_never_suppresses():
    clk = FakeClock()
    t = LogThrottle(interval_secs=0.0, clock=clk)
    assert t.allow() == (True, 0)
    assert t.allow() == (True, 0)


def test_default_clock_is_monotonic_and_usable():
    """No injected clock: the first call still passes, the immediate second does not."""
    t = LogThrottle(interval_secs=30.0)
    assert t.allow()[0] is True
    assert t.allow()[0] is False
