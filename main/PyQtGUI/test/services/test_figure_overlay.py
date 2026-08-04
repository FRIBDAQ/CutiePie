"""Headless net for the figure-overlay cluster extracted from GUI.py. Pins the
overlay placement arithmetic that was previously buried in
MainWindow.indexToStartPosition / moveFigure / the four fineXMove helpers: the
grid-position → start-corner math (with its +/-0.1 nudges and the 1-y flip),
the joystick direction/distance step, and the fine-nudge step."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

import pytest

from services.figure_overlay import (compute_overlay_position,
                                     apply_joystick_move, apply_fine_move)


# ------------------------------------------------------------ compute_overlay_position

def test_single_pad_grid_center():
    # 1x1 grid, pad (0,0): xoffs=yoffs=0.5
    #   xstart = 0.5*1 - 0.1 = 0.4
    #   ystart = 0.5*1 + 0.1 = 0.6 -> flipped -> 1-0.6 = 0.4
    x, y = compute_overlay_position(1, 1, 0, 0)
    assert x == pytest.approx(0.4)
    assert y == pytest.approx(0.4)


def test_2x2_grid_pad_indices():
    # 2x2 grid: xoffs=yoffs=0.25
    # pad (0,0): x = 0.25*1-0.1 = 0.15 ; y = 1-(0.25*1+0.1)=1-0.35=0.65
    x, y = compute_overlay_position(2, 2, 0, 0)
    assert x == pytest.approx(0.15)
    assert y == pytest.approx(0.65)
    # pad (1,1): x = 0.25*3-0.1 = 0.65 ; y = 1-(0.25*3+0.1)=1-0.85=0.15
    x, y = compute_overlay_position(2, 2, 1, 1)
    assert x == pytest.approx(0.65)
    assert y == pytest.approx(0.15)


def test_vertical_flip_is_applied():
    # top row (i=0) should land higher on the figure than bottom row (i=1)
    _, y_top = compute_overlay_position(2, 2, 0, 0)
    _, y_bottom = compute_overlay_position(2, 2, 1, 0)
    assert y_top > y_bottom


def test_non_square_grid():
    # 2 rows x 3 cols: xoffs=1/6, yoffs=0.25
    # pad (1,2): x = (1/6)*5-0.1 ; y = 1-(0.25*3+0.1)
    x, y = compute_overlay_position(2, 3, 1, 2)
    assert x == pytest.approx((1/6)*5 - 0.1)
    assert y == pytest.approx(1 - (0.25*3 + 0.1))


# ---------------------------------------------------------------- apply_joystick_move

@pytest.mark.parametrize("direction,dx,dy", [
    ("up", 0.0, +1.0),
    ("down", 0.0, -1.0),
    ("left", -1.0, 0.0),
    ("right", +1.0, 0.0),   # the else branch
])
def test_joystick_directions(direction, dx, dy):
    x0, y0, dist = 0.5, 0.5, 2.0
    step = dist * 0.03
    x, y = apply_joystick_move(x0, y0, direction, dist)
    assert x == pytest.approx(x0 + dx * step)
    assert y == pytest.approx(y0 + dy * step)


def test_joystick_unknown_direction_treated_as_right():
    x, y = apply_joystick_move(0.5, 0.5, "diagonal", 1.0)
    assert x == pytest.approx(0.5 + 0.03)
    assert y == pytest.approx(0.5)


# ------------------------------------------------------------------- apply_fine_move

@pytest.mark.parametrize("direction,dx,dy", [
    ("up", 0.0, +0.002),
    ("down", 0.0, -0.002),
    ("left", -0.002, 0.0),
    ("right", +0.002, 0.0),
])
def test_fine_move_directions(direction, dx, dy):
    x, y = apply_fine_move(0.5, 0.5, direction)
    assert x == pytest.approx(0.5 + dx)
    assert y == pytest.approx(0.5 + dy)


def test_fine_move_accumulates():
    x, y = 0.5, 0.5
    for _ in range(3):
        x, y = apply_fine_move(x, y, "up")
    assert y == pytest.approx(0.5 + 3 * 0.002)
