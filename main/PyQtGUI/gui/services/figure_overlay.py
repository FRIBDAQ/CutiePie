"""Placement math for overlaying an image on a spectrum pad, Qt-free so it can
be tested without a figure. The imaging popup uses it to position and nudge a
loaded picture."""

# Nudge step sizes (figure-relative axes units), matching the original methods.
_FINE_STEP = 0.002        # the four fineXMove buttons
_JOYSTICK_STEP = 0.03     # scaled by the joystick's reported distance


def compute_overlay_position(row, col, i, j):
    """Map a pad's grid position to the overlay's start corner (xstart,
    ystart). ``row``/``col`` are the tab's grid dimensions and ``i``/``j`` the
    pad's row/ column index (from ``plotPosition``)."""
    xoffs = float(1 / (2 * col))
    yoffs = float(1 / (2 * row))
    xstart = xoffs * (2 * j + 1) - 0.1
    ystart = yoffs * (2 * i + 1) + 0.1
    return xstart, 1 - ystart


def apply_joystick_move(xstart, ystart, direction, distance):
    """Apply a joystick move to the overlay start corner. ``direction`` is one
    of ``"up"``/``"down"``/``"left"``/anything-else (treated as right,
    matching the original ``else`` branch); the step is scaled by
    ``distance``."""
    step = distance * _JOYSTICK_STEP
    if direction == "up":
        ystart += step
    elif direction == "down":
        ystart -= step
    elif direction == "left":
        xstart -= step
    else:
        xstart += step
    return xstart, ystart


def apply_fine_move(xstart, ystart, direction):
    """Apply a single fine-nudge step to the overlay start corner.
    ``direction`` is one of ``"up"``/``"down"``/``"left"``/``"right"``."""
    if direction == "up":
        ystart += _FINE_STEP
    elif direction == "down":
        ystart -= _FINE_STEP
    elif direction == "left":
        xstart -= _FINE_STEP
    elif direction == "right":
        xstart += _FINE_STEP
    return xstart, ystart
