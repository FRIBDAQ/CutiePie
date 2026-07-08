"""Figure-overlay placement math — the Qt-free core of the figure-overlay cluster.

The imaging popup can overlay a loaded image (e.g. a LISE++ picture) on top of
a spectrum pad and nudge it around. The placement is expressed in figure-relative
axes coordinates ([0,1]); the arithmetic that turns a pad's grid position into a
start corner, and that applies joystick / fine-nudge moves, is pure and lives
here. The MainWindow keeps the Qt/matplotlib shell — reading the sliders/joystick
widgets, the cv2 image load, and the ``plt.axes``/``imshow`` drawing.

Qt-free by construction: plain numbers in, plain numbers out, no imports. Extracted
from ``GUI.py:indexToStartPosition`` / ``moveFigure`` / the four ``fineXMove``
helpers so the offset/flip logic is unit-testable without PyQt5 (mirrors
``geometry_io`` / ``dataframe_export`` / ``peak_finder``).
"""

# Nudge step sizes (figure-relative axes units), matching the original methods.
_FINE_STEP = 0.002        # the four fineXMove buttons
_JOYSTICK_STEP = 0.03     # scaled by the joystick's reported distance


def compute_overlay_position(row, col, i, j):
    """Map a pad's grid position to the overlay's start corner (xstart, ystart).

    ``row``/``col`` are the tab's grid dimensions and ``i``/``j`` the pad's row/
    column index (from ``plotPosition``). Returns the ``(xstart, ystart)`` pair
    exactly as ``indexToStartPosition`` assigned them — including the ``-0.1`` x
    nudge, the ``+0.1`` y nudge, and the ``1 - y`` vertical flip (mpl axes origin
    is bottom-left, the grid counts from the top). Moved verbatim.
    """
    xoffs = float(1 / (2 * col))
    yoffs = float(1 / (2 * row))
    xstart = xoffs * (2 * j + 1) - 0.1
    ystart = yoffs * (2 * i + 1) + 0.1
    return xstart, 1 - ystart


def apply_joystick_move(xstart, ystart, direction, distance):
    """Apply a joystick move to the overlay start corner.

    ``direction`` is one of ``"up"``/``"down"``/``"left"``/anything-else (treated
    as right, matching the original ``else`` branch); the step is scaled by
    ``distance``. Returns the new ``(xstart, ystart)``. Moved verbatim from
    ``moveFigure``.
    """
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

    ``direction`` is one of ``"up"``/``"down"``/``"left"``/``"right"``. Returns
    the new ``(xstart, ystart)``. Collapses the four ``fineXMove`` helpers, whose
    only difference was which coordinate moved by ``+/- _FINE_STEP``.
    """
    if direction == "up":
        ystart += _FINE_STEP
    elif direction == "down":
        ystart -= _FINE_STEP
    elif direction == "left":
        xstart -= _FINE_STEP
    elif direction == "right":
        xstart += _FINE_STEP
    return xstart, ystart
