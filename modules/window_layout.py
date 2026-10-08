"""Shared desktop bounds for the GUI and its Chromium window."""

import json
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class WindowBounds:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self):
        if self.width < 1 or self.height < 1:
            raise ValueError("Window bounds must have positive width and height.")

    def split(self) -> tuple["WindowBounds", "WindowBounds"]:
        if self.width < 2:
            raise ValueError("The desktop must be at least two pixels wide to split.")
        left_width = self.width // 2
        return (
            WindowBounds(self.x, self.y, left_width, self.height),
            WindowBounds(self.x + left_width, self.y, self.width - left_width, self.height),
        )


_MACOS_DESKTOP_SCRIPT = """
ObjC.import("AppKit");
const screen = $.NSScreen.screens.objectAtIndex(0);
const frame = screen.frame;
const visible = screen.visibleFrame;
JSON.stringify({
    x: Math.round(visible.origin.x),
    y: Math.round(frame.size.height - visible.origin.y - visible.size.height),
    width: Math.round(visible.size.width),
    height: Math.round(visible.size.height)
});
"""


def desktop_bounds(screen_width: int, screen_height: int) -> WindowBounds:
    """Use the primary desktop; on macOS exclude the menu bar and Dock."""
    if sys.platform != "darwin":
        return WindowBounds(0, 0, screen_width, screen_height)
    try:
        result = subprocess.run(
            ["osascript", "-l", "JavaScript", "-e", _MACOS_DESKTOP_SCRIPT],
            capture_output=True, text=True, check=True, timeout=5,
        )
        values = json.loads(result.stdout)
        if not isinstance(values, dict) or any(
            type(values.get(key)) is not int for key in ("x", "y", "width", "height")
        ):
            raise ValueError("Invalid desktop dimensions returned by macOS.")
        return WindowBounds(**values)
    except (OSError, subprocess.SubprocessError, ValueError, TypeError) as error:
        raise RuntimeError(f"Unable to determine the macOS usable desktop area: {error}") from error
