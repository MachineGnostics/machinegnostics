"""Device abstraction helpers for Magnet.

Author: Nirmal Parmar

Notes:
- Current backend supports CPU execution.
- Interface is intentionally future-proof for GPU integration.
"""

from __future__ import annotations


class Device:
    """Represents a target compute device."""

    SUPPORTED = {"cpu"}

    def __init__(self, name: str = "cpu"):
        canonical = name.lower().strip()
        if canonical not in self.SUPPORTED:
            raise ValueError(f"Unsupported device '{name}'. Supported: {sorted(self.SUPPORTED)}")
        self.name = canonical

    def __repr__(self) -> str:
        return f"Device('{self.name}')"


def get_default_device() -> Device:
    """Return default device instance."""
    return Device("cpu")


def is_gpu_available() -> bool:
    """Indicates whether GPU backend is available."""
    return False


__all__ = ["Device", "get_default_device", "is_gpu_available"]