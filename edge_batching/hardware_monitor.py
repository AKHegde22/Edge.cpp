"""
Cross-platform hardware monitor for thermal pressure and battery level.

Supported platforms:
  - macOS  : sysctl kern.thermal_pressure + pmset -g batt
  - Linux  : /sys/class/thermal + /sys/class/power_supply
  - Windows: WMI (via wmic / PowerShell)

On unsupported platforms or when a sensor read fails, the monitor falls
back to safe defaults (nominal thermal, 100% battery, AC power) so the
scheduler keeps running without any throttling.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

_SYSTEM = platform.system()  # "Darwin", "Linux", "Windows"


# ---------------------------------------------------------------------------
# macOS backend
# ---------------------------------------------------------------------------

def _macos_thermal_pressure() -> str:
    """Read thermal pressure via sysctl (macOS 12+)."""
    try:
        output = subprocess.check_output(
            ["sysctl", "-n", "kern.thermal_pressure"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        mapping = {"0": "nominal", "1": "fair", "2": "serious", "3": "critical"}
        return mapping.get(output, "nominal")
    except Exception:
        return "nominal"


def _macos_battery_level() -> float:
    """Read battery percentage from pmset."""
    try:
        output = subprocess.check_output(
            ["pmset", "-g", "batt"],
            stderr=subprocess.DEVNULL,
        ).decode()
        match = re.search(r"(\d+)%", output)
        if match:
            return float(match.group(1))
        return 100.0
    except Exception:
        return 100.0


def _macos_is_plugged_in() -> bool:
    """Check if the Mac is on AC power."""
    try:
        output = subprocess.check_output(
            ["pmset", "-g", "batt"],
            stderr=subprocess.DEVNULL,
        ).decode()
        return "AC Power" in output
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Linux backend
# ---------------------------------------------------------------------------

_THERMAL_ZONE_BASE = Path("/sys/class/thermal")
_POWER_SUPPLY_BASE = Path("/sys/class/power_supply")


def _linux_thermal_pressure() -> str:
    """
    Read the hottest thermal zone temperature and map it to a pressure label.

    Thresholds (°C → label):
      < 65  → nominal
      < 80  → fair
      < 95  → serious
      ≥ 95  → critical
    """
    try:
        max_temp = 0
        for zone in _THERMAL_ZONE_BASE.iterdir():
            temp_file = zone / "temp"
            if temp_file.exists():
                raw = temp_file.read_text().strip()
                # Value is in millidegrees Celsius
                temp_c = int(raw) / 1000.0
                max_temp = max(max_temp, temp_c)

        if max_temp == 0:
            return "nominal"
        if max_temp < 65:
            return "nominal"
        if max_temp < 80:
            return "fair"
        if max_temp < 95:
            return "serious"
        return "critical"
    except Exception:
        return "nominal"


def _linux_battery_level() -> float:
    """Read battery percentage from /sys/class/power_supply."""
    try:
        for supply in _POWER_SUPPLY_BASE.iterdir():
            type_file = supply / "type"
            if type_file.exists() and type_file.read_text().strip() == "Battery":
                capacity_file = supply / "capacity"
                if capacity_file.exists():
                    return float(capacity_file.read_text().strip())
        # No battery found — desktop machine, treat as fully charged
        return 100.0
    except Exception:
        return 100.0


def _linux_is_plugged_in() -> bool:
    """Check if AC power is connected."""
    try:
        for supply in _POWER_SUPPLY_BASE.iterdir():
            type_file = supply / "type"
            if type_file.exists() and type_file.read_text().strip() == "Mains":
                online_file = supply / "online"
                if online_file.exists():
                    return online_file.read_text().strip() == "1"
        # No mains supply found — assume plugged in (desktop)
        return True
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Windows backend
# ---------------------------------------------------------------------------

def _windows_thermal_pressure() -> str:
    """
    Read CPU temperature via WMI.  Falls back to nominal if the sensor
    is unavailable (common on non-OEM Windows installs).

    Same threshold mapping as Linux.
    """
    try:
        # MSAcpi_ThermalZoneTemperature is the standard WMI class
        output = subprocess.check_output(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance MSAcpi_ThermalZoneTemperature "
                "-Namespace root/wmi 2>$null | "
                "Select-Object -ExpandProperty CurrentTemperature",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()

        if not output:
            return "nominal"

        # WMI returns temperature in tenths of Kelvin
        max_temp = 0.0
        for line in output.splitlines():
            line = line.strip()
            if line:
                temp_c = (float(line) / 10.0) - 273.15
                max_temp = max(max_temp, temp_c)

        if max_temp <= 0:
            return "nominal"
        if max_temp < 65:
            return "nominal"
        if max_temp < 80:
            return "fair"
        if max_temp < 95:
            return "serious"
        return "critical"
    except Exception:
        return "nominal"


def _windows_battery_level() -> float:
    """Read battery percentage via WMIC."""
    try:
        output = subprocess.check_output(
            [
                "powershell", "-NoProfile", "-Command",
                "(Get-CimInstance Win32_Battery).EstimatedChargeRemaining",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()

        if output:
            return float(output.splitlines()[0].strip())
        # No battery — desktop, report fully charged
        return 100.0
    except Exception:
        return 100.0


def _windows_is_plugged_in() -> bool:
    """Check AC power status via WMI."""
    try:
        output = subprocess.check_output(
            [
                "powershell", "-NoProfile", "-Command",
                "(Get-CimInstance Win32_Battery).BatteryStatus",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()

        if output:
            # BatteryStatus: 2 = AC, 1 = discharging
            return output.splitlines()[0].strip() == "2"
        return True
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Platform dispatcher
# ---------------------------------------------------------------------------

def _get_thermal_pressure() -> str:
    if _SYSTEM == "Darwin":
        return _macos_thermal_pressure()
    if _SYSTEM == "Linux":
        return _linux_thermal_pressure()
    if _SYSTEM == "Windows":
        return _windows_thermal_pressure()
    return "nominal"


def _get_battery_level() -> float:
    if _SYSTEM == "Darwin":
        return _macos_battery_level()
    if _SYSTEM == "Linux":
        return _linux_battery_level()
    if _SYSTEM == "Windows":
        return _windows_battery_level()
    return 100.0


def _is_plugged_in() -> bool:
    if _SYSTEM == "Darwin":
        return _macos_is_plugged_in()
    if _SYSTEM == "Linux":
        return _linux_is_plugged_in()
    if _SYSTEM == "Windows":
        return _windows_is_plugged_in()
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class HardwareMonitor:
    """
    Cross-platform hardware health monitor.

    Periodically reads thermal pressure and battery level and pushes
    updates to a callback.  Works on macOS, Linux, and Windows.

    Callback signature:
        callback(thermal: str, battery: float) -> None

    Where thermal is one of: "nominal", "fair", "serious", "critical"
    and battery is a percentage (0.0 – 100.0).
    """

    def __init__(
        self,
        callback: Callable[[str, float], None],
        interval_s: float = 2.0,
    ) -> None:
        self.callback = callback
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def platform() -> str:
        """Return the detected OS name."""
        return _SYSTEM

    @staticmethod
    def read_thermal_pressure() -> str:
        """One-shot read of thermal pressure."""
        return _get_thermal_pressure()

    @staticmethod
    def read_battery_level() -> float:
        """One-shot read of battery percentage."""
        return _get_battery_level()

    @staticmethod
    def read_plugged_in() -> bool:
        """One-shot check of AC power status."""
        return _is_plugged_in()

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s + 1.0)

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            thermal = _get_thermal_pressure()
            battery = _get_battery_level()
            self.callback(thermal, battery)
            self._stop_event.wait(timeout=self.interval_s)


# Backwards-compatible alias
MacOSHardwareMonitor = HardwareMonitor
