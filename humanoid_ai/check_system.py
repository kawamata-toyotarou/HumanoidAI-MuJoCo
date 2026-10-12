from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import winreg
from dataclasses import asdict, dataclass


@dataclass
class PackageStatus:
    name: str
    installed: bool
    version: str | None
    location: str | None


def _package_status(name: str) -> PackageStatus:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return PackageStatus(name=name, installed=False, version=None, location=None)

    version = None
    try:
        module = __import__(name)
        version = getattr(module, "__version__", None)
    except Exception as exc:  # pragma: no cover - diagnostic path
        version = f"import failed: {exc}"

    return PackageStatus(name=name, installed=True, version=version, location=spec.origin)


def _cpu_name() -> str | None:
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value)
    except OSError:
        return None


def _memory_status() -> dict[str, float | int | None]:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    if not ok:
        return {"total_gb": None, "available_gb": None, "load_percent": None}

    return {
        "total_gb": round(status.ullTotalPhys / (1024**3), 2),
        "available_gb": round(status.ullAvailPhys / (1024**3), 2),
        "load_percent": int(status.dwMemoryLoad),
    }


def _run_text(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return None

    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    return output.strip() or None


def _gpu_info() -> dict[str, object]:
    pnputil = _run_text(["pnputil", "/enum-devices", "/class", "Display"])
    nvidia_smi = shutil.which("nvidia-smi")
    nvidia_output = _run_text([nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]) if nvidia_smi else None
    return {
        "display_devices_raw": pnputil,
        "nvidia_smi_path": nvidia_smi,
        "nvidia_smi_output": nvidia_output,
    }


def build_report() -> dict[str, object]:
    packages = [
        _package_status(name)
        for name in (
            "numpy",
            "mujoco",
            "gymnasium",
            "torch",
            "stable_baselines3",
            "tensorboard",
        )
    ]
    return {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "cpu": {
            "name": _cpu_name() or platform.processor() or None,
            "logical_count": os.cpu_count(),
        },
        "ram": _memory_status(),
        "gpu": _gpu_info(),
        "packages": [asdict(package) for package in packages],
    }


def main() -> None:
    print(json.dumps(build_report(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
