#!/usr/bin/env python

"""Inspect and optimize USB camera stability settings for Songling workflows.

This helper focuses on one common source of intermittent camera dropouts on Linux:
USB autosuspend and hub power-management transitions.

It can:
- inspect current power settings for configured cameras and their parent USB hubs
- optionally apply runtime optimization (`power/control=on`) to those devices
- optionally write a udev rules snippet for persistent behavior across reboots
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import draccus
except Exception:
    draccus = None

try:
    import yaml
except Exception:
    yaml = None


DEFAULT_CONFIG_PATH = Path("examples/songling_aloha/teleop.yaml")
USB_NODE_RE = re.compile(r"^\d+-\d+(?:\.\d+)*$")
SERIAL_RE = re.compile(r"([A-Z0-9]{8,})")


@dataclass
class UsbPowerEntry:
    usb_node: str
    sysfs_path: Path
    role: str
    id_vendor: str | None
    id_product: str | None
    product: str | None
    serial: str | None
    power_control: str | None
    autosuspend: str | None


@dataclass
class VideoUsbCandidate:
    video_name: str
    usb_node: str
    product: str | None
    manufacturer: str | None
    serial: str | None
    id_vendor: str | None
    id_product: str | None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _resolve_video_device(index_or_path: object) -> str | None:
    if isinstance(index_or_path, int):
        return f"video{index_or_path}"
    if not isinstance(index_or_path, str):
        return None

    raw = index_or_path.strip()
    if not raw:
        return None

    p = Path(raw).expanduser()
    try:
        p = p.resolve(strict=True)
    except FileNotFoundError:
        return None
    except Exception:
        return None

    name = p.name
    if name.startswith("video"):
        return name
    return None


def _extract_serial_hints(index_or_path: object) -> list[str]:
    text = ""
    if isinstance(index_or_path, Path):
        text = str(index_or_path)
    elif isinstance(index_or_path, str):
        text = index_or_path
    if not text:
        return []

    hints: list[str] = []
    for match in SERIAL_RE.findall(text.upper()):
        if match not in hints:
            hints.append(match)
    return hints


def _find_usb_sysfs_node_from_video(video_name: str) -> Path | None:
    device_path = (Path("/sys/class/video4linux") / video_name / "device").resolve()
    if not device_path.exists():
        return None

    current = device_path
    while True:
        name = current.name.split(":", maxsplit=1)[0]
        if USB_NODE_RE.match(name) and (current / "idVendor").exists():
            return current
        if current.parent == current:
            return None
        current = current.parent


def _scan_video_usb_candidates() -> list[VideoUsbCandidate]:
    candidates: list[VideoUsbCandidate] = []
    for video_dir in sorted(Path("/sys/class/video4linux").glob("video*")):
        usb_sysfs = _find_usb_sysfs_node_from_video(video_dir.name)
        if usb_sysfs is None:
            continue
        candidates.append(
            VideoUsbCandidate(
                video_name=video_dir.name,
                usb_node=usb_sysfs.name,
                product=_read_text(usb_sysfs / "product"),
                manufacturer=_read_text(usb_sysfs / "manufacturer"),
                serial=_read_text(usb_sysfs / "serial"),
                id_vendor=_read_text(usb_sysfs / "idVendor"),
                id_product=_read_text(usb_sysfs / "idProduct"),
            )
        )
    return candidates


def _match_candidate_from_config(index_or_path: object, candidates: list[VideoUsbCandidate]) -> VideoUsbCandidate | None:
    video_name = _resolve_video_device(index_or_path)
    if video_name is not None:
        for candidate in candidates:
            if candidate.video_name == video_name:
                return candidate

    serial_hints = _extract_serial_hints(index_or_path)
    if serial_hints:
        for hint in serial_hints:
            for candidate in candidates:
                if candidate.serial and hint in candidate.serial.upper():
                    return candidate

    if isinstance(index_or_path, int):
        fallback_video = f"video{index_or_path}"
        for candidate in candidates:
            if candidate.video_name == fallback_video:
                return candidate
    return None


def _collect_parent_hub_nodes(usb_node_name: str) -> list[str]:
    # Example: "1-2.2.4" -> ["1-2.2", "1-2"]
    parents: list[str] = []
    cur = usb_node_name
    while "." in cur:
        cur = cur.rsplit(".", maxsplit=1)[0]
        parents.append(cur)
    return parents


def _entry_from_usb_node(usb_node_name: str, *, role: str) -> UsbPowerEntry | None:
    sysfs = Path("/sys/bus/usb/devices") / usb_node_name
    if not sysfs.exists():
        return None

    return UsbPowerEntry(
        usb_node=usb_node_name,
        sysfs_path=sysfs,
        role=role,
        id_vendor=_read_text(sysfs / "idVendor"),
        id_product=_read_text(sysfs / "idProduct"),
        product=_read_text(sysfs / "product"),
        serial=_read_text(sysfs / "serial"),
        power_control=_read_text(sysfs / "power/control"),
        autosuspend=_read_text(sysfs / "power/autosuspend"),
    )


def _iter_config_camera_paths(raw_cfg: dict) -> Iterable[tuple[str, object]]:
    robot = raw_cfg.get("robot") if isinstance(raw_cfg.get("robot"), dict) else {}
    for side in ("left", "right"):
        arm_key = f"{side}_arm_config"
        arm = robot.get(arm_key) if isinstance(robot.get(arm_key), dict) else {}
        cameras = arm.get("cameras") if isinstance(arm.get("cameras"), dict) else {}
        for cam_name, cam_cfg in cameras.items():
            if not isinstance(cam_cfg, dict):
                continue
            yield f"{side}_{cam_name}", cam_cfg.get("index_or_path")


def _collect_entries_from_config(raw_cfg: dict) -> list[UsbPowerEntry]:
    by_usb_node: dict[str, UsbPowerEntry] = {}
    candidates = _scan_video_usb_candidates()

    for cam_role, index_or_path in _iter_config_camera_paths(raw_cfg):
        candidate = _match_candidate_from_config(index_or_path, candidates)
        if candidate is None:
            continue
        camera_entry = _entry_from_usb_node(candidate.usb_node, role=f"camera:{cam_role}")
        if camera_entry is not None:
            by_usb_node.setdefault(candidate.usb_node, camera_entry)

        for parent in _collect_parent_hub_nodes(candidate.usb_node):
            hub_entry = _entry_from_usb_node(parent, role="hub:parent")
            if hub_entry is not None:
                by_usb_node.setdefault(parent, hub_entry)

    return sorted(by_usb_node.values(), key=lambda x: x.usb_node)


def _apply_runtime_optimization(entries: list[UsbPowerEntry]) -> tuple[list[str], list[str]]:
    ok: list[str] = []
    failed: list[str] = []

    for entry in entries:
        control_path = entry.sysfs_path / "power/control"
        autosuspend_path = entry.sysfs_path / "power/autosuspend"
        try:
            control_path.write_text("on\n", encoding="utf-8")
            ok.append(f"{entry.usb_node}: power/control=on")
        except Exception as exc:
            failed.append(f"{entry.usb_node}: failed to set power/control=on ({exc})")

        # Best-effort: reduce autosuspend aggressiveness where supported.
        if autosuspend_path.exists():
            try:
                autosuspend_path.write_text("-1\n", encoding="utf-8")
                ok.append(f"{entry.usb_node}: power/autosuspend=-1")
            except Exception:
                # Not all kernels/devices accept -1; keep this as best-effort.
                pass

    return ok, failed


def _write_udev_rules(entries: list[UsbPowerEntry], output_path: Path) -> None:
    rules: list[str] = [
        "# Songling USB camera stability rules",
        "# Generated by optimize_usb_camera_stability.py",
        "",
    ]
    seen_rules: set[str] = set()
    for entry in entries:
        if not entry.id_vendor or not entry.id_product:
            continue
        rule = (
            'ACTION=="add", SUBSYSTEM=="usb", KERNEL=="%s", ATTR{idVendor}=="%s", ATTR{idProduct}=="%s", TEST=="power/control", ATTR{power/control}="on"'
            % (entry.usb_node, entry.id_vendor.lower(), entry.id_product.lower())
        )
        if rule in seen_rules:
            continue
        seen_rules.add(rule)
        rules.append(rule)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rules) + "\n", encoding="utf-8")


def _print_entries(entries: list[UsbPowerEntry]) -> None:
    if not entries:
        print("No USB camera entries were resolved from config.")
        return

    print("Resolved USB power entries:")
    for entry in entries:
        product = entry.product or "unknown"
        vendor_product = (
            f"{entry.id_vendor}:{entry.id_product}" if entry.id_vendor and entry.id_product else "unknown"
        )
        print(
            f"- {entry.usb_node:<8} role={entry.role:<12} "
            f"vid:pid={vendor_product:<12} control={entry.power_control or '--':<5} "
            f"autosuspend={entry.autosuspend or '--':<3} product={product}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect/apply USB camera power stability settings.")
    parser.add_argument(
        "--config-path",
        "--config_path",
        dest="config_path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to Songling teleop yaml.",
    )
    parser.add_argument(
        "--apply-runtime",
        action="store_true",
        help="Apply runtime optimization (power/control=on) to resolved cameras and parent hubs.",
    )
    parser.add_argument(
        "--write-udev-rules",
        dest="write_udev_rules",
        type=Path,
        default=None,
        help="Write a udev rules file to this path for persistent optimization.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.config_path.exists():
        raise FileNotFoundError(f"Config file not found: {args.config_path}")

    with args.config_path.open("r", encoding="utf-8") as f:
        if draccus is not None:
            raw_cfg = draccus.load(dict, f)
        elif yaml is not None:
            raw_cfg = yaml.safe_load(f)
        else:
            raise ModuleNotFoundError(
                "No YAML loader available. Install either `draccus` or `PyYAML`."
            )
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"Config at {args.config_path} is not a valid mapping.")

    entries = _collect_entries_from_config(raw_cfg)
    _print_entries(entries)

    if args.apply_runtime:
        ok, failed = _apply_runtime_optimization(entries)
        print("")
        print("Runtime apply results:")
        for item in ok:
            print(f"  OK    {item}")
        for item in failed:
            print(f"  FAIL  {item}")
        if failed:
            print("")
            print("Tip: write access to /sys usually requires root.")
            print("Try re-running with sudo:")
            print(
                "  sudo "
                + " ".join(
                    [
                        str(Path(__file__).resolve()),
                        f"--config-path={args.config_path}",
                        "--apply-runtime",
                    ]
                )
            )

    if args.write_udev_rules is not None:
        _write_udev_rules(entries, args.write_udev_rules)
        print("")
        print(f"Wrote udev rules to: {args.write_udev_rules}")
        print("To install persistently:")
        print(f"  sudo cp {args.write_udev_rules} /etc/udev/rules.d/")
        print("  sudo udevadm control --reload-rules")
        print("  sudo udevadm trigger")


if __name__ == "__main__":
    main()
