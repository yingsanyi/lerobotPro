# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helper to find the USB port associated with your MotorsBus.

Example:

```shell
lerobot-find-port
```

Find CAN network interfaces by unplugging a USB-CAN adapter:

```shell
lerobot-find-port --kind can --can-mode interface --interfaces can0,can1
```

Find which CAN interface corresponds to an integrated chain by observing traffic drop
after unplugging:

```shell
lerobot-find-port --kind can --can-mode traffic --interfaces can0,can1
```
"""

import argparse
import platform
import re
import time
from pathlib import Path


def _parse_csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [x.strip() for x in value.split(",") if x.strip()]
    return items or None


def find_available_serial_ports() -> list[str]:
    """List likely serial ports across platforms (uses pyserial when available)."""
    try:
        from serial.tools import list_ports  # Part of pyserial library
    except Exception:  # nosec B110
        list_ports = None

    if list_ports is not None:
        return sorted([port.device for port in list_ports.comports()])

    # Fallback: keep this list reasonably small for diff-based detection.
    if platform.system() == "Windows":
        return []
    ports = []
    for pat in ("ttyUSB*", "ttyACM*", "ttyAMA*", "ttyS*"):
        ports.extend([str(p) for p in Path("/dev").glob(pat)])
    return sorted(set(ports))


def _is_can_iface(iface: str) -> bool:
    """Best-effort check for CAN interface type via sysfs."""
    type_path = Path("/sys/class/net") / iface / "type"
    try:
        # ARPHRD_CAN == 280
        return type_path.read_text(encoding="utf-8").strip() == "280"
    except Exception:  # nosec B110
        # If we can't read sysfs (non-Linux), fall back to naming heuristics.
        return False


def find_available_can_ports(
    *,
    interfaces: list[str] | None = None,
    prefixes: list[str] | None = None,
    pattern: str | None = None,
) -> list[str]:
    """List CAN network interfaces on Linux using sysfs."""
    if platform.system() != "Linux":
        raise OSError("CAN interface discovery is only supported on Linux.")

    sysfs = Path("/sys/class/net")
    if not sysfs.exists():
        raise OSError("Missing /sys/class/net. Cannot list network interfaces.")

    all_ifaces = sorted([p.name for p in sysfs.iterdir() if p.is_dir()])

    if interfaces is not None:
        candidates = [i for i in all_ifaces if i in set(interfaces)]
    else:
        candidates = list(all_ifaces)

    if pattern is not None:
        rx = re.compile(pattern)
        candidates = [i for i in candidates if rx.search(i)]
    else:
        prefixes = prefixes or ["can", "slcan"]
        candidates = [i for i in candidates if any(i.startswith(pref) for pref in prefixes)]

    # Prefer sysfs type filtering when available; keep name-based fallbacks.
    can_ifaces = [i for i in candidates if _is_can_iface(i)]
    if can_ifaces:
        return can_ifaces
    return candidates


def find_available_ports(
    *,
    kind: str,
    interfaces: list[str] | None = None,
    prefixes: list[str] | None = None,
    pattern: str | None = None,
) -> list[str]:
    if kind == "serial":
        return find_available_serial_ports()
    if kind == "can":
        return find_available_can_ports(interfaces=interfaces, prefixes=prefixes, pattern=pattern)
    raise ValueError(f"Unknown kind={kind!r}")


def find_port(
    *,
    kind: str = "serial",
    can_mode: str = "auto",
    interfaces: list[str] | None = None,
    prefixes: list[str] | None = None,
    pattern: str | None = None,
    settle_time_s: float = 0.8,
    wait_timeout_s: float | None = None,
    poll_interval_s: float = 0.2,
    bitrate: int | None = None,
    use_fd: bool = False,
    data_bitrate: int | None = None,
    traffic_window_s: float = 2.0,
    min_msgs: int = 1,
    traffic_direction: str = "rx",
) -> str:
    print(f"Finding all available ports for the MotorsBus (kind={kind}, can_mode={can_mode}).")
    ports_before = find_available_ports(kind=kind, interfaces=interfaces, prefixes=prefixes, pattern=pattern)
    print("Ports before disconnecting:", ports_before)

    def _read_net_counter(iface: str, counter: str) -> int | None:
        p = Path("/sys/class/net") / iface / "statistics" / counter
        try:
            return int(p.read_text(encoding="utf-8").strip())
        except Exception:  # nosec B110
            return None

    def _measure_net_deltas(ifaces: list[str], duration_s: float, direction: str) -> dict[str, int]:
        """Measure traffic deltas via sysfs counters (preferred; no sockets needed)."""
        if platform.system() != "Linux":
            raise OSError("Traffic-based CAN detection is only supported on Linux.")

        if direction not in {"rx", "tx", "both"}:
            raise ValueError(f"Invalid traffic_direction={direction!r}")

        before: dict[str, tuple[int, int]] = {}
        for iface in ifaces:
            rx = _read_net_counter(iface, "rx_packets")
            tx = _read_net_counter(iface, "tx_packets")
            if rx is None or tx is None:
                continue
            before[iface] = (rx, tx)

        time.sleep(max(duration_s, 0.0))

        after: dict[str, tuple[int, int]] = {}
        for iface in ifaces:
            rx = _read_net_counter(iface, "rx_packets")
            tx = _read_net_counter(iface, "tx_packets")
            if rx is None or tx is None:
                continue
            after[iface] = (rx, tx)

        deltas: dict[str, int] = {}
        for iface in ifaces:
            if iface not in before:
                continue
            # If interface disappeared after unplug (USB-CAN), treat as a drop to 0.
            rx0, tx0 = before[iface]
            rx1, tx1 = after.get(iface, (rx0, tx0))
            drx = rx1 - rx0
            dtx = tx1 - tx0
            if direction == "rx":
                deltas[iface] = drx
            elif direction == "tx":
                deltas[iface] = dtx
            else:
                deltas[iface] = drx + dtx
        return deltas

    def _traffic_detect_can_iface(
        ifaces: list[str],
        *,
        can_mode_label: str,
        bitrate: int | None,
        use_fd: bool,
        data_bitrate: int | None,
        traffic_window_s: float,
        settle_time_s: float,
        wait_timeout_s: float | None,
        min_msgs: int,
        traffic_direction: str,
    ) -> str:
        # Prefer sysfs stats (works even if no process is reading from the socket).
        before_counts = _measure_net_deltas(ifaces, 0.0, traffic_direction)
        before_counts = _measure_net_deltas(ifaces, traffic_window_s, traffic_direction)
        before_hz = {iface: (c / max(traffic_window_s, 1e-6)) for iface, c in before_counts.items()}
        print("Traffic mode:", can_mode_label)
        print(f"Traffic direction={traffic_direction}.")
        print("Packets before (pkts/s):", {k: round(v, 1) for k, v in before_hz.items()})

        if all(c < min_msgs for c in before_counts.values()):
            raise OSError(
                "No sufficient CAN traffic detected on any interface during baseline window. "
                "Move the arm (or increase --traffic-window-s / lower --min-msgs) and retry."
            )

        if wait_timeout_s is None:
            input("Now unplug the target arm CAN cable (or power off that chain) and press Enter...\n")
        else:
            print(
                f"Now unplug the target arm CAN cable within {wait_timeout_s:.1f}s "
                f"(will wait then re-measure)."
            )
            time.sleep(max(wait_timeout_s, 0.0))

        time.sleep(max(settle_time_s, 0.0))

        after_counts = _measure_net_deltas(ifaces, traffic_window_s, traffic_direction)
        after_hz = {iface: (c / max(traffic_window_s, 1e-6)) for iface, c in after_counts.items()}
        print("Packets after  (pkts/s):", {k: round(v, 1) for k, v in after_hz.items()})

        deltas = {iface: after_counts.get(iface, 0) - before_counts.get(iface, 0) for iface in before_counts}
        print("Delta packets:", deltas)

        ranked = sorted(deltas.items(), key=lambda kv: kv[1])
        if not ranked:
            raise OSError("No candidate interfaces to rank for traffic detection.")

        best_iface, best_delta = ranked[0]
        if best_delta >= 0:
            raise OSError(
                "Traffic did not drop on any interface. Make sure you unplugged the correct chain "
                "and that it was generating traffic."
            )
        print(f"Detected CAN interface (largest traffic drop): '{best_iface}' (delta={best_delta})")
        return best_iface

    def _interface_diff_detect(kind: str) -> str | None:
        # Default mode: detect ports by appearance/disappearance.
        if wait_timeout_s is None:
            prompt = (
                "Remove the USB cable from your MotorsBus and press Enter when done."
                if kind == "serial"
                else "Unplug the USB-CAN adapter (or otherwise remove the CAN interface) and press Enter when done."
            )
            input(prompt + "\n")
        else:
            print(
                f"Waiting up to {wait_timeout_s:.1f}s for ports to change (poll_interval_s={poll_interval_s:.2f}s)..."
            )
            start_s = time.time()
            while True:
                if (time.time() - start_s) >= wait_timeout_s:
                    break
                time.sleep(poll_interval_s)
                current = find_available_ports(kind=kind, interfaces=interfaces, prefixes=prefixes, pattern=pattern)
                if set(current) != set(ports_before):
                    break

        time.sleep(max(settle_time_s, 0.0))  # Allow some time for the port/interface to be released
        ports_after = find_available_ports(kind=kind, interfaces=interfaces, prefixes=prefixes, pattern=pattern)
        removed = sorted(set(ports_before) - set(ports_after))
        added = sorted(set(ports_after) - set(ports_before))
        print("Ports after disconnecting:", ports_after)
        print("Removed:", removed)
        print("Added  :", added)

        if len(removed) == 1 and not added:
            port = removed[0]
            print(f"Detected port: '{port}'")
            if kind == "serial":
                print("Reconnect the USB cable.")
            return port
        if len(added) == 1 and not removed:
            port = added[0]
            print(f"Detected port: '{port}' (added)")
            return port
        if not removed and not added:
            return None
        raise OSError(f"Could not detect the port uniquely. removed={removed}, added={added}")

    if kind == "can" and can_mode in {"auto", "traffic"}:
        print(
            "CAN traffic detection notes:\n"
            "- Use this mode for integrated chains where the interface name (can0/can1) stays present.\n"
            "- If the chain is quiet, move a joint/gripper to generate bus traffic.\n"
        )

    if kind == "can" and can_mode == "auto":
        # First try interface add/remove detection (USB-CAN unplug). If no change, fall back to traffic.
        port = _interface_diff_detect(kind)
        if port is not None:
            return port
        return _traffic_detect_can_iface(
            ports_before,
            can_mode_label="auto->traffic",
            bitrate=bitrate,
            use_fd=use_fd,
            data_bitrate=data_bitrate,
            traffic_window_s=traffic_window_s,
            settle_time_s=settle_time_s,
            wait_timeout_s=wait_timeout_s,
            min_msgs=min_msgs,
            traffic_direction=traffic_direction,
        )

    if kind == "can" and can_mode == "traffic":
        return _traffic_detect_can_iface(
            ports_before,
            can_mode_label="traffic",
            bitrate=bitrate,
            use_fd=use_fd,
            data_bitrate=data_bitrate,
            traffic_window_s=traffic_window_s,
            settle_time_s=settle_time_s,
            wait_timeout_s=wait_timeout_s,
            min_msgs=min_msgs,
            traffic_direction=traffic_direction,
        )

    # For serial and can_mode=interface
    port = _interface_diff_detect(kind)
    if port is not None:
        return port

    raise OSError("Could not detect the port. No difference was found.")


def _replace_songling_side_port_in_text(text: str, side: str, port: str) -> tuple[str, int, int]:
    """
    Replace `port:` under `teleop.<side>_arm_config` and `robot.<side>_arm_config`
    while preserving comments/formatting.
    """
    if side not in {"left", "right"}:
        raise ValueError(f"Invalid side={side!r}. Expected 'left' or 'right'.")

    target_arm_key = f"{side}_arm_config:"
    lines = text.splitlines(keepends=True)
    in_top: str | None = None
    in_arm = False
    teleop_hits = 0
    robot_hits = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))

        # Track top-level section.
        if indent == 0 and stripped.endswith(":"):
            section = stripped[:-1]
            in_top = section if section in {"teleop", "robot"} else None
            in_arm = False
            continue

        # Enter side arm subsection.
        if in_top and indent == 2 and stripped == target_arm_key:
            in_arm = True
            continue

        # Leave side arm subsection on sibling key.
        if in_top and indent == 2 and stripped.endswith(":") and stripped != target_arm_key:
            in_arm = False

        if in_top and in_arm and indent == 4 and stripped.startswith("port:"):
            prefix = line[: len(line) - len(line.lstrip(" "))]
            nl = "\n" if line.endswith("\n") else ""
            lines[i] = f"{prefix}port: {port}{nl}"
            if in_top == "teleop":
                teleop_hits += 1
            else:
                robot_hits += 1
            in_arm = False

    return "".join(lines), teleop_hits, robot_hits


def update_songling_config_ports(config_path: Path, side: str, port: str) -> None:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    original = config_path.read_text(encoding="utf-8")
    updated, teleop_hits, robot_hits = _replace_songling_side_port_in_text(original, side=side, port=port)

    if teleop_hits != 1 or robot_hits != 1:
        raise RuntimeError(
            f"Failed to update ports uniquely in {config_path}. "
            f"Expected teleop_hits=1 and robot_hits=1, got teleop_hits={teleop_hits}, robot_hits={robot_hits}."
        )

    config_path.write_text(updated, encoding="utf-8")
    print(
        f"Updated Songling config: side={side}, port={port}, "
        f"targets=teleop.{side}_arm_config.port + robot.{side}_arm_config.port"
    )


def main():
    parser = argparse.ArgumentParser(
        prog="lerobot-find-port",
        description="Find the port/interface associated with a device by unplugging it and comparing the system state.",
    )
    parser.add_argument(
        "--kind",
        "--type",
        default="serial",
        choices=["serial", "can"],
        help="What kind of port to detect: serial (/dev/tty*) or can (can0/slcan0).",
    )
    parser.add_argument(
        "--can-mode",
        default="auto",
        choices=["auto", "interface", "traffic"],
        help="For kind=can: 'auto' tries interface add/remove first, then traffic-drop detection. "
        "'interface' detects add/remove of canX interfaces (USB-CAN unplug). "
        "'traffic' detects which interface loses packets when you unplug the arm.",
    )
    parser.add_argument(
        "--interfaces",
        default=None,
        help="Comma-separated candidate interface names (e.g. 'can0,can1'). If omitted, auto-detect.",
    )
    parser.add_argument(
        "--prefixes",
        default=None,
        help="Comma-separated prefixes to match (only used for kind=can when --pattern/--interfaces are omitted). "
        "Defaults to 'can,slcan'.",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Regex pattern to match interface names (kind=can). Overrides --prefixes.",
    )
    parser.add_argument(
        "--settle-time-s",
        type=float,
        default=0.8,
        help="Extra delay after unplug before sampling ports (seconds).",
    )
    parser.add_argument(
        "--wait-timeout-s",
        type=float,
        default=None,
        help="Non-interactive mode: poll for changes up to this many seconds instead of waiting for Enter.",
    )
    parser.add_argument(
        "--poll-interval-s",
        type=float,
        default=0.2,
        help="Polling interval for --wait-timeout-s mode (seconds).",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=None,
        help="CAN bitrate passed to python-can (optional; for socketcan it is usually already configured).",
    )
    parser.add_argument(
        "--use-fd",
        action="store_true",
        help="Enable CAN FD when opening the interface in traffic mode.",
    )
    parser.add_argument(
        "--data-bitrate",
        type=int,
        default=None,
        help="CAN FD data bitrate passed to python-can (optional).",
    )
    parser.add_argument(
        "--traffic-window-s",
        type=float,
        default=2.0,
        help="Measurement window length for CAN RX traffic sampling (traffic mode).",
    )
    parser.add_argument(
        "--min-msgs",
        type=int,
        default=1,
        help="Minimum baseline RX messages required to consider traffic detection valid.",
    )
    parser.add_argument(
        "--traffic-direction",
        default="rx",
        choices=["rx", "tx", "both"],
        help="Which sysfs counters to use in traffic mode: rx, tx, or both.",
    )
    parser.add_argument(
        "--write-songling-config",
        type=Path,
        default=None,
        help="Optional path to Songling YAML to auto-update after detection "
        "(e.g. examples/songling_aloha/teleop.yaml).",
    )
    parser.add_argument(
        "--write-songling-teleop",
        type=Path,
        default=None,
        help="Deprecated alias of --write-songling-config.",
    )
    parser.add_argument(
        "--songling-side",
        choices=["left", "right"],
        default=None,
        help="Required with --write-songling-config. Which side to update.",
    )
    args = parser.parse_args()

    detected_port = find_port(
        kind=args.kind,
        can_mode=args.can_mode,
        interfaces=_parse_csv_arg(args.interfaces),
        prefixes=_parse_csv_arg(args.prefixes),
        pattern=args.pattern,
        settle_time_s=args.settle_time_s,
        wait_timeout_s=args.wait_timeout_s,
        poll_interval_s=args.poll_interval_s,
        bitrate=args.bitrate,
        use_fd=args.use_fd,
        data_bitrate=args.data_bitrate,
        traffic_window_s=args.traffic_window_s,
        min_msgs=args.min_msgs,
        traffic_direction=args.traffic_direction,
    )

    write_songling_config = args.write_songling_config
    if args.write_songling_teleop is not None:
        if write_songling_config is None:
            write_songling_config = args.write_songling_teleop
        elif write_songling_config != args.write_songling_teleop:
            raise ValueError(
                "--write-songling-config and --write-songling-teleop were both provided with different paths."
            )

    if write_songling_config is not None:
        if args.songling_side is None:
            raise ValueError("--songling-side is required when --write-songling-config is used.")
        update_songling_config_ports(
            config_path=write_songling_config,
            side=args.songling_side,
            port=detected_port,
        )


if __name__ == "__main__":
    main()
