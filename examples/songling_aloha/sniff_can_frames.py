#!/usr/bin/env python

"""Record raw CAN frames for custom MotorsBus adaptation.

Example:
    python examples/songling_aloha/sniff_can_frames.py \
      --interfaces can0 can1 \
      --duration-s 20 \
      --output-dir /tmp/songling_can_sniff
"""

import argparse
import csv
import threading
import time
from pathlib import Path

import can


def sniff_interface(interface: str, bitrate: int, duration_s: float, output_csv: Path, use_fd: bool) -> None:
    kwargs = {"channel": interface, "interface": "socketcan", "bitrate": bitrate}
    if use_fd:
        kwargs["fd"] = True

    bus = can.interface.Bus(**kwargs)
    end_ts = time.time() + duration_s

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "interface", "arbitration_id", "dlc", "is_extended_id", "is_fd", "data_hex"])
        while time.time() < end_ts:
            msg = bus.recv(timeout=0.1)
            if msg is None:
                continue
            writer.writerow(
                [
                    f"{time.time():.6f}",
                    interface,
                    f"0x{msg.arbitration_id:X}",
                    msg.dlc,
                    int(msg.is_extended_id),
                    int(getattr(msg, "is_fd", False)),
                    msg.data.hex().upper(),
                ]
            )

    bus.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sniff raw CAN frames for Songling ALOHA protocol analysis.")
    parser.add_argument("--interfaces", nargs="+", default=["can0", "can1"], help="CAN interfaces to sniff.")
    parser.add_argument("--duration-s", type=float, default=20.0, help="Capture duration for each interface.")
    parser.add_argument("--bitrate", type=int, default=1000000, help="CAN bitrate.")
    parser.add_argument("--use-fd", action="store_true", help="Enable CAN FD mode.")
    parser.add_argument("--output-dir", default="/tmp/songling_can_sniff", help="Directory to save CSV logs.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    threads = []
    for iface in args.interfaces:
        output_csv = output_dir / f"{iface}.csv"
        t = threading.Thread(
            target=sniff_interface,
            args=(iface, args.bitrate, args.duration_s, output_csv, args.use_fd),
            daemon=False,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Saved CAN captures to: {output_dir}")


if __name__ == "__main__":
    main()
