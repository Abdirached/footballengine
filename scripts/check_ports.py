#!/usr/bin/env python3
"""Scan localhost for listening TCP ports.

Quick mode checks common dev ports (frontend 3000, backend 8000, DB, etc.);
full mode sweeps 1-65535 concurrently.

Usage:
    python scripts/check_ports.py [--host 127.0.0.1] [--quick|--full]
"""
from __future__ import annotations

import argparse
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed

COMMON_PORTS = [22, 80, 443, 3000, 8000, 8080, 5432, 6379, 5000, 27017]

MAX_PORT = 65535


def is_open(port: int, host: str, timeout: float) -> int | None:
    """Return port if it accepts a TCP connection on host, else None."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            if s.connect_ex((host, port)) == 0:
                return port
        except (OSError, socket.gaierror):
            pass
    return None


def scan(ports: list[int], host: str, timeout: float, workers: int) -> list[int]:
    open_ports: list[int] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(is_open, p, host, timeout): p for p in ports}
        for fut in as_completed(futures):
            port = fut.result()
            if port is not None:
                open_ports.append(port)
    return sorted(open_ports)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check for open TCP ports on localhost.")
    ap.add_argument("--host", default="127.0.0.1", help="host to probe (default 127.0.0.1)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="only common dev ports (default)")
    mode.add_argument("--full", action="store_true", help="all ports 1-65535")
    mode.add_argument("--range", default="", help="custom range, e.g. 3000-3010,8000")
    ap.add_argument("--timeout", type=float, default=0.3, help="connect timeout in s (default 0.3)")
    ap.add_argument("--workers", type=int, default=256)
    args = ap.parse_args()

    if args.range:
        ports: list[int] = []
        for part in args.range.split(","):
            if "-" in part:
                lo, hi = map(int, part.split("-", 1))
                ports.extend(range(lo, hi + 1))
            else:
                ports.append(int(part))
    elif args.full:
        ports = list(range(1, MAX_PORT + 1))
    else:
        ports = COMMON_PORTS

    print(f"Scanning {len(ports)} port(s) on {args.host} ...")
    open_ports = scan(ports, args.host, args.timeout, args.workers)

    if not open_ports:
        print("No open ports found.")
        return

    print(f"Open ports ({len(open_ports)}):")
    for p in open_ports:
        try:
            svc = socket.getservbyport(p)
        except OSError:
            svc = "unknown"
        print(f"  {p:>6}  {svc}")


if __name__ == "__main__":
    main()
