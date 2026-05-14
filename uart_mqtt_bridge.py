#!/usr/bin/env python3
"""
UART → MQTT bridge for STM32N6570-DK person detection firmware.

The firmware prints lines like:
    MQTT_JSON:{"event_type":"crowd_density","level":0,"grid_data":[...],...}

This script reads from a serial port and publishes the JSON payload to the
MQTT broker configured in app_config.h.

Usage:
    pip install pyserial paho-mqtt
    python3 uart_mqtt_bridge.py --port /dev/ttyUSB0 --broker 192.168.1.10
"""

import argparse
import json
import sys
from pathlib import Path
import serial
import paho.mqtt.client as mqtt

MQTT_TOPIC = "stadium/events/congestion"
BAUD_RATE  = 115200
PREFIX     = "MQTT_JSON:"


def _load_env_file(path: Path) -> dict:
    values = {}
    if not path.exists():
        return values

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _detect_default_broker() -> str:
    # Priority:
    # 1) process env var
    # 2) root .env.vm (VM deployments)
    # 3) root .env (local docker-compose)
    # 4) fallback localhost
    import os

    env_host = os.getenv("MQTT_BROKER_HOST")
    if env_host:
        return env_host

    repo_root = Path(__file__).resolve().parent.parent
    env_vm = _load_env_file(repo_root / ".env.vm")
    if env_vm.get("MQTT_BROKER_HOST"):
        return env_vm["MQTT_BROKER_HOST"]
    if env_vm.get("VM_IP"):
        return env_vm["VM_IP"]

    env_local = _load_env_file(repo_root / ".env")
    if env_local.get("MQTT_BROKER_HOST"):
        host = env_local["MQTT_BROKER_HOST"]
        # "mosquitto" only resolves inside docker network.
        if host == "mosquitto" and env_local.get("VM_IP"):
            return env_local["VM_IP"]
        return host

    return "localhost"


def main():
    parser = argparse.ArgumentParser(description="STM32N6 UART → MQTT bridge")
    parser.add_argument("--port",   required=True,          help="Serial port (e.g. /dev/ttyUSB0 or COM3)")
    parser.add_argument("--broker", default=_detect_default_broker(), help="MQTT broker IP/host")
    parser.add_argument("--port-mqtt", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--topic",  default=MQTT_TOPIC,     help="MQTT topic")
    parser.add_argument("--baud",   type=int, default=BAUD_RATE, help="Serial baud rate")
    parser.add_argument("--verbose", action="store_true",   help="Print every published payload")
    args = parser.parse_args()

    mqttc = mqtt.Client(client_id="uart-bridge")
    mqttc.connect(args.broker, args.port_mqtt, keepalive=60)
    mqttc.loop_start()
    print(f"[bridge] Connected to MQTT broker {args.broker}:{args.port_mqtt}")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=2)
        print(f"[bridge] Listening on {args.port} @ {args.baud} baud")
    except serial.SerialException as e:
        print(f"[bridge] ERROR opening serial port: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        while True:
            try:
                raw = ser.readline()
            except serial.SerialException as e:
                print(f"[bridge] Serial read error: {e}", file=sys.stderr)
                break

            if not raw:
                continue

            try:
                line = raw.decode("utf-8", errors="replace").rstrip()
            except Exception:
                continue

            if not line.startswith(PREFIX):
                # Forward other lines to stdout (debug output from firmware)
                print(f"[uart] {line}")
                continue

            payload = line[len(PREFIX):]

            # Validate JSON before publishing
            try:
                json.loads(payload)
            except json.JSONDecodeError as e:
                print(f"[bridge] Invalid JSON skipped ({e}): {payload[:80]}", file=sys.stderr)
                continue

            mqttc.publish(args.topic, payload, qos=0)
            if args.verbose:
                print(f"[mqtt→] {payload}")

    except KeyboardInterrupt:
        print("\n[bridge] Stopped.")
    finally:
        ser.close()
        mqttc.loop_stop()
        mqttc.disconnect()


if __name__ == "__main__":
    main()
