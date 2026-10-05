#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="$ROOT/apps/flexon_scheduler_helper"
INSTALL_DIR="/usr/local/libexec"
INSTALL_PATH="$INSTALL_DIR/flexon_scheduler_helper"

if [[ ! -x "$HELPER" ]]; then
    echo "Missing built helper: $HELPER" >&2
    echo "Build the FlexOn applications first." >&2
    exit 1
fi

command -v sudo >/dev/null || { echo "sudo is required" >&2; exit 1; }
command -v setcap >/dev/null || { echo "setcap is required (install libcap2-bin)" >&2; exit 1; }

sudo install -d -m 0755 "$INSTALL_DIR"
sudo install -o root -g root -m 0755 "$HELPER" "$INSTALL_PATH"
sudo setcap cap_sys_nice=ep "$INSTALL_PATH"

if ! sudo getcap "$INSTALL_PATH" | grep -q 'cap_sys_nice=ep'; then
    echo "Failed to verify CAP_SYS_NICE on $INSTALL_PATH" >&2
    exit 1
fi

echo "Installed privileged scheduler helper: $INSTALL_PATH"
echo "$(sudo getcap "$INSTALL_PATH")"
