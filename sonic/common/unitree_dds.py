"""Process-local CycloneDDS setup for Unitree's Python channel wrapper."""
from __future__ import annotations

import os
from pathlib import Path


def configure_process_local_cyclonedds_log(channel_module) -> Path:
    """Replace Unitree's shared ``/tmp/cdds.LOG`` with a per-process path.

    ``ChannelFactoryInitialize(..., interface)`` passes the SDK's embedded XML
    directly to CycloneDDS. On shared machines, an existing ``/tmp/cdds.LOG``
    owned by another user can make domain creation fail under
    ``fs.protected_regular``. A UID/PID-qualified path also prevents concurrent
    simulator and robot processes from truncating each other's diagnostics.
    """

    default_entry = "<OutputFile>/tmp/cdds.LOG</OutputFile>"
    log_path = Path("/tmp") / f"e_skin-cdds-{os.getuid()}-{os.getpid()}.log"
    replacement = f"<OutputFile>{log_path}</OutputFile>"
    config = channel_module.ChannelConfigHasInterface
    if default_entry in config:
        channel_module.ChannelConfigHasInterface = config.replace(
            default_entry, replacement, 1
        )
    elif replacement not in config:
        raise RuntimeError(
            "Unitree CycloneDDS configuration changed; refusing to initialize "
            "without a verified process-local log path"
        )
    return log_path
