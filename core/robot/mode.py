"""Handoff from Unitree's high-level service to low-level control.

The motion service must be released before constructing any ``LowCmd``
publisher.  This mirrors NVIDIA's C++ deployment and Unitree's Python
low-level examples while keeping the transition independently testable.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


def release_active_motion_mode(
    client_factory: Callable[[], Any] | None = None,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    retry_s: float = 5.0,
    max_releases: int = 6,
) -> Any:
    """Release any active Unitree high-level motion service and return its client.

    The returned client should be retained for the lifetime of the deployment.
    A failed or unconfirmed release raises before the caller creates its robot
    interface, so low- and high-level controllers cannot knowingly coexist.
    """
    if client_factory is None:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
            MotionSwitcherClient,
        )

        client_factory = MotionSwitcherClient

    if max_releases < 1:
        raise ValueError("max_releases must be at least one")

    client = client_factory()
    client.SetTimeout(5.0)
    client.Init()

    for release_count in range(max_releases + 1):
        status, result = client.CheckMode()
        if status != 0 or not isinstance(result, dict):
            raise RuntimeError(
                "Unitree MotionSwitcher CheckMode failed "
                f"(status={status}, result={result!r}); no LowCmd publisher "
                "was created"
            )

        name = str(result.get("name") or "")
        form = str(result.get("form") or "")
        if not name:
            print(
                "[MODE SWITCH] High-level motion service release confirmed; "
                "low-level control is available"
            )
            return client

        if release_count == max_releases:
            raise RuntimeError(
                "Unitree high-level motion service is still active after "
                f"{max_releases} release requests: name={name!r}, form={form!r}; "
                "no LowCmd publisher was created"
            )

        print(
            "[MODE SWITCH] Active high-level service "
            f"name={name!r}, form={form!r}; requesting automatic release "
            f"({release_count + 1}/{max_releases})"
        )
        release_status, _ = client.ReleaseMode()
        if release_status != 0:
            raise RuntimeError(
                "Unitree MotionSwitcher ReleaseMode failed "
                f"(status={release_status}, name={name!r}, form={form!r}); "
                "no LowCmd publisher was created"
            )
        sleep_fn(float(retry_s))

    raise AssertionError("unreachable")
