"""Dependency-free state transitions for SONIC decoder action history."""

from __future__ import annotations

"""Vendored from the SONIC research repo so this deploy is standalone.
Source: sonic/policy/history.py
Pure numpy; do not edit here -- re-copy if upstream changes.
"""

import numpy as np


def clear_last_action_rows(last_action: np.ndarray, env_ids) -> None:
    """Clear selected episode seeds in-place before history warm-fill."""

    last_action[np.asarray(env_ids, dtype=np.int64)] = 0.0


def clear_action_state_rows(
    decoded_action: np.ndarray,
    feedback_action: np.ndarray,
    env_ids,
) -> None:
    """Clear both observable decoder output and feedback history on reset."""

    clear_last_action_rows(decoded_action, env_ids)
    clear_last_action_rows(feedback_action, env_ids)


def masked_last_action(action, unapplied_indices=None) -> np.ndarray:
    """Copy decoded actions and zero joints that were not physically applied."""

    out = np.asarray(action, dtype=np.float64).copy()
    if unapplied_indices is not None:
        out[:, np.asarray(unapplied_indices, dtype=np.int64)] = 0.0
    return out


def action_state_from_decoded(
    action,
    unapplied_indices=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the raw decoder output and the physically consistent feedback copy."""

    decoded = np.asarray(action, dtype=np.float64).copy()
    feedback = masked_last_action(decoded, unapplied_indices)
    return decoded, feedback
