"""Hybrid gait reference: mocap gait cycles for the lower body, planner upper.

Replaces only the planner's LOWER-body reference channels with a
phase-indexed template extracted from real walking mocap
(scripts/extract_gait_cycle.py). Motivation: the planner cannot walk slower
than ~0.4 m/s; templates from genuinely slow clips stay in SONIC's training
distribution at any commanded speed.

Formulated as additive deviations about the parent window, so alpha -> 0
hands over exactly to the parent's idle-hold / stand behavior:

    q  = parent_q  + alpha * (cycle(phase) - template_stand)
    dq = parent_dq + alpha * cycle_dq(phase) * (natural_period / period)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from policy.pico_reference import LOWER_BODY_IL_IDX, PicoReferenceWindow

MAX_TURN_RATE = 0.6        # rad/s toward the commanded heading
ALPHA_UP_S = 0.4           # walk envelope attack
ALPHA_DOWN_S = 0.6         # release
WALK_DEFAULT_SPEED = 0.4   # commanded speed for WALK's sentinel -1
MIN_SPEED = 0.03


class _Template:
    def __init__(self, path):
        d = np.load(Path(path).expanduser())
        self.q = np.asarray(d["q_cycle"], np.float64)          # (P,12)
        self.dq = np.asarray(d["dq_cycle"], np.float64)
        self.z = np.asarray(d["root_z_cycle"], np.float64)
        self.stand = np.asarray(d["stand_q"], np.float64)
        self.stride = float(d["stride_m"])
        self.period = float(d["period_s"])
        self.speed = self.stride / self.period
        self.meta = json.loads(bytes(d["meta"]).decode())
        assert list(d["lower_il_idx"]) == list(LOWER_BODY_IL_IDX), \
            "template lower-body indices do not match this runtime"

    def sample(self, phase):
        """Cyclic linear interp; phase scalar or array in cycles."""
        p = np.atleast_1d(np.asarray(phase, np.float64)) % 1.0
        x = p * len(self.q)
        i0 = np.floor(x).astype(int) % len(self.q)
        i1 = (i0 + 1) % len(self.q)
        w = (x - np.floor(x))[:, None]
        q = (1 - w) * self.q[i0] + w * self.q[i1]
        dq = (1 - w) * self.dq[i0] + w * self.dq[i1]
        z = (1 - w[:, 0]) * self.z[i0] + w[:, 0] * self.z[i1]
        return q, dq, z


def _yaw_of(q):
    w, x, y, z = np.asarray(q, np.float64).reshape(4)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _yaw_quat(theta):
    return np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)], np.float64)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class HybridGaitReference(PicoReferenceWindow):
    """PicoReferenceWindow with template-generated lower-body gait."""

    def __init__(self, config, client, templates, **kwargs):
        super().__init__(config, client, **kwargs)
        tpls = [_Template(p) for p in (templates if isinstance(templates, (list, tuple))
                                       else [templates])]
        self.templates = sorted(tpls, key=lambda t: t.speed)
        self._phase = 0.0
        self._alpha = 0.0
        self._theta = 0.0
        self._omega = 0.0
        self._speed = 0.0
        self._period = self.templates[0].period
        names = ", ".join(f"{t.meta['source'].split('/')[-1]}@{t.speed:.2f}m/s"
                          for t in self.templates)
        print(f"Hybrid gait reference: lower body from templates [{names}], "
              f"upper body from the planner")

    # -- gait state ------------------------------------------------------

    def initialize_from_robot(self, base_quat_wxyz, dof_pos) -> None:
        super().initialize_from_robot(base_quat_wxyz, dof_pos)
        self._theta = _yaw_of(base_quat_wxyz)

    def _command_speed(self):
        mode = self.client.mode()
        nav = self.client.nav_command()
        if mode is None or nav is None or int(mode) == 0:
            return 0.0, None
        move = np.asarray(nav["movement"], np.float64)[:2]
        if np.linalg.norm(move) < 1e-6:
            return 0.0, None
        speed = float(nav["speed"])
        if speed <= 0:
            speed = WALK_DEFAULT_SPEED
        facing = np.asarray(nav["facing"], np.float64)[:2]
        theta_cmd = float(np.arctan2(facing[1], facing[0]))
        return max(speed, MIN_SPEED), theta_cmd

    def _blend_templates(self, v):
        ts = self.templates
        if len(ts) == 1 or v <= ts[0].speed:
            return ts[0], None, 0.0
        if v >= ts[-1].speed:
            return ts[-1], None, 0.0
        for a, b in zip(ts, ts[1:]):
            if a.speed <= v <= b.speed:
                return a, b, (v - a.speed) / (b.speed - a.speed)
        return ts[0], None, 0.0

    def step(self) -> None:
        super().step()
        dt = float(self.config.control_dt)
        v, theta_cmd = self._command_speed()

        if v > 0:
            self._alpha = min(1.0, self._alpha + dt / ALPHA_UP_S)
            self._speed = v
        else:
            self._alpha = max(0.0, self._alpha - dt / ALPHA_DOWN_S)

        if theta_cmd is not None:
            self._omega = float(np.clip(2.0 * _wrap(theta_cmd - self._theta),
                                        -MAX_TURN_RATE, MAX_TURN_RATE))
        else:
            self._omega = 0.0
        self._theta = _wrap(self._theta + self._omega * dt)

        if self._alpha > 0:
            t, _, _ = self._blend_templates(self._speed)
            # slower than the slowest template: shrink stride, stretch period
            v = max(self._speed, MIN_SPEED)
            if v < t.speed:
                stride = t.stride * (v / t.speed) ** 0.5
            else:
                stride = t.stride
            self._period = float(np.clip(stride / v, 0.6, 6.0))
            self._phase = (self._phase + self._alpha * dt / self._period) % 1.0

    # -- window override -------------------------------------------------

    def _gait_deviation(self, phase):
        """(dq12, dvel12, dz) deviations for one phase sample."""
        a, b, w = self._blend_templates(self._speed)
        qa, dqa, za = a.sample(phase)
        if b is not None:
            qb, dqb, zb = b.sample(phase)
            q = (1 - w) * qa + w * qb
            dq = (1 - w) * dqa + w * dqb
            z = (1 - w) * za + w * zb
            stand = (1 - w) * a.stand + w * b.stand
            nat = (1 - w) * a.period + w * b.period
            zmean = (1 - w) * a.z.mean() + w * b.z.mean()
        else:
            q, dq, z = qa, dqa, za
            stand, nat, zmean = a.stand, a.period, a.z.mean()
        rate = nat / max(self._period, 1e-3)
        return (q[0] - stand), dq[0] * rate, (z[0] - zmean)

    def window(self) -> dict:
        out = super().window()
        a = self._alpha
        if a <= 1e-3:
            return out

        n = out["dof_fut"].shape[1]
        stride_dt = self.model_profile.g1_future_stride * float(self.config.control_dt)
        dof = np.asarray(out["dof_fut"][0], np.float64)
        dvel = np.asarray(out["dvel_fut"][0], np.float64)
        rootq = np.asarray(out["rootq_fut"][0], np.float64)
        rootz = np.asarray(out["rootz_fut"][0], np.float64)

        for k in range(n):
            tk = k * stride_dt
            phase_k = self._phase + tk / self._period
            dq12, dv12, dz = self._gait_deviation(phase_k)
            dof[k, LOWER_BODY_IL_IDX] += a * dq12
            dvel[k, LOWER_BODY_IL_IDX] += a * dv12
            rootz[k] += a * dz
            theta_k = self._theta + self._omega * tk
            qk = _yaw_quat(theta_k)
            blended = (1 - a) * rootq[k] + a * qk
            rootq[k] = blended / max(np.linalg.norm(blended), 1e-9)

        out["dof_fut"] = dof[None].astype(np.float32)
        out["dvel_fut"] = dvel[None].astype(np.float32)
        out["rootq_fut"] = rootq[None].astype(np.float32)
        out["rootz_fut"] = rootz[None].astype(np.float32)

        cq = np.asarray(out["body_quat_cur"][0, 0], np.float64)
        ours = _yaw_quat(self._theta)
        blended = (1 - a) * cq + a * ours
        out["body_quat_cur"][0, 0] = (blended / max(np.linalg.norm(blended), 1e-9)
                                      ).astype(np.float32)
        out["reference_gait_active"] = np.bool_(True)

        self._last_reference_q_il = np.asarray(out["dof_fut"][0, 0], np.float32).copy()
        self._last_reference_dq_il = np.asarray(out["dvel_fut"][0, 0], np.float32).copy()
        return out
