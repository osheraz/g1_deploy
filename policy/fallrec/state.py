"""Planner state history from FK + IMU instead of the simulator; both encoders are frame-relative."""

from __future__ import annotations

import torch


class DeployStateHistory:
    """Newest-first rings matching `PlannerStateHistory.query`."""

    def __init__(self, bundle, fk, dof_names, body_names, device):
        self.fk = fk
        self.device = device
        self.H = int(bundle.H)
        self.body_names = list(body_names)
        self.fk_index = torch.tensor(
            [fk.link_index(n) for n in self.body_names], dtype=torch.long
        )
        nb, ndof = len(self.body_names), len(dof_names)
        self.body_pos = torch.zeros(1, self.H, nb, 3, device=device)
        self.body_rot = torch.zeros(1, self.H, nb, 4, device=device)
        self.body_ang_vel = torch.zeros(1, self.H, nb, 3, device=device)
        self.body_vel = torch.zeros(1, self.H, nb, 3, device=device)
        self.dof_pos = torch.zeros(1, self.H, ndof, device=device)
        self.dof_vel = torch.zeros(1, self.H, ndof, device=device)
        self.filled = 0

    def reset(self) -> None:
        self.filled = 0

    @torch.no_grad()
    def push(self, dof_pos, dof_vel, base_quat_xyzw, base_ang_vel) -> None:
        from motion_diffusion.features.rotation import quat_apply, quat_mul

        pos_l, rot_l = self.fk.forward_link_pos_rot_xyzw(dof_pos)
        pos_l = torch.as_tensor(pos_l, device=self.device)[:, self.fk_index]
        rot_l = torch.as_tensor(rot_l, device=self.device)[:, self.fk_index]

        q = base_quat_xyzw.to(self.device).reshape(1, 1, 4).expand(-1, pos_l.shape[1], -1)
        body_pos = quat_apply(q, pos_l)
        body_rot = quat_mul(q, rot_l)
        ang_vel = base_ang_vel.to(self.device).reshape(1, 1, 3).expand_as(body_pos)

        for buf, val in ((self.body_pos, body_pos), (self.body_rot, body_rot),
                         (self.body_ang_vel, ang_vel)):
            buf.copy_(torch.roll(buf, shifts=1, dims=1))
            buf[:, 0] = val
        for buf, val in ((self.dof_pos, dof_pos), (self.dof_vel, dof_vel)):
            buf.copy_(torch.roll(buf, shifts=1, dims=1))
            buf[:, 0] = val.to(self.device).reshape(1, -1)

        if self.filled < self.H:
            self.filled += 1
            if self.filled == 1:
                for buf in (self.body_pos, self.body_rot, self.body_ang_vel,
                            self.dof_pos, self.dof_vel):
                    buf[:, 1:] = buf[:, :1]

    def window(self) -> dict:
        return {
            "body_pos": self.body_pos, "body_rot": self.body_rot,
            "body_vel": self.body_vel, "body_ang_vel": self.body_ang_vel,
            "dof_pos": self.dof_pos, "dof_vel": self.dof_vel,
        }

    @property
    def ready(self) -> bool:
        return self.filled >= self.H
