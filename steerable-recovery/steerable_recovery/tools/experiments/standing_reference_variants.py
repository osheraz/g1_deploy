"""Research-only reference alternatives; not imported by robot deployment."""
import torch
from motion_diffusion.behavioral.standing import StandingReference
from humanoidverse2.agents.action_vae.heading import heading_quat
from humanoidverse2.agents.action_vae.representation import build_state67


class ExperimentalStandingReference(StandingReference):
    def activate(self, time_s, q, quat, position, heading, previous_code=None, dq=None):
        if self.active:
            return
        self.start_time = float(time_s)
        self.start_q = q.detach().clone()
        self.start_dq = torch.zeros_like(q) if dq is None else dq.detach().clone()
        self.transition_s = self.blend_s
        if getattr(self, 'adaptive_transition', False):
            distance=float((self.default_q[None]-q).abs().max())
            speed=float(self.start_dq.abs().max())
            self.transition_s=min(1.5,max(self.blend_s,distance/1.,speed/6.))
        # The default remains the checkpoint posture. A captured-pose comparison
        # is used only by the online standing experiment until separately selected.
        self.target_q = (self.start_q.clone() if getattr(self, 'pose_mode', 'default') == 'captured'
                         else self.default_q[None].clone())
        self.start_quat = quat.detach().clone()
        self.target_quat = heading_quat(heading).detach().clone()
        self.target_position = position.detach().clone()
        self.start_code = None if previous_code is None else previous_code.detach().clone()

    @torch.no_grad()
    def forward(self, time_s, position, heading):
        if not self.active:
            raise RuntimeError('Standing reference has not been activated')
        times = (float(time_s) - self.start_time + self.offsets).clamp_min(0)
        u = (times / self.transition_s).clamp(0, 1)
        weight = 10*u**3 - 15*u**4 + 6*u**5
        rate = (30*u**2 - 60*u**3 + 30*u**4) / self.transition_s
        delta = self.target_q - self.start_q
        q = self.start_q[:, None] + weight[None, :, None]*delta[:, None]
        dq = rate[None, :, None]*delta[:, None]
        # Quintic Hermite boundary conditions: measured q/dq initially, desired
        # standing q and zero dq/acceleration finally. Zero dq reproduces v1.
        velocity_shape=u-6*u**3+8*u**4-3*u**5
        velocity_rate=1-18*u**2+32*u**3-15*u**4
        q=q+self.transition_s*velocity_shape[None,:,None]*self.start_dq[:,None]
        dq=dq+velocity_rate[None,:,None]*self.start_dq[:,None]
        # Shortest-arc normalized interpolation; initial attitude is preserved
        # at offset zero, then smoothly approaches upright at captured heading.
        end = torch.where((self.start_quat*self.target_quat).sum(-1, keepdim=True) < 0,
                          -self.target_quat, self.target_quat)
        quat = (1-weight[None, :, None])*self.start_quat[:, None] + weight[None, :, None]*end[:, None]
        quat = quat/quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        target = position if self.frame == 'local' else self.target_position
        tokens = build_state67(q, dq, target[:, None].expand(-1, 5, -1), quat,
                               position, heading, self.default_q)
        mask = torch.ones(tokens.shape[:2], device=tokens.device, dtype=tokens.dtype)
        mu, logvar = self.encoder(torch.cat((self.normalizer.normalize(tokens).flatten(1), mask), -1))
        code = self.encoder.latent(mu, logvar, stochastic=False)
        if self.start_code is not None:
            # Pose-space continuity alone does not make the planner's last z
            # equal to the encoder's first z. Preserve the actually executed
            # code at the switch, then ease into encoded standing intent.
            if self.start_code.shape != code.shape or not torch.isfinite(self.start_code).all():
                raise ValueError('Standing handoff requires the previous finite native code')
            code = self.start_code + weight[0]*(code-self.start_code)
        if not torch.isfinite(code).all():
            raise FloatingPointError('Standing encoder returned a nonfinite code')
        return code
