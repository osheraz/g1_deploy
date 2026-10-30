"""ROS-independent LiDAR-to-pelvis geometry. No goals, planner or heading policy.

Transforms map child coordinates into parent coordinates. Quaternions are xyzw.
The head and LiDAR mount are fixed. Only measured waist joints change the
LiDAR-to-pelvis transform. This does not correct DLIO yaw drift.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def rotation(quat):
    q = np.asarray(quat, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-6:
        raise ValueError('Expected a finite nonzero xyzw quaternion')
    return Rotation.from_quat(q).as_matrix()


def yaw_alignment(reference, estimate):
    """Best world-Z alignment using the full rotation, also when prone/vertical."""
    delta = reference @ estimate.T
    return float(np.arctan2(delta[1, 0] - delta[0, 1], delta[0, 0] + delta[1, 1]))


class CausalPositionFilter:
    """20 ms exponential XYZ filter; never synthesizes a new measurement.

    Track the weighted measurement time as well as position so downstream
    freshness checks include smoothing delay. Reset across actual data gaps.
    Geometry/clock validity remains the producer's decision, not this filter's.
    """
    def __init__(self,tau_s=.02,max_gap_s=.05):
        if not np.isfinite(tau_s) or tau_s<=0 or not np.isfinite(max_gap_s) or max_gap_s<=0:
            raise ValueError('Filter time constants must be finite and positive')
        self.tau_s=float(tau_s);self.max_gap_s=float(max_gap_s);self.reset()

    def reset(self):
        self.position=None;self.stamp=None;self.effective_stamp=None

    def update(self,position,stamp):
        value=np.asarray(position,dtype=float)
        if value.shape!=(3,) or not np.isfinite(value).all() or not np.isfinite(stamp):
            self.reset();raise ValueError('Position filter requires finite XYZ and measurement time')
        if self.stamp is not None and stamp<=self.stamp:
            raise ValueError('Position filter requires strictly increasing measurement times')
        if self.stamp is None or stamp-self.stamp>self.max_gap_s:
            self.position=value.copy();self.effective_stamp=float(stamp)
        else:
            alpha=-np.expm1(-(stamp-self.stamp)/self.tau_s)
            self.position+=alpha*(value-self.position)
            self.effective_stamp+=alpha*(stamp-self.effective_stamp)
        self.stamp=float(stamp)
        return self.position.copy(),max(0.,self.stamp-self.effective_stamp)


class PelvisPoseEstimator:
    """Continuous common-frame pelvis pose. Episode re-zeroing stays in runtime."""
    def __init__(self, body, *, mount_rotation=None):
        self.body = body
        self.mount_rotation = mount_rotation
        if mount_rotation is not None:
            r = np.asarray(mount_rotation, dtype=float)
            if r.shape != (3, 3) or not np.isfinite(r).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1):
                raise ValueError('Mount rotation must be a proper 3x3 rotation')
            self.mount_rotation = r
        self.align = None
        self.last_stamp = None

    def update(self, stamp, lidar_position, lidar_quat_xyzw, state):
        if not np.isfinite(stamp) or stamp <= 0 or (self.last_stamp is not None and stamp <= self.last_stamp):
            raise ValueError('Nonpositive or out-of-order odometry timestamp')
        p = np.asarray(lidar_position, dtype=float)
        joints = np.asarray(state['q'], dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all() or joints.shape != (29,) or not np.isfinite(joints).all():
            raise ValueError('Invalid LiDAR position or joint state')
        rl, rp = rotation(lidar_quat_xyzw), rotation(state['quat_xyzw'])
        # Arms and legs do not belong to the pelvis-to-LiDAR chain.
        getattr(self.body,'set_waist_joints',self.body.set_joints)(joints)
        torso_in_pelvis = self.body.transform('torso_link', 'pelvis').copy()
        nominal = self.body.transform('mid360_link', 'torso_link').copy()
        if self.mount_rotation is not None:
            nominal[:3, :3] = self.mount_rotation
        info = {}
        if state.get('torso_quat_xyzw') is not None:
            measured_torso = rotation(state['torso_quat_xyzw'])
            fk_torso = rp @ torso_in_pelvis[:3, :3]
            info['torso_fk_tilt_disagreement_rad'] = float(np.arccos(np.clip(measured_torso[2] @ fk_torso[2], -1., 1.)))
        lidar_in_pelvis = torso_in_pelvis @ nominal
        r_est = rl @ lidar_in_pelvis[:3, :3].T
        p_est = p - r_est @ lidar_in_pelvis[:3, 3]
        if self.align is None:
            self.align = yaw_alignment(rp, r_est)
        align = Rotation.from_euler('z', self.align).as_matrix()
        r_est, p_est = align @ r_est, align @ p_est
        tilt = float(np.arccos(np.clip(r_est[2] @ rp[2], -1., 1.)))
        info.update(position_m=p_est.tolist(), pelvis_quat_xyzw=Rotation.from_matrix(r_est).as_quat().tolist(),
                    imu_quat_xyzw=Rotation.from_matrix(rp).as_quat().tolist(),
                    orientation_disagreement_rad=float(Rotation.from_matrix(rp.T @ r_est).magnitude()),
                    tilt_disagreement_rad=tilt, yaw_align_rad=self.align,
                    yaw_disagreement_rad=yaw_alignment(rp, r_est))
        self.last_stamp = stamp
        return info


def simulation_packet(position, quat_xyzw, capture_monotonic_s):
    """Explicit simulator-truth producer for the existing odometry receiver.

    Caller supplies simulator world axes matching the low-state quaternion and a
    capture time mapped to the consuming process clock. No episode zeroing here.
    """
    rotation(quat_xyzw)
    p = np.asarray(position, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or not np.isfinite(capture_monotonic_s):
        raise ValueError('Invalid simulator pose/time')
    return dict(schema='hv2.pelvis_odometry.v1', frame='pelvis_imu_gravity_frame',
                estimator='simulator_ground_truth', tracking_valid=True, clock_uncertainty_s=0.,
                position_m=p.tolist(), pelvis_quat_xyzw=list(quat_xyzw),
                capture_monotonic_s=float(capture_monotonic_s))


class EpisodePose:
    """Causal display/log origin at the first policy observation, never a late reset.

    This consumes the runtime's already-updated heading; it does not implement a
    second heading recurrence or change planner goals. Unqualified XYZ is labelled.
    """
    def __init__(self):
        self.started=False;self.position0=None;self.rotation0=None;self.heading0=None

    def update(self, sample, quat_xyzw, heading):
        attitude=Rotation.from_quat(quat_xyzw)
        if not self.started:
            self.started=True;self.rotation0=attitude;self.heading0=float(heading)
            if sample['available']:self.position0=np.asarray(sample['position_m'],dtype=float)
        result=dict(sample,position_reset_m=None,height_rel_m=None,
            origin_available=self.position0 is not None,
            orientation_relative_xyzw=(self.rotation0.inv()*attitude).as_quat().tolist(),
            yaw_rel_rad=float(np.arctan2(np.sin(heading-self.heading0),np.cos(heading-self.heading0))))
        if self.position0 is not None and sample['available']:
            delta=np.asarray(sample['position_m'])-self.position0
            c,s=np.cos(self.heading0),np.sin(self.heading0)
            result['position_reset_m']=[float(c*delta[0]+s*delta[1]),float(-s*delta[0]+c*delta[1]),float(delta[2])]
            result['height_rel_m']=float(delta[2])
        return result
