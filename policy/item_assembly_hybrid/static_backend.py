"""Observation-only kinematics for an OPTIONAL frozen-simulation adapter.

This replaces the native robot API, not a calibrated real robot. Its gate
midpoint check is self-consistency only. Native IK is deliberately unavailable;
the unchanged controller uses its existing bounded local-IK fallback.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .robot_fk import CHAIN, link6


def as_array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class StaticRobotBackend:
    def __init__(self, state):
        self.robot = self
        self._entities = [self]
        self.update(state)

    def update(self, state):
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError('Expected finite canonical 14D observed state')
        self.state = state.copy()

    def compute_fk(self, *, name, qpos, to_matrix=True):
        if name not in ('left_arm', 'right_arm') or not to_matrix:
            raise ValueError('Only left/right matrix FK is supported')
        state = self.state.copy()
        state[:6] = as_array(qpos).reshape(6) if name == 'left_arm' else state[:6]
        if name == 'right_arm':
            state[7:13] = as_array(qpos).reshape(6)
        return link6(state, name.split('_')[0])

    def compute_ik(self, **kwargs):
        return None  # Use the original bounded local_ik.solve, never an unbounded solve.

    def get_link_pose(self, name):
        side, link = name.split('_')
        if side not in ('left', 'right') or link not in ('link7', 'link8'):
            raise ValueError(name)
        joint = next(j for j in CHAIN if j['name'] == 'joint' + link[-1])
        origin = np.eye(4)
        origin[:3, :3] = Rotation.from_euler('xyz', joint['rpy']).as_matrix()
        origin[:3, 3] = joint['xyz']
        motion = np.eye(4)
        # Frozen simulation geometry only; not an action-interface conversion.
        opening = self.state[6 if side == 'left' else 13] * .05
        motion[:3, 3] = np.asarray(joint['axis']) * opening
        return link6(self.state, side) @ origin @ motion
