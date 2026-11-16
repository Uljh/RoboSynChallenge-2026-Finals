"""Coarse handoff estimate from observed joints and fixed robot geometry only."""
import json
import numpy as np
import torch

class KinematicGate:
    def __init__(self, base, initial_state, arr, contract, output):
        self.base=base;self.arr=arr;self.output=output;self.trace=[]
        self.offset=np.asarray(contract['end_offsets_finger_midpoint_frame'])
        self.local_mid=[]
        for i in range(2):
            self.local_mid.append(np.linalg.inv(self.tcp(initial_state,i))@self.actual_mid(i))
    def tcp(self,state,i):
        side='left_arm' if i==0 else 'right_arm';sl=slice(0,6) if i==0 else slice(7,13)
        return self.arr(self.base.robot.compute_fk(name=side,qpos=torch.as_tensor(np.asarray(state)[sl].copy()),to_matrix=True)).reshape(4,4)
    def actual_mid(self,i):
        # Robot links only; used to calibrate/audit the fixed kinematic transform.
        robot=self.base.robot._entities[0];prefix='left' if i==0 else 'right'
        a=np.asarray(robot.get_link_pose(prefix+'_link7')).reshape(4,4).copy()
        b=np.asarray(robot.get_link_pose(prefix+'_link8')).reshape(4,4)
        a[:3,3]=(a[:3,3]+b[:3,3])/2
        return a
    def estimate(self,state,control):
        mids=[self.tcp(state,i)@self.local_mid[i] for i in range(2)]
        errors=[float(np.linalg.norm(mids[i][:3,3]-self.actual_mid(i)[:3,3])) for i in range(2)]
        assert max(errors)<.003,'Robot FK/midfinger mapping mismatch: '+str(errors)
        tips=[mids[i][:3,3]+mids[i][:3,:3]@self.offset[i] for i in range(2)]
        gap=float(np.linalg.norm(tips[0]-tips[1]))
        self.trace.append(dict(control=control,estimated_endpoint_m=gap,robot_midpoint_fk_error_m=errors))
        p=self.output/'kinematic_gate_trace.tmp';p.write_text(json.dumps(self.trace));p.replace(self.output/'kinematic_gate_trace.json')
        return gap

