import numpy as np,torch,json
from scipy.spatial.transform import Rotation
class LearnedServo:
 def __init__(self,base,arr,policy,gate,previous,limits,output):
  self.base=base;self.arr=arr;self.policy=policy;self.previous=previous.copy();self.limits=limits;self.output=output;self.history=[];self.trace=[]
  T=gate.local_mid[0];self.local_tip=T[:3,3]+T[:3,:3]@gate.offset[0]
 def infer(self,obs,control):
  raw=self.policy.request('precision',obs)[0,:9];self.history.append(raw);p=np.median(np.array(self.history[-3:]),axis=0);axis=p[3:6];axis/=np.linalg.norm(axis);leftaxis=p[6:9];leftaxis/=np.linalg.norm(leftaxis);delta=p[:3];gap=-float(delta@axis);lateral=delta+gap*axis;distance=float(np.linalg.norm(lateral));angle=float(np.arccos(np.clip(axis@leftaxis,-1,1)));cross=np.cross(leftaxis,axis);rot=Rotation.from_rotvec(cross/max(np.linalg.norm(cross),1e-9)*min(angle,np.deg2rad(.15))).as_matrix()
  aligned=distance<=.001 and angle<=np.deg2rad(.8)
  forward=.0004 if aligned else (-.0004 if gap<.03 else 0.)
  correction=-lateral*min(.35,.0006/max(distance,1e-9))+axis*forward
  state=np.asarray(obs['observation/state']);tcp=self.arr(self.base.robot.compute_fk(name='left_arm',qpos=torch.as_tensor(state[:6].copy(),dtype=torch.float32),to_matrix=True)).reshape(4,4);target=tcp.copy();target[:3,:3]=rot@tcp[:3,:3];target[:3,3]+=correction+(np.eye(3)-rot)@(tcp[:3,:3]@self.local_tip)
  ret=self.base.robot.compute_ik(pose=target,joint_seed=self.previous[:6],name='left_arm');q=self.arr(ret[1]).reshape(-1,6)[0] if ret is not None and bool(self.arr(ret[0]).all()) else None
  if q is None or np.max(np.abs(q-self.previous[:6]))>.005 or np.any(q<self.limits[:6,0]) or np.any(q>self.limits[:6,1]):
   from .local_ik import solve
   q,info=solve(self.base,self.arr,target,self.previous[:6],self.limits[:6],np.full(6,.05),'left_arm')
  assert np.max(np.abs(q-self.previous[:6]))<=.00501
  cmd=self.previous.copy();cmd[:6]=q;self.previous=cmd.copy();record=dict(control=control,estimated_endpoint_delta_m=delta.tolist(),estimated_gap_mm=gap*1000,estimated_lateral_mm=distance*1000,estimated_angle_deg=float(np.degrees(angle)),forward_mm=forward*1000,raw_estimate=raw.tolist(),input='RGB, camera calibration and observed joints only',right_arm_held=True)
  with (self.output/'learned_servo_trace.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  return cmd[None]

