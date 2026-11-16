"""Continuous native-FK solution, bounded by original position and speed limits."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def solve(base,arr,target,previous,limits,vlimit,side):
 import torch
 low=np.maximum(limits[:,0]+1e-7,previous-vlimit*.1);high=np.minimum(limits[:,1]-1e-7,previous+vlimit*.1);assert np.all(low<high)
 def fk(q):return arr(base.robot.compute_fk(name=side,qpos=torch.as_tensor(q,dtype=torch.float32),to_matrix=True)).reshape(4,4)
 def residual(q):
  actual=fk(q);return np.r_[(actual[:3,3]-target[:3,3])*10,Rotation.from_matrix(actual[:3,:3]@target[:3,:3].T).as_rotvec()]
 def jac(q):
  cols=[]
  for j in range(6):
   h=2e-4;a=q.copy();b=q.copy();a[j]+=h;b[j]-=h;cols.append((residual(a)-residual(b))/(2*h))
  return np.stack(cols,axis=1)
 result=least_squares(residual,np.clip(previous,low+1e-10,high-1e-10),jac=jac,bounds=(low,high),max_nfev=100,ftol=1e-10,xtol=1e-9,gtol=1e-9)
 q=result.x.astype(np.float32);actual=fk(q);position=float(np.linalg.norm(actual[:3,3]-target[:3,3]));angle=float(Rotation.from_matrix(actual[:3,:3]@target[:3,:3].T).magnitude())
 info=dict(method='bounded_native_fk_local_ik',side=side,nfev=result.nfev,position_error_m=position,orientation_error_rad=angle)
 assert position<.0003 and angle<.007,info
 assert np.all(np.abs(q-previous)/.1<=vlimit+1e-5)
 return q,info

