from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
CHAIN=json.loads((Path(__file__).resolve().parent/'assets/robot_chain.json').read_text())['joints']
def link6(state,side,base_position=(0,0,.835)):
 q=np.asarray(state);single=q.ndim==1;q=np.atleast_2d(q);angles=q[:,:6] if side=='left' else q[:,7:13];T=np.broadcast_to(np.eye(4),(len(q),4,4)).copy();T[:,:3,3]=np.array(base_position)+np.array([.233,.3 if side=='left' else -.3,0])
 for i in range(6):
  j=next(x for x in CHAIN if x['name']=='joint'+str(i+1));origin=np.eye(4);origin[:3,:3]=Rotation.from_euler('xyz',j['rpy']).as_matrix();origin[:3,3]=j['xyz'];motion=np.broadcast_to(np.eye(4),(len(q),4,4)).copy();motion[:,:3,:3]=Rotation.from_rotvec(angles[:,i,None]*np.array(j['axis'])).as_matrix();T=T@origin@motion
 return T[0] if single else T
def finger_midpoint(state,side,base_position=(0,0,.835)):
 T=link6(state,side,base_position);j=next(x for x in CHAIN if x['name']=='joint7');M=np.eye(4);M[:3,:3]=Rotation.from_euler('xyz',j['rpy']).as_matrix();M[:3,3]=j['xyz'];return T@M
