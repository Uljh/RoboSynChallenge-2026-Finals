from pathlib import Path
from functools import lru_cache
import cv2,numpy as np,trimesh
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
asset_root=Path(__file__).resolve().parent/'assets'
Rcam=Rotation.from_quat([.6797,-.6797,-.1949,.1949]).as_matrix()@np.diag([1,-1,-1]);tcam=np.array([-.12,.01,.03]);intr=np.array([603.117004,602.418640,326.727478,251.93399])
@lru_cache(maxsize=2)
def assets(side):
 name='1' if side==0 else '2';scale=np.array([2.1,2,2] if side==0 else [1.8,1.7,1.7]);mesh=trimesh.load(asset_root/'guijiao_standard'/f'{name}guijiao_standard.ply',process=False);mesh.merge_vertices();mesh.fix_normals()
 return np.asarray(mesh.vertices)*scale,mesh.faces,mesh.face_adjacency_edges,mesh.face_adjacency

def rotation(axis):
 a=axis/np.linalg.norm(axis);b=np.cross(a,[0,0,1]);b/=np.linalg.norm(b);return np.stack([a,b,np.cross(a,b)],1)

def fit(image,local,side):
 v,faces,edge,adj=assets(side);v=v.copy()
 tip=v[:,0].max() if side==0 else v[:,0].min();v[:,0]-=tip
 init=rotation(local[3:]);center=local[:3].copy()
 gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY);edges=cv2.Canny(cv2.GaussianBlur(gray,(3,3),.7),25,65);distance=cv2.distanceTransform(255-edges,cv2.DIST_L2,5).astype(np.float64)
 def project(points):
  cam=(points-tcam)@Rcam;uv=cam[:,:2]/cam[:,2:]*intr[:2]+intr[2:];return uv,cam[:,2]
 # Freeze silhouette edge identities around the learned initial pose, preserving smooth derivatives.
 world=v@init.T+center;camera=(world-tcam)@Rcam;tri=camera[faces];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);front=np.einsum('ij,ij->i',normal,tri.mean(1))<0
 chosen=edge[front[adj[:,0]]!=front[adj[:,1]]]
 # Add sharp visible ring edges, excluding back-facing interiors.
 norm=normal/np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
 sharp=(np.sum(norm[adj[:,0]]*norm[adj[:,1]],axis=1)<.7)&(front[adj[:,0]]|front[adj[:,1]])
 chosen=np.unique(np.sort(np.concatenate([chosen,edge[sharp]]),axis=1),axis=0)
 samples=(v[chosen[:,0],None,:]*(1-np.linspace(0,1,6)[None,:,None])+v[chosen[:,1],None,:]*np.linspace(0,1,6)[None,:,None]).reshape(-1,3)
 uv,depth=project(samples@init.T+center);valid=(depth>.01)&(uv[:,0]>12)&(uv[:,0]<628)&(uv[:,1]>12)&(uv[:,1]<468)
 samples=samples[valid]
 # Ray/triangle self-occlusion at the initial estimate, no scene truth or segmentation.
 rays=(samples@init.T+center-tcam)@Rcam;visible=[];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0];tv=-tri[:,0];qv=np.cross(tv,e1)
 for ray in rays:
  h=np.cross(ray,e2);det=np.sum(e1*h,axis=1);inv=np.divide(1,det,out=np.zeros_like(det),where=np.abs(det)>1e-12);u=np.sum(tv*h,axis=1)*inv;vv=np.sum(ray*qv,axis=1)*inv;tt=np.sum(e2*qv,axis=1)*inv;hit=(np.abs(det)>1e-12)&(u>=-1e-5)&(vv>=-1e-5)&(u+vv<=1+1e-5)&(tt>0)&(tt<.9995);visible.append(not hit.any())
 samples=samples[np.array(visible)];assert len(samples)>20,(side,len(samples))
 def calc(x):return samples@((Rotation.from_rotvec(x[3:]*.01).as_matrix()@init).T)+center+x[:3]*.001
 def residual(x):
  uv,_=project(calc(x));d=map_coordinates(distance,[uv[:,1],uv[:,0]],order=1,mode='constant',cval=15)
  return np.r_[d,x*.02]
 def jacobian(x):
  return np.stack([(residual(x+np.eye(6)[i]*.05)-residual(x-np.eye(6)[i]*.05))/.1 for i in range(6)],1)
 result=least_squares(residual,np.zeros(6),bounds=(-np.array([8,8,8,12,12,12]),np.array([8,8,8,12,12,12])),loss='soft_l1',f_scale=2,jac=jacobian,max_nfev=100)
 refined=np.r_[center+result.x[:3]*.001,(Rotation.from_rotvec(result.x[3:]*.01).as_matrix()@init)[:,0]]

 return refined,dict(initial=float(np.mean(residual(np.zeros(6))[:len(samples)])),final=float(np.mean(residual(result.x)[:len(samples)])),delta=result.x.tolist(),samples=len(samples))
