from pathlib import Path
import numpy as np, torch, cv2
from .wrist_model import WristPose
from .robot_fk import link6
from .edge_refinement import fit
import json,time

class LearnedGeometry:
 def __init__(self, root, log_directory=None):
  self.log_directory=Path(log_directory) if log_directory is not None else None
  if self.log_directory is not None:self.log_directory.mkdir(parents=True,exist_ok=True)
  torch.set_num_threads(2)
  self.c=torch.load(Path(root)/'full_fit_1000.pt',map_location='cpu',weights_only=False)
  assert self.c['full_pool_fit'] and self.c['step']==1000
  self.model=WristPose().eval();self.model.load_state_dict(self.c['model']);self.reset()

 def reset(self):
  self.filtered=None;self.count=0

 @torch.no_grad()
 def infer(self,obs):
  parts=[]
  for key in ['observation/left_wrist_image','observation/right_wrist_image']:
   image=np.asarray(obs[key]);assert image.shape==(480,640,3)
   padded=np.zeros((256,256,3),np.uint8)
   padded[32:224]=cv2.resize(image,(256,192),interpolation=cv2.INTER_AREA)
   parts.append(padded.transpose(2,0,1))
  image=torch.tensor(np.concatenate(parts)[None],dtype=torch.float32)/127.5-1
  state=np.asarray(obs['observation/state']);assert state.shape==(14,)
  opening=torch.tensor(((state[[6,13]]-self.c['om'])/self.c['oscale'])[None],dtype=torch.float32)
  local=(self.model(image,opening).numpy()[0]*self.c['ys']+self.c['ym']).reshape(2,6)
  diagnostics=[]
  for i,key in enumerate(['observation/left_wrist_image','observation/right_wrist_image']):
   refined,record=fit(np.asarray(obs[key]),local[i],i)
   assert record['final']<8 and record['samples']>=40,record
   local[i]=refined;diagnostics.append(record)
  if self.log_directory is not None:
   with (self.log_directory/'edge_inference_trace.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),fits=diagnostics))+'\n')
  measured=local.copy()
  if self.filtered is None:self.filtered=local.copy()
  else:
   dp=local[:,:3]-self.filtered[:,:3];dp*=np.minimum(1,.002/np.maximum(np.linalg.norm(dp,axis=1,keepdims=True),1e-9))
   self.filtered[:,:3]+=.2*dp
   for i in range(2):
    a=self.filtered[i,3:];a/=np.linalg.norm(a);b=local[i,3:];b/=np.linalg.norm(b);angle=np.arccos(np.clip(a@b,-1,1));alpha=.2*min(1,np.deg2rad(1)/max(angle,1e-9));c=(1-alpha)*a+alpha*b;self.filtered[i,3:]=c/np.linalg.norm(c)
  self.count+=1;local=self.filtered.copy()
  if self.log_directory is not None:
   with (self.log_directory/'tracking_trace.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),count=self.count,measured=measured.tolist(),filtered=local.tolist()))+'\n')
  # Both arms share the configured robot base translation, which cancels in tip delta.
  transforms=np.stack([link6(state,side) for side in ['left','right']])
  tips=np.einsum('hij,hj->hi',transforms[:,:3,:3],local[:,:3])+transforms[:,:3,3]
  axes=np.einsum('hij,hj->hi',transforms[:,:3,:3],local[:,3:])
  axes/=np.maximum(np.linalg.norm(axes,axis=-1,keepdims=True),1e-9)
  packet=np.zeros((1,14),np.float32);packet[0,:9]=np.r_[tips[0]-tips[1],axes[1],axes[0]]
  assert np.isfinite(packet).all()
  return packet
