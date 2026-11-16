"""Bind a mixed VLA to the frozen Q geometry controller interface.

precision packets are geometric features, NEVER direct joint commands.
LearnedServo consumes those packets and performs the original bounded IK.
"""
from pathlib import Path
import json,hashlib
import numpy as np

class FrozenHybridAdapter:
 def __init__(self,base_policy,geometry,contract):
  self.base=base_policy;self.geometry=geometry;self.contract=contract
  self.low=np.asarray(contract['state_low']);self.high=np.asarray(contract['state_high']);self.arm=np.r_[0:6,7:13]
  assert self.low.shape==self.high.shape==(14,) and np.all(self.high>self.low)
  assert contract['policy_horizons']==dict(base=50,precision=1)
  self.reset()
 def reset(self):
  self.phase='base';self.calls=0;self.base_calls=0;self.precision_calls=0;self.stage_events=[];self.servo=None;self.geometry.reset()
 def request(self,role,obs):
  if role=='base':
   actions=np.asarray(self.base.infer({k:v for k,v in obs.items() if k!='observation/high_intrinsics'})['actions']);self.base_calls+=1
  elif role=='precision':
   actions=np.asarray(self.geometry.infer({k:obs[k] for k in ['observation/left_wrist_image','observation/right_wrist_image','observation/state']}));self.precision_calls+=1
  else:raise ValueError(role)
  assert actions.shape==(self.contract['policy_horizons'][role],14) and np.isfinite(actions).all()
  return actions
 def infer(self,observation,kinematic_gap_m):
  state=np.asarray(observation['observation/state']);normalized=2*(state-self.low)/(self.high-self.low)-1
  maximum=float(np.max(np.abs(normalized[self.arm])));control=self.calls*10
  if self.phase=='precision' and (maximum>1.0 or state[[6,13]].max()>.70):
   self.phase='base';self.stage_events.append(dict(control=control,event='fallback_base',normalized_arm_max=maximum))
  if self.phase=='precision':actions=self.request('precision',observation)
  else:
   actions=self.request('base',observation)
   if control>=150 and maximum<=.95 and state[[6,13]].max()<=.60 and np.max(actions[:10,[6,13]])<=.25 and kinematic_gap_m<=.055:
    self.phase='precision';self.stage_events.append(dict(control=control,event='enter_precision',normalized_arm_max=maximum,estimated_endpoint_m=kinematic_gap_m));actions=self.request('precision',observation)
  self.calls+=1;return actions
 def joint_actions(self,observation,control,robot_env,arr,gate,previous,limits,output):
  """The actual Q collector path: geometric packet -> original bounded IK."""
  if self.phase=='base':
   batch=self.request('base',observation);state=np.asarray(observation['observation/state'])
   normalized=2*(state-self.low)/(self.high-self.low)-1;maximum=float(np.max(np.abs(normalized[self.arm])))
   gap=gate.estimate(state,control)
   if control>=150 and maximum<=.95 and state[[6,13]].max()<=.60 and np.max(batch[:10,[6,13]])<=.25 and gap<=.055:
    from .learned_servo import LearnedServo
    self.phase='precision';self.servo=LearnedServo(robot_env,arr,self,gate,previous,limits,Path(output))
    self.stage_events.append(dict(control=control,event='enter_learned_visual_geometry_servo',estimated_endpoint_m=gap))
  if self.phase=='precision':batch=np.asarray(self.servo.infer(observation,control))
  assert batch.shape==((1 if self.phase=='precision' else 50),14) and np.isfinite(batch).all()
  return batch
 def close(self):return dict(base_calls=self.base_calls,precision_calls=self.precision_calls,stage_events=self.stage_events)
