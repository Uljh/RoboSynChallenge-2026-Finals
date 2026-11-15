"""Exercise a downloaded model through its portable observation interface."""
import argparse,json
from pathlib import Path
import numpy as np
from deploy_policy import get_model

def main():
    p=argparse.ArgumentParser();p.add_argument('task');p.add_argument('checkpoint');p.add_argument('--platform',default='cpu');a=p.parse_args()
    model=get_model({'task_name':a.task,'checkpoint_path':a.checkpoint,'jax_platform':a.platform})
    observation={'robot':{'qpos':np.zeros((1,14),np.float32)},'sensor':{
        name:{'color':np.zeros((1,224,224,3),np.uint8)} for name in ['cam_high','cam_left_wrist','cam_right_wrist']}}
    noise=np.random.default_rng(20261116).normal(size=(50,32)).astype(np.float32)
    result=model.infer(observation,noise=noise)
    print(json.dumps({'task':a.task,'shape':list(result.shape),'finite':bool(np.isfinite(result).all()),'scope':'synthetic interface smoke only; not a task success measurement'}))

if __name__=='__main__':main()
