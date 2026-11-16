import torch
from torch import nn
def encoder():
 blocks=[];c=5
 for n in [24,48,64,96]:
  blocks.extend([nn.Conv2d(c,n,3,2,1),nn.GroupNorm(8,n),nn.SiLU(),nn.Conv2d(n,n,3,1,1),nn.GroupNorm(8,n),nn.SiLU()]);c=n
 return nn.Sequential(*blocks,nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(96*16,256),nn.SiLU())
class WristPose(nn.Module):
 def __init__(self):
  super().__init__();self.enc=encoder();self.heads=nn.ModuleList([nn.Sequential(nn.Linear(257,128),nn.SiLU(),nn.Linear(128,6)) for _ in range(2)]);q=torch.linspace(-1,1,256);yy,xx=torch.meshgrid(q,q,indexing='ij');self.register_buffer('coords',torch.stack([xx,yy])[None])
 def forward(self,image,opening):
  n=image.shape[0];x=torch.cat([image[:,:3],image[:,3:]],0);features=self.enc(torch.cat([x,self.coords.expand(n*2,-1,-1,-1)],1));return torch.cat([h(torch.cat([features[i*n:(i+1)*n],opening[:,i:i+1]],1)) for i,h in enumerate(self.heads)],1)
