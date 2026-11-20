from __future__ import annotations
import random, collections
import numpy as np
import torch
from torch import nn
from .game import ACTIONS, encode


class QNet(nn.Module):
    """Four linear layers, four Q-value outputs."""
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(16,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,4))
    def forward(self,x): return self.layers(x)

class Replay:
    def __init__(self,capacity=100_000): self.data=collections.deque(maxlen=capacity)
    def add(self,*transition): self.data.append(transition)
    def sample(self,n): return random.sample(self.data,n)
    def __len__(self): return len(self.data)

class DQNAgent:
    def __init__(self,device=None,lr=1e-4,gamma=.99,batch=128):
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.online=QNet().to(self.device); self.target=QNet().to(self.device); self.target.load_state_dict(self.online.state_dict())
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=lr)
        self.gamma=gamma; self.batch=batch; self.replay=Replay(); self.steps=0; self.epsilon=1.0
        self.loss_value=0.0; self.sync_every=1000
    @torch.no_grad()
    def q_values(self,board): return self.online(torch.tensor(encode(board),device=self.device).unsqueeze(0)).squeeze(0).cpu().numpy()
    def act(self,board,legal,explore=True):
        if not legal: return random.randrange(4)
        if explore and random.random()<self.epsilon: return ACTIONS.index(random.choice(legal))
        q=self.q_values(board); valid=[ACTIONS.index(a) for a in legal]
        return max(valid,key=lambda i:q[i])
    def learn(self):
        if len(self.replay)<self.batch: return None
        batch=self.replay.sample(self.batch)
        states,actions,rewards,next_states,dones=zip(*batch)
        s=torch.tensor(np.stack(states),device=self.device); a=torch.tensor(actions,device=self.device).long(); r=torch.tensor(rewards,device=self.device,dtype=torch.float32); ns=torch.tensor(np.stack(next_states),device=self.device); d=torch.tensor(dones,device=self.device,dtype=torch.float32)
        q=self.online(s).gather(1,a[:,None]).squeeze(1)
        with torch.no_grad(): target=r+self.gamma*(1-d)*self.target(ns).max(1).values
        loss=nn.functional.smooth_l1_loss(q,target)
        self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.online.parameters(),10); self.optimizer.step()
        self.steps+=1
        if self.steps%self.sync_every==0: self.target.load_state_dict(self.online.state_dict())
        self.loss_value=float(loss.item()); return self.loss_value
    def save(self,path,episodes=0): torch.save({'model':self.online.state_dict(),'target':self.target.state_dict(),'optimizer':self.optimizer.state_dict(),'episodes':episodes,'steps':self.steps,'epsilon':self.epsilon},path)
    def load(self,path):
        data=torch.load(path,map_location=self.device,weights_only=False); self.online.load_state_dict(data['model']); self.target.load_state_dict(data.get('target',data['model'])); self.optimizer.load_state_dict(data['optimizer']); self.steps=data.get('steps',0); self.epsilon=data.get('epsilon',.1); return data.get('episodes',0)

