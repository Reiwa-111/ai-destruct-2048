from __future__ import annotations
import random
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
    """Fixed-size replay storage on the selected compute device."""
    def __init__(self,capacity=100_000,device='cpu'):
        self.capacity=capacity; self.device=torch.device(device); self.size=0; self.pos=0
        self.states=torch.empty((capacity,16),device=self.device,dtype=torch.float32)
        self.next_states=torch.empty_like(self.states)
        self.actions=torch.empty((capacity,),device=self.device,dtype=torch.int64)
        self.rewards=torch.empty((capacity,),device=self.device,dtype=torch.float32)
        self.dones=torch.empty((capacity,),device=self.device,dtype=torch.float32)
    def add(self,state,action,reward,next_state,done):
        i=self.pos
        self.states[i]=torch.as_tensor(state,device=self.device,dtype=torch.float32)
        self.next_states[i]=torch.as_tensor(next_state,device=self.device,dtype=torch.float32)
        self.actions[i]=action; self.rewards[i]=reward; self.dones[i]=float(done)
        self.pos=(i+1)%self.capacity; self.size=min(self.size+1,self.capacity)
    def sample(self,n):
        idx=torch.randint(self.size,(n,),device=self.device)
        return self.states[idx],self.actions[idx],self.rewards[idx],self.next_states[idx],self.dones[idx]
    def export(self):
        return [(self.states[i].cpu().numpy(),int(self.actions[i].item()),float(self.rewards[i].item()),self.next_states[i].cpu().numpy(),bool(self.dones[i].item())) for i in range(self.size)]
    def restore(self,items):
        self.size=0; self.pos=0
        for transition in items[-self.capacity:]: self.add(*transition)
    def __len__(self): return self.size

class DQNAgent:
    def __init__(self,device=None,lr=1e-4,gamma=.99,batch=128):
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.online=QNet().to(self.device); self.target=QNet().to(self.device); self.target.load_state_dict(self.online.state_dict())
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=lr)
        self.gamma=gamma; self.batch=batch; self.replay=Replay(device=self.device); self.steps=0; self.epsilon=1.0
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
        s,a,r,ns,d=self.replay.sample(self.batch)
        q=self.online(s).gather(1,a[:,None]).squeeze(1)
        with torch.no_grad(): target=r+self.gamma*(1-d)*self.target(ns).max(1).values
        loss=nn.functional.smooth_l1_loss(q,target)
        self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.online.parameters(),10); self.optimizer.step()
        self.steps+=1
        if self.steps%self.sync_every==0: self.target.load_state_dict(self.online.state_dict())
        self.loss_value=float(loss.item()); return self.loss_value
    def save(self,path,episodes=0,include_replay=True):
        torch.save({'model':self.online.state_dict(),'target':self.target.state_dict(),'optimizer':self.optimizer.state_dict(),'episodes':episodes,'steps':self.steps,'epsilon':self.epsilon,'replay':self.replay.export() if include_replay else None,'python_rng':random.getstate()},path)
    def load(self,path):
        data=torch.load(path,map_location=self.device,weights_only=False); self.online.load_state_dict(data['model']); self.target.load_state_dict(data.get('target',data['model'])); self.optimizer.load_state_dict(data['optimizer']); self.steps=data.get('steps',0); self.epsilon=data.get('epsilon',.1)
        if data.get('replay') is not None: self.replay.restore(data['replay'])
        if data.get('python_rng') is not None: random.setstate(data['python_rng'])
        return data.get('episodes',0)

