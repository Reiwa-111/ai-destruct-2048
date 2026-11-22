from __future__ import annotations
import random
import os
import numpy as np
import torch
from torch import nn
from .game import ACTIONS, encode


def _resolve_device(device=None):
    """Resolve a device; an explicitly requested GPU must never become CPU."""

    requested = str(device or os.environ.get("REIWA_DEVICE") or "cpu").strip()
    normalized = requested.lower()
    if normalized.startswith("hip"):
        raise ValueError("PyTorch exposes AMD ROCm devices through the 'cuda' device name, not 'hip'.")
    if normalized.startswith("cuda"):
        try:
            available = torch.cuda.is_available()
        except Exception as error:
            raise RuntimeError(
                f"GPU device '{requested}' was explicitly requested, but the CUDA/HIP runtime probe failed. "
                "Refusing to silently fall back to CPU."
            ) from error
        if not available:
            raise RuntimeError(
                f"GPU device '{requested}' was explicitly requested, but PyTorch reports no available GPU. "
                "Refusing to silently fall back to CPU."
            )
        require_rocm = os.environ.get("REIWA_REQUIRE_ROCM", "").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if require_rocm and torch.version.hip is None:
            raise RuntimeError(
                "REIWA_REQUIRE_ROCM is enabled, but this PyTorch build is not compiled with ROCm/HIP."
            )
    return torch.device(requested)


class QNet(nn.Module):
    """Compact four-hidden-layer MLP retained as the DQN baseline."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 4),
        )
    def forward(self, x):
        return self.layers(x)


class LegacyQNet(nn.Module):
    """Previous oversized DQN architecture, kept only for old checkpoints."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        return self.layers(x)


class Replay:
    """Device-resident replay with next-state legal-action masks."""
    def __init__(self, capacity=200_000, device='cpu'):
        self.capacity = int(capacity); self.device = torch.device(device); self.size = 0; self.pos = 0
        self.states = torch.empty((capacity, 16), device=self.device, dtype=torch.float32)
        self.next_states = torch.empty_like(self.states)
        self.actions = torch.empty((capacity,), device=self.device, dtype=torch.int64)
        self.rewards = torch.empty((capacity,), device=self.device, dtype=torch.float32)
        self.dones = torch.empty((capacity,), device=self.device, dtype=torch.float32)
        self.next_masks = torch.empty((capacity, 4), device=self.device, dtype=torch.bool)
    def add(self, state, action, reward, next_state, done, next_mask=None):
        i = self.pos
        self.states[i] = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        self.next_states[i] = torch.as_tensor(next_state, device=self.device, dtype=torch.float32)
        self.actions[i] = int(action); self.rewards[i] = float(reward); self.dones[i] = float(done)
        if next_mask is None: next_mask = [True] * 4 if not done else [False] * 4
        self.next_masks[i] = torch.as_tensor(next_mask, device=self.device, dtype=torch.bool)
        self.pos = (i + 1) % self.capacity; self.size = min(self.size + 1, self.capacity)
    def sample(self, n):
        idx = torch.randint(self.size, (n,), device=self.device)
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx], self.next_masks[idx]
    def export(self):
        # Bulk device transfer avoids thousands of per-item GPU synchronizations.
        return {'states':self.states[:self.size].cpu(), 'actions':self.actions[:self.size].cpu(), 'rewards':self.rewards[:self.size].cpu(), 'next_states':self.next_states[:self.size].cpu(), 'dones':self.dones[:self.size].cpu(), 'next_masks':self.next_masks[:self.size].cpu()}
    def restore(self, items):
        self.size=0; self.pos=0
        if isinstance(items,dict):
            count=min(self.capacity,len(items['actions']))
            for key,dst in [('states',self.states),('actions',self.actions),('rewards',self.rewards),('next_states',self.next_states),('dones',self.dones),('next_masks',self.next_masks)]:
                dst[:count]=torch.as_tensor(items[key][-count:],device=self.device,dtype=dst.dtype) if count else dst[:0]
            self.size=count; self.pos=count%self.capacity
            return
        for item in items[-self.capacity:]:
            if len(item)==5: self.add(*item)
            else: self.add(*item[:5],next_mask=item[5])
    def __len__(self): return self.size


class DQNAgent:
    def __init__(self, device=None, lr=1e-5, gamma=.99, batch=128, warmup=5_000, replay_capacity=200_000, epsilon_start=1.0, epsilon_end=.05, epsilon_decay_steps=1_500_000, tau=.005):
        # Do not call torch.cuda.is_available() implicitly: on some WSL/ROCm
        # installations its lazy HIP probe can hang during GUI startup.
        self.device = _resolve_device(device)
        self.online = QNet().to(self.device); self.target = QNet().to(self.device); self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.model_architecture = 'mlp_16_128x4_4_v1'
        self.gamma=float(gamma); self.batch=int(batch); self.warmup=int(warmup)
        self.replay=Replay(capacity=replay_capacity,device=self.device); self.steps=0; self.environment_steps=0; self.epsilon=float(epsilon_start)
        self.epsilon_start=float(epsilon_start); self.epsilon_end=float(epsilon_end); self.epsilon_decay_steps=int(epsilon_decay_steps)
        self.tau=float(tau); self.loss_value=0.0; self.loss_avg=0.0; self.sync_every=0
        self.td_error_mean=0.0; self.td_error_max=0.0; self.q_mean=0.0; self.target_mean=0.0; self.grad_norm=0.0
    @torch.no_grad()
    def q_values(self, board): return self.online(torch.tensor(encode(board),device=self.device).unsqueeze(0)).squeeze(0).cpu().numpy()
    def legal_mask(self, legal):
        mask = np.zeros(len(ACTIONS), dtype=bool)
        for action in legal: mask[ACTIONS.index(action)] = True
        return mask
    def q_values_masked(self, board, legal):
        """Return raw Q values plus a boolean mask of legal actions."""
        return self.q_values(board), self.legal_mask(legal)
    def act(self, board, legal, explore=True):
        if not legal: raise ValueError('Cannot select an action from a terminal board')
        if explore and random.random() < self.epsilon: return ACTIONS.index(random.choice(legal))
        q=self.q_values(board); valid=[ACTIONS.index(a) for a in legal]
        return max(valid,key=lambda i:q[i])
    def set_learning_rate(self, lr):
        if lr <= 0: raise ValueError('learning rate must be positive')
        for group in self.optimizer.param_groups: group['lr'] = float(lr)
    @property
    def learning_rate(self):
        return float(self.optimizer.param_groups[0]['lr'])
    def stats(self):
        return {'device':str(self.device),'architecture':self.model_architecture,'steps':self.steps,'environment_steps':self.environment_steps,'epsilon':self.epsilon,'loss_avg':self.loss_avg,'td_error_mean':self.td_error_mean,'td_error_max':self.td_error_max,'grad_norm':self.grad_norm,'replay':len(self.replay),'learning_rate':self.learning_rate}
    def update_epsilon(self):
        frac=min(1.0,self.environment_steps/max(1,self.epsilon_decay_steps))
        self.epsilon=self.epsilon_start+(self.epsilon_end-self.epsilon_start)*frac
    def learn(self, force=False):
        if len(self.replay)<max(self.batch,self.warmup) and not force: return None
        if len(self.replay)<self.batch: return None
        s,a,r,ns,d,next_mask=self.replay.sample(self.batch)
        q=self.online(s).gather(1,a[:,None]).squeeze(1)
        with torch.no_grad():
            next_online=self.online(ns).masked_fill(~next_mask,-torch.inf)
            # Terminal states have no valid actions; prevent all -inf argmax issues.
            next_online[d.bool()] = 0
            next_actions=next_online.argmax(1)
            next_q=self.target(ns).gather(1,next_actions[:,None]).squeeze(1)
            next_q[d.bool()]=0
            target=r+self.gamma*(1-d)*next_q
        td_error=target-q.detach()
        self.td_error_mean=float(td_error.abs().mean().item())
        self.td_error_max=float(td_error.abs().max().item())
        self.q_mean=float(q.detach().mean().item())
        self.target_mean=float(target.mean().item())
        loss=nn.functional.smooth_l1_loss(q,target)
        self.optimizer.zero_grad(set_to_none=True); loss.backward()
        self.grad_norm=float(nn.utils.clip_grad_norm_(self.online.parameters(),5.0).item())
        self.optimizer.step()
        with torch.no_grad():
            for target_param, online_param in zip(self.target.parameters(),self.online.parameters()):
                target_param.lerp_(online_param,self.tau)
        self.steps+=1; self.loss_value=float(loss.item()); self.loss_avg=self.loss_value if self.steps==1 else .98*self.loss_avg+.02*self.loss_value
        self.update_epsilon()
        return self.loss_value
    def save(self,path,episodes=0,include_replay=True):
        torch.save({'model':self.online.state_dict(),'target':self.target.state_dict(),'optimizer':self.optimizer.state_dict(),'episodes':episodes,'steps':self.steps,'environment_steps':self.environment_steps,'epsilon':self.epsilon,'replay':self.replay.export() if include_replay else None,'python_rng':random.getstate(),'architecture':self.model_architecture,'config':{'batch':self.batch,'warmup':self.warmup,'gamma':self.gamma,'tau':self.tau,'epsilon_decay_steps':self.epsilon_decay_steps}},path)
    def load(self,path):
        data=torch.load(path,map_location=self.device,weights_only=False)
        model_state=data['model']
        first_weight=model_state.get('layers.0.weight')
        if first_weight is not None and tuple(first_weight.shape)==(256,16):
            self.online=LegacyQNet().to(self.device)
            self.target=LegacyQNet().to(self.device)
            self.model_architecture='legacy_mlp_16_256_512_512_256_4'
        elif first_weight is not None and tuple(first_weight.shape)==(128,16):
            # A compact checkpoint must be loadable even when the live agent
            # currently holds the legacy architecture (and vice versa).
            self.online=QNet().to(self.device)
            self.target=QNet().to(self.device)
            self.model_architecture='mlp_16_128x4_4_v1'
        self.online.load_state_dict(model_state)
        self.target.load_state_dict(data.get('target',model_state))
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=self.optimizer.param_groups[0]['lr'])
        if data.get('optimizer') is not None:
            self.optimizer.load_state_dict(data['optimizer'])
        self.steps=data.get('steps',0); self.environment_steps=data.get('environment_steps',self.steps); self.epsilon=data.get('epsilon',.1)
        cfg=data.get('config',{}); self.epsilon_decay_steps=cfg.get('epsilon_decay_steps',self.epsilon_decay_steps); self.tau=cfg.get('tau',self.tau); self.gamma=cfg.get('gamma',self.gamma); self.batch=cfg.get('batch',self.batch); self.warmup=cfg.get('warmup',self.warmup)
        if data.get('replay') is not None: self.replay.restore(data['replay'])
        if data.get('python_rng') is not None: random.setstate(data['python_rng'])
        return data.get('episodes',0)
