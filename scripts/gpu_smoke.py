"""Small ROCm/PyTorch smoke test for the DQN compute path."""
import torch
from destroyer import DQNAgent, Game2048, encode

if not torch.cuda.is_available():
    raise SystemExit("PyTorch GPU is unavailable")

agent = DQNAgent(batch=2)
game = Game2048()
state = encode(game.board)
for i in range(2):
    agent.replay.add(state, i, 0.25, state, i == 1)
loss = agent.learn()
print(f"GPU={torch.cuda.get_device_name(0)} DEVICE={agent.device} LOSS={loss:.6f}")
assert agent.replay.states.is_cuda
assert agent.online.layers[0].weight.is_cuda
