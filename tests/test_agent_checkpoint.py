from destroyer.agent import DQNAgent
from destroyer.game import Game2048, encode


def test_checkpoint_restores_training_state_and_replay(tmp_path):
    agent = DQNAgent(device='cpu', batch=2)
    game = Game2048()
    state = encode(game.board)
    agent.replay.add(state, 0, 1.0, state, False)
    agent.replay.add(state, 1, -0.2, state, True)
    agent.steps = 17
    agent.epsilon = 0.37
    path = tmp_path / 'checkpoint.pt'
    agent.save(path, episodes=9)

    restored = DQNAgent(device='cpu', batch=2)
    episodes = restored.load(path)
    assert episodes == 9
    assert restored.steps == 17
    assert restored.epsilon == 0.37
    assert len(restored.replay) == 2
    assert restored.replay.actions[0].item() == 0
    assert restored.replay.states.device.type == 'cpu'


def test_replay_samples_tensor_batch_on_selected_device():
    agent = DQNAgent(device='cpu', batch=2)
    game = Game2048()
    state = encode(game.board)
    agent.replay.add(state, 0, 0.5, state, False)
    agent.replay.add(state, 1, -1.0, state, True)
    batch = agent.replay.sample(2)
    assert [tensor.shape for tensor in batch] == [(2, 16), (2,), (2,), (2, 16), (2,)]
    assert all(t.device.type == 'cpu' for t in batch)
