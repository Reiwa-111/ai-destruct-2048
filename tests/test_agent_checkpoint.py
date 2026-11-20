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
    assert restored.replay.data[0][1] == 0
