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
    assert [tensor.shape for tensor in batch] == [(2, 16), (2,), (2,), (2, 16), (2,), (2, 4)]
    assert all(t.device.type == 'cpu' for t in batch)


def test_replay_restores_legal_masks_and_env_step_count(tmp_path):
    agent = DQNAgent(device='cpu', batch=2, warmup=0)
    state = encode(Game2048().board)
    mask = [True, False, True, False]
    agent.replay.add(state, 2, 0.25, state, False, mask)
    agent.environment_steps = 1234
    path = tmp_path / 'mask_checkpoint.pt'
    agent.save(path, episodes=3)
    restored = DQNAgent(device='cpu', batch=2, warmup=0)
    restored.load(path)
    assert restored.environment_steps == 1234
    assert restored.replay.next_masks[0].tolist() == mask


def test_double_dqn_learn_runs_with_terminal_mask():
    agent = DQNAgent(device='cpu', batch=2, warmup=0)
    game = Game2048(); state = encode(game.board)
    agent.replay.add(state, 0, 1.0, state, False, [True, False, False, True])
    agent.replay.add(state, 1, -1.0, state, True, [False, False, False, False])
    assert agent.learn(force=True) is not None
    assert agent.steps == 1
    assert agent.loss_value >= 0


def test_dqn_baseline_uses_requested_compact_configuration():
    agent = DQNAgent(device='cpu')
    assert agent.batch == 128
    assert agent.optimizer.param_groups[0]['lr'] == 1e-5
    assert sum(parameter.numel() for parameter in agent.online.parameters()) == 52_228


def test_dqn_default_startup_does_not_probe_rocm(monkeypatch):
    import torch

    def forbidden_probe():
        raise AssertionError('implicit GPU availability probe must not run')

    monkeypatch.setattr(torch.cuda, 'is_available', forbidden_probe)
    agent = DQNAgent()
    assert agent.device.type == 'cpu'


def test_old_oversized_checkpoint_remains_loadable(tmp_path):
    import torch
    from destroyer.agent import LegacyQNet

    agent = DQNAgent(device='cpu', batch=2)
    legacy = LegacyQNet()
    path_state = legacy.state_dict()
    assert path_state['layers.0.weight'].shape == (256, 16)

    # Build a structurally valid old-format checkpoint without depending on
    # a user's mutable models/ directory.
    path = tmp_path / 'legacy.pt'
    optimizer = torch.optim.Adam(legacy.parameters(), lr=1e-4)
    torch.save({'model': path_state, 'target': path_state,
                'optimizer': optimizer.state_dict(), 'episodes': 4,
                'steps': 10, 'epsilon': 0.2}, path)
    restored = DQNAgent(device='cpu')
    assert restored.load(path) == 4
    assert restored.model_architecture == 'legacy_mlp_16_256_512_512_256_4'
    assert restored.online(torch.zeros(1, 16)).shape == (1, 4)
