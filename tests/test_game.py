import unittest
from destroyer.game import Game2048
from destroyer.agent import QNet
import torch
class GameTests(unittest.TestCase):
    def test_pair_merges_once(self):
        g=Game2048(); g.board=[[2,2,2,2],[0]*4,[0]*4,[0]*4]
        s=g.move('LEFT'); self.assertEqual(s.score_delta,8); self.assertEqual(s.merges,2); self.assertEqual(sum(g.board[0]),8)
    def test_invalid_move_no_spawn(self):
        g=Game2048(); g.board=[[2,0,0,0],[0]*4,[0]*4,[0]*4]; before=[r[:] for r in g.board]; rng=g.rng.getstate()
        s=g.move('LEFT'); self.assertFalse(s.moved); self.assertEqual(g.board,before); self.assertEqual(g.rng.getstate(),rng)
    def test_legal_actions_are_side_effect_free(self):
        g=Game2048(); g.board=[[2,0,0,0],[0]*4,[0]*4,[0]*4]; rng=g.rng.getstate()
        self.assertEqual(g.legal_actions(),['DOWN','RIGHT']); self.assertEqual(g.score,0); self.assertEqual(g.moves,0); self.assertEqual(g.rng.getstate(),rng)
    def test_game_over_full_no_merges(self):
        g=Game2048(); g.board=[[2,4,8,16],[32,64,128,256],[512,1024,2,4],[8,16,32,64]]
        self.assertTrue(g.game_over())
    def test_legal_actions_include_vertical_moves(self):
        g=Game2048(); g.board=[[2,0,0,0],[2,0,0,0],[0]*4,[0]*4]
        self.assertIn('UP',g.legal_actions()); self.assertIn('DOWN',g.legal_actions())
    def test_four_layer_network_shape_and_parameter_count(self):
        model=QNet(); self.assertEqual(model(torch.zeros(1,16)).shape,(1,4)); self.assertEqual(sum(p.numel() for p in model.parameters()),27204)
if __name__=='__main__': unittest.main()
