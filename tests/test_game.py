import unittest
from destroyer.game import Game2048, apply_move, encode, spawn_outcomes
from destroyer.agent import QNet
import torch
class GameTests(unittest.TestCase):
    def test_pair_merges_once(self):
        g=Game2048(); g.board=[[2,2,2,2],[0]*4,[0]*4,[0]*4]
        s=g.move('LEFT'); self.assertEqual(s.score_delta,8); self.assertEqual(s.merges,2)
        self.assertIn(sum(sum(row) for row in g.board)-8,(2,4))
        self.assertEqual(sum(value != 0 for row in g.board for value in row),3)
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
    def test_afterstate_is_deterministic_and_does_not_spawn(self):
        board=((2,2,4,0),(0,0,0,0),(0,0,0,0),(0,0,0,0))
        result=apply_move(board,'LEFT')
        self.assertEqual(result.board[0],(4,4,0,0))
        self.assertEqual(result.score_delta,4)
        self.assertEqual(result.merges,1)
        self.assertTrue(result.moved)
        self.assertEqual(board[0],(2,2,4,0))
    def test_merge_conserves_tile_mass_and_never_merges_a_tile_twice(self):
        board=((2,2,2,2),(4,4,8,8),(0,0,0,0),(0,0,0,0))
        for action in ('LEFT','RIGHT','UP','DOWN'):
            result=apply_move(board,action)
            self.assertEqual(sum(map(sum,result.board)),sum(map(sum,board)))
        horizontal=apply_move(board,'LEFT')
        self.assertEqual(horizontal.board[0],(4,4,0,0))
        self.assertEqual(horizontal.board[1],(8,16,0,0))
        self.assertEqual(horizontal.merges,4)
    def test_board_validation_rejects_non_power_of_two_tiles(self):
        with self.assertRaises(ValueError): apply_move(((1,0,0,0),)+((0,0,0,0),)*3,'LEFT')
        with self.assertRaises(ValueError): apply_move(((3,0,0,0),)+((0,0,0,0),)*3,'LEFT')
    def test_encode_preserves_ranks_above_32768(self):
        board=[[0]*4 for _ in range(4)]
        board[0][0]=32768
        board[0][1]=65536
        board[0][2]=131072
        values=encode(board)
        self.assertAlmostEqual(float(values[0]),15/15)
        self.assertAlmostEqual(float(values[1]),16/15,delta=1e-6)
        self.assertAlmostEqual(float(values[2]),17/15,delta=1e-6)
        self.assertEqual(len({float(values[i]) for i in range(3)}),3)
    def test_spawn_outcomes_are_exactly_normalized(self):
        board=((2,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0))
        outcomes=spawn_outcomes(board)
        self.assertEqual(len(outcomes),30)
        self.assertAlmostEqual(sum(prob for prob,_ in outcomes),1.0)
        self.assertEqual(sum(prob for prob,state in outcomes if 4 in sum((list(r) for r in state),[])),0.1)
    def test_four_hidden_layer_network_shape_and_parameter_count(self):
        model=QNet(); self.assertEqual(model(torch.zeros(1,16)).shape,(1,4)); self.assertEqual(sum(p.numel() for p in model.parameters()),52228)
if __name__=='__main__': unittest.main()
