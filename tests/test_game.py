import unittest
from destroyer.game import Game2048

class GameTests(unittest.TestCase):
    def test_pair_merges_once(self):
        g=Game2048(); g.board=[[2,2,2,2],[0]*4,[0]*4,[0]*4]
        s=g.move('LEFT'); self.assertEqual(s.score_delta,8); self.assertEqual(s.merges,2); self.assertEqual(sum(g.board[0]),8)
    def test_invalid_move_no_spawn(self):
        g=Game2048(); g.board=[[2,0,0,0],[0]*4,[0]*4,[0]*4]; before=[r[:] for r in g.board]
        s=g.move('LEFT'); self.assertFalse(s.moved); self.assertEqual(g.board,before)
    def test_game_over_full_no_merges(self):
        g=Game2048(); g.board=[[2,4,8,16],[32,64,128,256],[512,1024,2,4],[8,16,32,64]]
        self.assertTrue(g.game_over())
    def test_legal_actions(self):
        g=Game2048(); g.board=[[2,0,0,0],[0]*4,[0]*4,[0]*4]
        self.assertEqual(g.legal_actions(),['DOWN','RIGHT']); self.assertEqual(g.score,0); self.assertEqual(g.moves,0)
if __name__=='__main__': unittest.main()

