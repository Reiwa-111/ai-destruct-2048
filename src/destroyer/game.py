from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Callable

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")

@dataclass
class Step:
    board: list[list[int]]
    score_delta: int
    moved: bool
    game_over: bool
    merges: int

class Game2048:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.board = [[0]*4 for _ in range(4)]
        self.score = 0
        self.moves = 0
        self.merges = 0
        self.reset()

    def reset(self):
        self.board = [[0]*4 for _ in range(4)]
        self.score = self.moves = self.merges = 0
        self.spawn(); self.spawn()
        return self.board

    def spawn(self):
        empty = [(r,c) for r in range(4) for c in range(4) if not self.board[r][c]]
        if not empty: return False
        r,c = self.rng.choice(empty)
        self.board[r][c] = 4 if self.rng.random() < .1 else 2
        return True

    @staticmethod
    def _line(line):
        packed = [x for x in line if x]
        out=[]; gained=0; merges=0; i=0
        while i < len(packed):
            if i+1 < len(packed) and packed[i] == packed[i+1]:
                v=packed[i]*2; out.append(v); gained+=v; merges+=1; i+=2
            else: out.append(packed[i]); i+=1
        return out+[0]*(4-len(out)), gained, merges

    def move(self, action: str):
        action=action.upper()
        if action not in ACTIONS: raise ValueError(f"Unknown action: {action}")
        old=[row[:] for row in self.board]
        gained=merges=0
        if action in ("LEFT","RIGHT"):
            for r in range(4):
                line=self.board[r] if action=="LEFT" else self.board[r][::-1]
                out,sc,m=self._line(line)
                self.board[r]=out if action=="LEFT" else out[::-1]
                gained+=sc; merges+=m
        else:
            for c in range(4):
                line=[self.board[r][c] for r in range(4)]
                if action=="DOWN": line.reverse()
                out,sc,m=self._line(line)
                if action=="DOWN": out.reverse()
                for r in range(4): self.board[r][c]=out[r]
                gained+=sc; merges+=m
        moved=self.board!=old
        if moved:
            self.score+=gained; self.moves+=1; self.merges+=merges; self.spawn()
        return Step([row[:] for row in self.board],gained,moved,self.game_over(),merges)

    def legal_actions(self):
        return [a for a in ACTIONS if self._simulate(a)!=self.board]
    def _simulate(self, action):
        old=self.board; self.board=[row[:] for row in old]; score=self.score; moves=self.moves; merges=self.merges; rng_state=self.rng.getstate()
        self.move(action); result=[row[:] for row in self.board]; self.board=old; self.score=score; self.moves=moves; self.merges=merges; self.rng.setstate(rng_state)
        return result
    def game_over(self):
        if any(not x for row in self.board for x in row): return False
        for r in range(4):
            for c in range(4):
                if r<3 and self.board[r][c]==self.board[r+1][c]: return False
                if c<3 and self.board[r][c]==self.board[r][c+1]: return False
        return True
    @property
    def max_tile(self): return max(max(row) for row in self.board)
    @property
    def empty_count(self): return sum(x==0 for row in self.board for x in row)

def encode(board):
    import numpy as np
    return np.asarray([0.0 if v==0 else min(15, v.bit_length()-1) / 15.0 for row in board for v in row],dtype=np.float32)

