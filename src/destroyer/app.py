import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading, time, random, json, os, math, queue
import numpy as np
import torch
from destroyer import Game2048, DQNAgent, ACTIONS, encode

BG='#111820'; PANEL='#1b2632'; FG='#e9f0f4'; MUTED='#91a2b2'; ACC='#50d890'; BLUE='#52a8ff'; TILE_COLORS={0:'#253442',2:'#e9e4d8',4:'#ede0b9',8:'#f2b477',16:'#f18b55',32:'#f16c4c',64:'#e9543d',128:'#e6cf6c',256:'#dfc352',512:'#d8b23d',1024:'#b6a23c',2048:'#9c9133'}
class App:
 def __init__(self,root):
  self.root=root; root.title('2048 DESTROYER | Neural Agent Lab'); root.geometry('1420x900'); root.minsize(1180,760); root.configure(bg=BG)
  self.game=Game2048(); self.agent=DQNAgent(); self.training=False; self.stop_flag=threading.Event(); self.uiq=queue.Queue(); self.episode=0; self.episodes=[]; self.history=[]; self.last_action='—'; self.last_reward=0; self.last_loss=0; self.last_move='—'; self.eval_games=0; self.eval_2048=0; self.eval_best=0; self.train_thread=None
  self._style(); self._layout(); self.render(); self.refresh_stats(); self.root.bind('<Up>',lambda e:self.manual('UP')); self.root.bind('<Down>',lambda e:self.manual('DOWN')); self.root.bind('<Left>',lambda e:self.manual('LEFT')); self.root.bind('<Right>',lambda e:self.manual('RIGHT')); self.root.bind('<space>',lambda e:self.manual_auto()); self.root.after(150,self.poll)
 def _style(self):
  s=ttk.Style(); s.theme_use('clam'); s.configure('TButton',font=('Segoe UI',10),padding=7); s.configure('TLabel',background=BG,foreground=FG,font=('Segoe UI',10)); s.configure('TFrame',background=BG); s.configure('TLabelframe',background=BG,foreground=FG); s.configure('TLabelframe.Label',background=BG,foreground=FG,font=('Segoe UI',10,'bold'))
 def _layout(self):
  header=tk.Frame(self.root,bg=BG); header.pack(fill='x',padx=16,pady=(12,6)); tk.Label(header,text='2048 DESTROYER',bg=BG,fg=FG,font=('Segoe UI',23,'bold')).pack(side='left'); tk.Label(header,text='NEURAL NETWORK LAB  /  DQN  /  4 LAYERS',bg=BG,fg=ACC,font=('Consolas',10)).pack(side='left',padx=18,pady=9)
  toolbar=tk.Frame(self.root,bg=BG); toolbar.pack(fill='x',padx=16,pady=6)
  for label,cmd in [('▶ TRAIN',self.start_training),('■ STOP',self.stop_training),('↻ NEW GAME',self.new_game),('STEP AI',self.ai_step),('▶ AUTO PLAY',self.auto_play),('SAVE MODEL',self.save_model),('LOAD MODEL',self.load_model),('EVALUATE ×100',self.evaluate)]: ttk.Button(toolbar,text=label,command=cmd).pack(side='left',pad=(0,6))
  body=tk.Frame(self.root,bg=BG); body.pack(fill='both',expand=True,padx=16,pady=8); body.columnconfigure(0,weight=1); body.columnconfigure(1,weight=1); body.columnconfigure(2,weight=1); body.rowconfigure(0,weight=1)
  left=tk.Frame(body,bg=BG); left.grid(row=0,column=0,sticky='nsew',pad=(0,8,10,10)); mid=tk.Frame(body,bg=BG); mid.grid(row=0,column=1,sticky='nsew',pad=8); right=tk.Frame(body,bg=BG); right.grid(row=0,column=2,sticky='nsew',pad=(8,0,0,8))
  self.board_canvas=tk.Canvas(left,width=472,height=472,bg='#344452',highlightthickness=0); self.board_canvas.pack(pady=8)
  score=tk.Frame(left,bg=BG); score.pack(fill='x',padx=10); self.score_label=self.metric(score,'SCORE','0'); self.best_label=self.metric(score,'BEST TILE','2'); self.moves_label=self.metric(score,'MOVES','0')
  self.board_info=tk.Label(left,text='Arrow keys: manual moves   |   SPACE: autoplay',bg=BG,fg=MUTED,font=('Consolas',10)); self.board_info.pack(pady=5)
  self.q_frame=tk.LabelFrame(mid,text='Q-VALUE ANALYSIS',bg=BG,fg=FG,padx=8,pady=8); self.q_frame.pack(fill='x'); self.qbars={}
  for action in ACTIONS:
   row=tk.Frame(self.q_frame,bg=BG); row.pack(fill='x',pady=3); tk.Label(row,text=action,width=7,bg=BG,fg=FG,font=('Consolas',10)).pack(side='left'); bar=tk.Canvas(row,width=200,height=18,bg=PANEL,highlightthickness=0); bar.pack(side='left',pad=4); val=tk.Label(row,text='0.000',width=9,bg=BG,fg=FG,font=('Consolas',10)); val.pack(side='right'); self.qbars[action]=(bar,val)
  self.net=tk.LabelFrame(mid,text='NETWORK / TRAINING',bg=BG,fg=FG,padx=8,pady=8); self.net.pack(fill='x',pady=8); self.stats={}
  for k in ['Episode','Train steps','Epsilon','Loss','Device','Replay','Avg score (50)','Avg max tile (50)','2048 rate','4096 rate']:
   row=tk.Frame(self.net,bg=BG); row.pack(fill='x'); tk.Label(row,text=k,bg=BG,fg=MUTED,width=19,anchor='w').pack(side='left'); v=tk.Label(row,text='—',bg=BG,fg=FG,font=('Consolas',10),anchor='e'); v.pack(side='right'); self.stats[k]=v
  self.debug=tk.LabelFrame(mid,text='LIVE DEBUG',bg=BG,fg=FG,padx=8,pady=6); self.debug.pack(fill='both',expand=True); self.debug_text=tk.Text(self.debug,height=10,bg='#0c1218',fg='#a7e4c6',insertbackground=FG,font=('Consolas',9),wrap='word'); self.debug_text.pack(fill='both',expand=True); self.log('Application ready. Model parameters: 27,844')
  right_top=tk.LabelFrame(right,text='TRAINING HISTORY',bg=BG,fg=FG,padx=5,pady=5); right_top.pack(fill='x'); self.chart=tk.Canvas(right_top,height=190,bg='#0c1218',highlightthickness=0); self.chart.pack(fill='x')
  self.history_tree=ttk.Treeview(right,columns=('ep','score','tile','moves','loss'),show='headings',height=13); 
  for c,t,w in [('ep','EP',50),('score','SCORE',80),('tile','MAX',60),('moves','MOVES',60),('loss','LOSS',70)]: self.history_tree.heading(c,text=t); self.history_tree.column(c,width=w,anchor='center')
  self.history_tree.pack(fill='both',expand=True,pady=8); bottom=tk.Label(right,text='EPISODE LOG  |  latest 100 records',bg=BG,fg=MUTED,font=('Consolas',9)); bottom.pack(anchor='w')
 def metric(self,frame,name,value):
  box=tk.Frame(frame,bg=PANEL,padx=10,pady=6); box.pack(side='left',fill='x',expand=True,padx=3); tk.Label(box,text=name,bg=PANEL,fg=MUTED,font=('Consolas',9)).pack(); v=tk.Label(box,text=value,bg=PANEL,fg=FG,font=('Segoe UI',18,'bold')); v.pack(); return v
 def log(self,msg):
  self.debug_text.insert('end',f'[{time.strftime("%H:%M:%S")}] {msg}\n'); self.debug_text.see('end')
 def render(self):
  c=self.board_canvas; c.delete('all'); board=self.game.board; cell=110; gap=8; margin=8
  for r in range(4):
   for col in range(4):
    x=margin+col*(cell+gap); y=margin+r*(cell+gap); v=board[r][col]; fill=TILE_COLORS.get(v,'#8c7a2c')
    if v>2048: fill='#82702c'
    c.create_rectangle(x,y,x+cell,y+cell,fill=fill,outline=fill)
    if v: c.create_text(x+cell/2,y+cell/2,text=str(v),fill='#27313a' if v<=64 else '#fff7e7',font=('Segoe UI',24 if v<1024 else 19,'bold'))
  self.score_label.config(text=str(self.game.score)); self.best_label.config(text=str(self.game.max_tile)); self.moves_label.config(text=str(self.game.moves)); q=self.agent.q_values(self.game.board)
  for i,a in enumerate(ACTIONS):
   bar,label=self.qbars[a]; bar.delete('all'); scale= max(1.0,float(np.max(np.abs(q))))
   zero=100; x=zero+(q[i]/scale)*90; bar.create_line(zero,0,zero,18,fill='#5d6d7a'); bar.create_rectangle(min(zero,x),2,max(zero,x),16,fill=ACC if q[i]>=0 else '#e77969',outline=''); label.config(text=f'{q[i]:+.3f}')
  self.refresh_stats(); self.draw_chart()
 def refresh_stats(self):
  recent=self.episodes[-50:]; avg_score=sum(e['score'] for e in recent)/max(1,len(recent)); avg_tile=sum(e['tile'] for e in recent)/max(1,len(recent)); rate=sum(e['tile']>=2048 for e in recent)/max(1,len(recent)); rate4=sum(e['tile']>=4096 for e in recent)/max(1,len(recent)); vals={'Episode':self.episode,'Train steps':self.agent.steps,'Epsilon':f'{self.agent.epsilon:.4f}','Loss':f'{self.agent.loss_value:.5f}','Device':str(self.agent.device),'Replay':len(self.agent.replay),'Avg score (50)':f'{avg_score:.1f}','Avg max tile (50)':f'{avg_tile:.1f}','2048 rate':f'{rate*100:.1f}%','4096 rate':f'{rate4*100:.1f}%'}
  for k,v in vals.items(): self.stats[k].config(text=str(v))
 def draw_chart(self):
  c=self.chart; c.delete('all'); data=self.episodes[-80:]; w=max(400,c.winfo_width()); h=190
  if len(data)<2: c.create_text(w/2,h/2,text='Train episodes to populate chart',fill=MUTED); return
  vals=[e['score'] for e in data]; mx=max(vals) or 1; pts=[]
  for i,v in enumerate(vals): pts.append((10+i*(w-20)/(len(vals)-1),h-15-(v/mx)*(h-30)))
  c.create_line(*[z for p in pts for z in p],fill=BLUE,width=2); c.create_text(42,12,text=f'Score  max={mx}',fill=FG,anchor='w')
 def new_game(self): self.game.reset(); self.last_action='RESET'; self.render(); self.log('New episode started (manual).')
 def manual(self,a):
  if self.training: return
  s=self.game.move(a); self.last_action=a; self.last_reward=s.score_delta; self.render(); self.log(f'Manual {a}: +{s.score_delta}, merges={s.merges}, max={self.game.max_tile}')
 def ai_step(self):
  legal=self.game.legal_actions(); idx=self.agent.act(self.game.board,legal,explore=False); action=ACTIONS[idx]
  step=self.game.move(action); self.last_action=action; self.last_reward=step.score_delta; self.render(); self.log(f'AI {action}: reward +{step.score_delta}, legal={legal}, max={self.game.max_tile}')
  if step.game_over: self.log('GAME OVER. New board spawned.'); self.game.reset(); self.render()
 def auto_play(self):
  if self.training: return
  for _ in range(1): self.ai_step(); self.root.after(100,self.auto_play)
 def manual_auto(self): self.auto_play()
 def start_training(self):
  if self.training: return
  self.training=True; self.stop_flag.clear(); self.train_thread=threading.Thread(target=self.train_loop,daemon=True); self.train_thread.start(); self.log('TRAINING STARTED')
 def stop_training(self): self.stop_flag.set(); self.training=False; self.log('Stop requested.')
 def train_loop(self):
  try:
   while not self.stop_flag.is_set():
    g=Game2048(); done=False; total=0; steps=0
    while not done and not self.stop_flag.is_set():
     state=encode(g.board); legal=g.legal_actions(); aidx=self.agent.act(g.board,legal,True); action=ACTIONS[aidx]; step=g.move(action); done=step.game_over; nextstate=encode(g.board)
     reward=float(step.score_delta)/10.0 + (0.02*g.empty_count) - (0.1 if not step.moved else 0)
     if step.moved and g.max_tile>0: reward+=0.05
     self.agent.replay.add(state,aidx,reward,nextstate,done); loss=self.agent.learn(); total+=step.score_delta; steps+=1
     self.agent.epsilon=max(.05,self.agent.epsilon*.999995)
     if steps%200==0: self.uiq.put(('progress',g.board,total,steps,action,reward,done))
    if self.stop_flag.is_set(): break
    self.episode+=1; rec={'episode':self.episode,'score':total,'tile':g.max_tile,'moves':steps,'loss':self.agent.loss_value}; self.episodes.append(rec); self.uiq.put(('episode',rec))
  except Exception as e: self.uiq.put(('error',repr(e)))
 def poll(self):
  try:
   while True:
    item=self.uiq.get_nowait()
    if item[0]=='progress': _,board,score,steps,act,reward,done=item; self.game.board=[r[:] for r in board]; self.game.score=score; self.game.moves=steps; self.last_action=act; self.last_reward=reward; self.render(); self.log(f'train step={steps} act={act} r={reward:+.3f} terminal={done}')
    elif item[0]=='episode': rec=item[1]; self.history.append(rec); self.history_tree.insert('',0,values=(rec['episode'],rec['score'],rec['tile'],rec['moves'],f'{rec["loss"]:.5f}')); self.log(f'EP {rec["episode"]} score={rec["score"]} max={rec["tile"]} steps={rec["moves"]} loss={rec["loss"]:.5f}')
    elif item[0]=='log': self.log(item[1])
    elif item[0]=='eval': self.log('EVALUATION '+str(item[1]))
    elif item[0]=='error': self.log('TRAIN ERROR: '+item[1]); self.training=False; self.stop_flag.set()
  except queue.Empty: pass
  self.refresh_stats(); self.root.after(150,self.poll)
 def save_model(self):
  p=filedialog.asksaveasfilename(defaultextension='.pt',filetypes=[('PyTorch checkpoint','*.pt')]);
  if p: self.agent.save(p,self.episode); self.log('Saved checkpoint '+p)
 def load_model(self):
  p=filedialog.askopenfilename(filetypes=[('PyTorch checkpoint','*.pt')]);
  if p:
   try: self.episode=self.agent.load(p); self.log('Loaded checkpoint '+p); self.render()
   except Exception as e: messagebox.showerror('Checkpoint error',str(e))
 def evaluate(self):
  if self.training: messagebox.showinfo('Busy','Stop training before evaluation.'); return
  self.log('Evaluating 100 games...'); threading.Thread(target=self.eval_loop,daemon=True).start()
 def eval_loop(self):
  scores=[]; tiles=[]
  for i in range(100):
   g=Game2048();
   while not g.game_over():
    legal=g.legal_actions(); idx=self.agent.act(g.board,legal,False); g.move(ACTIONS[idx])
   scores.append(g.score); tiles.append(g.max_tile)
  result={'games':100,'avg_score':sum(scores)/100,'median_score':float(np.median(scores)),'max_tile':max(tiles),'rate2048':sum(t>=2048 for t in tiles)/100,'rate4096':sum(t>=4096 for t in tiles)/100}
  self.uiq.put(('eval',result))
  self.uiq.put(('log',f'EVAL 100: avg={result["avg_score"]:.1f}, median={result["median_score"]:.1f}, max={result["max_tile"]}, 2048={result["rate2048"]*100:.1f}%'))
if __name__=='__main__':
 root=tk.Tk(); app=App(root); root.mainloop()

if __name__ == '__main__':
 root=tk.Tk(); app=App(root); root.mainloop()


