ENV_NAME = 'LunarLander-v3'  # @param ['ALE/Breakout-v5','CartPole-v1','LunarLander-v3']
TRAIN_STEPS = 2_000_000 # stop after these *environment* steps
EVAL_EVERY  = 100_000 # evaluate every N steps

# DEPENDENCIES
import ale_py
import shimmy
import os, random, collections, math, time, itertools, imageio
import numpy as np, gymnasium as gym, torch, torch.nn as nn, torch.optim as optim
from IPython import display
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PRE‑PROCESSORS
class PixelProcessor:
    """⇢ 210×160 RGB ➜ 84×84 grayscale uint8"""
    def __call__(self, frame):
        import cv2
        f = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        f = cv2.resize(f, (84,84), interpolation=cv2.INTER_AREA)
        return f[None] # shape (1,84,84)

STACK     = 4 
obs_shape = (STACK, 84, 84)
n_actions = gym.make(ENV_NAME).action_space.n

# NETWORK HEADS
class ConvDuelingDQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(STACK,32,8,4), nn.ReLU(),
            nn.Conv2d(32,64,4,2),    nn.ReLU(),
            nn.Conv2d(64,64,3,1),    nn.ReLU(),
            nn.Conv2d(64,512,7,1),   nn.ReLU())
        self.adv = nn.Linear(512, n_actions)
        self.val = nn.Linear(512, 1)
    def forward(self,x):
        feats = self.feat(x/255.0) # feats: (B, 512, 1, 1)
        flat  = feats.view(feats.size(0), -1) #flat: (B, 512)
        a = self.adv(flat) # a: (B, n_actions)
        v = self.val(flat) # v: (B, 1)
        # dueling: Q(s, a) = V(s) + A(s,a) - mean_a A(s,a)
        return v + a - a.mean(dim=1, keepdim=True)

Net = ConvDuelingDQN(n_actions) 
policy_net, target_net = Net.to(device), Net.to(device); target_net.load_state_dict(policy_net.state_dict())

# REPLAY BUFFER
Transition = collections.namedtuple('T',('s','a','r','s1','d'))
class Replay:
    def __init__(self, cap=int(1e6)): self.buf, self.cap = collections.deque(maxlen=cap), cap
    def push(self,*args): self.buf.append(Transition(*args))
    def sample(self,bs):  return random.sample(self.buf,bs)
    def __len__(self):    return len(self.buf)
replay = Replay()

# UTILITIES
def stack_frames(stk, frame):
    stk.append(frame); return np.concatenate(list(stk),0)
def ε_by_step(step):
    if step < 50_000:  return 1.0
    if step < 1_050_000: return 1.0 - 0.9*(step-50_000)/1_000_000
    return 0.1 - 0.09*(step-1_050_000)/(TRAIN_STEPS-1_050_000)

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
huber = nn.SmoothL1Loss()

def select_action(state, ε):
    if random.random() < ε: return random.randrange(n_actions)
    with torch.no_grad():
        q = policy_net(torch.as_tensor(state, device=device, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(1))

steps_list, eps_list, loss_list, q_list, reward_list = [], [], [], [], []

def optimize(bs=32, γ=0.99):
    if len(replay) < 50_000 or len(replay) < bs:
        return None
    B = Transition(*zip(*replay.sample(bs)))
    s  = torch.as_tensor(np.stack(B.s),  device=device).float()
    a  = torch.as_tensor(B.a,            device=device).long().unsqueeze(1)
    r  = torch.as_tensor(B.r,            device=device).float()
    s1 = torch.as_tensor(np.stack(B.s1), device=device).float()
    d  = torch.as_tensor(B.d,            device=device).float()
    q  = policy_net(s).gather(1,a).squeeze()
    with torch.no_grad():
        a1 = policy_net(s1).argmax(1,keepdim=True)
        q1 = target_net(s1).gather(1,a1).squeeze()
        y  = r + γ*(1-d)*q1
    loss = huber(q,y)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(),10)
    optimizer.step()
    return loss.item()

# TRAIN 
env = gym.make(ENV_NAME, render_mode=None)
state = env.reset()[0]

stk = collections.deque([proc(state)]*STACK, maxlen=STACK)
state = np.concatenate(list(stk),0)


episode_r, best_eval = 0, -float('inf')
for step in range(1, TRAIN_STEPS+1):
    # 1) ε and step count
    ε = ε_by_step(step)
    steps_list.append(step)
    eps_list.append(ε)

    # 2) choose action, step env
    act = select_action(state, ε)
    nxt, r, term, trunc, _ = env.step(act)
    done = term or trunc

    # 3) record raw reward (or sign for Atari)
    reward_list.append(r)

    # 4) preprocess, push to replay
    r_clip    = np.sign(r)
    nxt_proc  = proc(nxt)
    nxt_state = stack_frames(stk, nxt_proc)
    replay.push(state, act, r_clip, nxt_state, done)

    # 5) optimize & record loss
    loss = optimize()
    loss_list.append(loss if loss is not None else float('nan'))

    # 6) record mean Q-value of current state
    with torch.no_grad():
        qvals = policy_net(torch.as_tensor(state, device=device, dtype=torch.float32).unsqueeze(0))
        q_list.append(qvals.max().item())

    # 7) update state, episode reward
    state, episode_r = nxt_state, episode_r + r

    # 8) sync target net
    if step % 10_000 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # 9) end of episode logging
    if done:
        print(f"{step:7,d} | ep reward: {episode_r:6.1f} | ε={ε:.3f}")
        state, episode_r = env.reset()[0], 0
        stk = collections.deque([proc(state)]*STACK, maxlen=STACK)
        state = np.concatenate(list(stk),0)
    

    # quick evaluation
    if step % EVAL_EVERY == 0:
        e_env = gym.make(ENV_NAME, render_mode=None)
        e_rewards=[]
        for _ in range(5):
            s = e_env.reset()[0]
        
            sstk=collections.deque([proc(s)]*STACK,maxlen=STACK)
            s=np.concatenate(list(sstk),0)
        
            er=0
            while True:
                a = select_action(s, 0.05)
                sn,r,done,_,_=e_env.step(a)
                s = stack_frames(sstk, proc(sn))
                er += r
                if done: break
            e_rewards.append(er)
        avg= np.mean(e_rewards)
        if avg>best_eval: best_eval=avg; torch.save(policy_net.state_dict(), f'dqn_{ENV_NAME}.pt')
        print(f"── eval @ {step:,}: {avg:.1f} (best {best_eval:.1f})")
