import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random
from collections import deque
from Game import Game
import cv2

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

class Deep_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name = 'Deep_QNet', path='./'):
        torch.save(self.state_dict(), path + file_name+'.pth')


class QAgent:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model

        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=100_000)
        self.epsilon = 0.01
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.train_step(state, action, reward, next_state, done)
    
    def train_long_memory(self, batch_size):
        if len(self.memory) > batch_size:
            mini_sample = random.sample(self.memory, batch_size)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_action(self, state):
        final_move = [0, 0, 0, 0]
        if np.random.rand() <= self.epsilon:
            move = np.random.randint(len(final_move))
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float)#.cuda()
            prediction = self.model(state0)#.cuda()
            move = torch.argmax(prediction).item()
            final_move[move] = 1

            print(LINE_UP, end=LINE_CLEAR)
            print(final_move)
        return final_move

    @torch.enable_grad()
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype = torch.float)
        next_state = torch.tensor(np.array(next_state), dtype = torch.float)
        action = torch.tensor(np.array(action), dtype = torch.long)
        reward = torch.tensor(np.array(reward), dtype = torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)     
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()


# Action 
# [1, 0, 0, 0] -> forward
# [0, 1, 0, 0] -> backward
# [0, 0, 1, 0] -> left
# [0, 0, 0, 1] -> right

def process_action(action):
    if action == [0, 0, 0, 0]:
        return (0, 0)
    if action == [1, 0, 0, 0]:
        return (1, 0)
    if action == [0, 1, 0, 0]:
        return (-1, 0)
    if action == [0, 0, 1, 0]:
        return (0, 1)
    if action == [0, 0, 0, 1]:
        return (0, -1)

if __name__ == "__main__":
    env = Game()
    env.init_render()
    env.clock.tick(60)
    env.init()

    state_size = 14
    action_size = 4

    batch_size = 1024
    episodes = 5000

    model = Deep_QNet(14, 256, 4)
    agent = QAgent(model, 0.001, 0.95)

    model.load_state_dict(torch.load('Deep_QNet4700.pth'))
    model.train()

    training_info = []
    record = 0

    # model.save("Deep_QNet1")

    # Training loop
    for episode in range(4700, episodes):
        state = env.reset()
        done = False
        time_step = 0
        total_reward = 0
        checkpoint = 0

        while True:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(process_action(action))
            total_reward += reward
            env.render()

            agent.remember(state, action, reward, next_state, done)
            agent.train_short_memory(state, action, reward, next_state, done)
            state = next_state
            time_step += 1

            # if time_step % 1_000 == 0:
            #     if checkpoint == env.checkpoint_id[1]:
            #         done = True
            #     checkpoint = env.checkpoint_id[1]

            if done:
                training_info.append([episode+1, time_step, agent.epsilon, env.checkpoint_id[1], total_reward])
                print("Episode: {}/{}, Score: {}, Epsilon: {:.2}, Checkpoint: {}, Reward: {}".format(episode + 1, episodes, time_step, agent.epsilon, env.checkpoint_id, total_reward))
                print()
                if env.checkpoint_id[1] > record:
                    record = env.checkpoint_id[1]
                    model.save('Deep_QNet'+str(episode+1)+'_'+str(record))
                    np.save("training_info"+str(episode+1)+'_'+str(record), training_info)
                agent.train_long_memory(batch_size)
                break

        if (episode+1) % 100 == 0:
            model.save('Deep_QNet'+str(episode+1))
            np.save("training_info"+str(episode+1), training_info)


    env.close()
    model.save()
    np.save("training_info", training_info)
