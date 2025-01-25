import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class DQNNetwork(nn.Module):
    def __init__(self, input_lay, hidden_lay, output_lay):
        super().__init__()
        self.linear1 = nn.Linear(input_lay, hidden_lay)
        self.linear2 = nn.Linear(hidden_lay, output_lay)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, fn='model.pth'):
        mpath = './learning/models'
        os.makedirs(mpath, exist_ok=True)
        
        path = os.path.join(mpath, fn)
        torch.save(self.state_dict(), path)

class DQNTrain:
    def __init__(self, model, alpha, gamma):
        self.model = model
        self.alpha = alpha
        self.gamma = gamma
        self.optmz = optim.Adam(model.parameters(), lr=self.alpha)
        self.lossFunc = nn.MSELoss()

    def train(self, state, action, reward, next_state, done):
        state = np.array(state)
        action = np.array(action)
        next_state = np.array(next_state, dtype=float)

        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done, )

        pred = self.model(state)

        target = pred.clone()
        for i in range(len(done)):
            Qn = reward[i]
            if not done[i]:
                Qn = reward[i] + self.gamma * torch.max(self.model(next_state[i])) #bellman equation

            target[i][torch.argmax(action).item()] = Qn

        self.optmz.zero_grad()
        loss = self.lossFunc(target, pred)
        loss.backward()

        self.optmz.step()