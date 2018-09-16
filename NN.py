#Details of neural networks
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameters import *

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES[0], 128)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization

        if DUEL:
            self.fc2_adv = nn.Linear(128, 128)
            self.fc2_adv.weight.data.normal_(0, 0.1)   # initialization
            self.fc2_val = nn.Linear(128, 128)
            self.fc2_val.weight.data.normal_(0, 0.1)   # initialization

            self.out_adv = nn.Linear(128, N_ACTIONS)
            self.out_adv.weight.data.normal_(0, 0.1)   # initialization
            self.out_val = nn.Linear(128, 1)
            self.out_val.weight.data.normal_(0, 0.1)   # initialization

        else:
            self.fc2 = nn.Linear(128, 128)
            self.fc2.weight.data.normal_(0, 0.1)   # initialization

            self.out = nn.Linear(128, N_ACTIONS)
            self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = F.relu(self.fc1(x))

        if DUEL:
            adv = F.relu(self.fc2_adv(x))
            val = F.relu(self.fc2_val(x))

            adv = self.out_adv(adv)
            val = self.out_val(val).expand(x.size(0), N_ACTIONS)
            action_value = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), N_ACTIONS)

        else:
            x = F.relu(self.fc2(x))
            action_value = self.out(x)
            
        return action_value

    def save(self, PATH):
        torch.save(self.state_dict(),PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(STATE_LEN, 32, kernel_size=8, stride=4)
        nn.init.orthogonal_(self.conv1.weight, gain=np.sqrt(2))
        nn.init.constant_(self.conv1.bias, 0.0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        nn.init.orthogonal_(self.conv2.weight, gain=np.sqrt(2))
        nn.init.constant_(self.conv2.bias, 0.0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        nn.init.orthogonal_(self.conv3.weight, gain=np.sqrt(2))
        nn.init.constant_(self.conv3.bias, 0.0)
        self.fc = nn.Linear(3136, 512)
        nn.init.orthogonal_(self.fc.weight, gain=np.sqrt(2))
        nn.init.constant_(self.fc.bias, 0.0)

        if DUEL:
            self.fc3_adv = nn.Linear(3136, 512)
            self.fc3_adv.weight.data.normal_(0, 0.1)   # initialization
            self.fc3_val = nn.Linear(3136, 512)
            self.fc3_val.weight.data.normal_(0, 0.1)   # initialization

            self.fc4_adv = nn.Linear(512, N_ACTIONS)
            self.fc4_adv.weight.data.normal_(0, 0.1)   # initialization
            self.fc4_val = nn.Linear(512, 1)
            self.fc4_val.weight.data.normal_(0, 0.1)   # initialization

        else:
            self.fc3 = nn.Linear(3136, 512)
            self.fc3.weight.data.normal_(0,0.1)
            self.fc4 = nn.Linear(512, N_ACTIONS)
            self.fc4.weight.data.normal_(0,0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x / 255.0))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        if DUEL:
            x = x.view(x.size(0), -1)
            adv = F.relu(self.fc3_adv(x))
            val = F.relu(self.fc3_val(x))

            adv = self.fc4_adv(adv)
            val = self.fc4_val(val).expand(x.size(0), N_ACTIONS)
            action_value = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), N_ACTIONS)

        else:
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc3(x))
            action_value = self.fc4(x)

        return action_value

    def save(self, PATH):
        torch.save(self.state_dict(),PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))