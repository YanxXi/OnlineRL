import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedCategorial(torch.distributions.Categorical):
    @property
    def deterministic(self):
        return self.probs.argmax(dim=-1, keepdim=True)

    
class FixedNormal(torch.distributions.Normal):
    @property
    def deterministic(self):
        return self.mean


class Categorial(nn.Module):
    def __init__(self, input_size, output_size):
        super(Categorial, self).__init__()
        self.logits_net = nn.Linear(input_size, output_size)
        
    def forward(self, x: F.Tensor):
        logits = self.logits_net(x)
        return FixedCategorial(logits=logits)


class Normal(nn.Module):
    def __init__(self, input_size, output_size):
        super(Normal, self).__init__()
        self.mu_net = nn.Linear(input_size, output_size)
        self.log_std_net = nn.Linear(input_size, output_size)

    def forward(self, x: F.Tensor):
        mu = self.mu_net(x)
        log_std = self.log_std_net(x)
        std = torch.exp(log_std)
        return FixedNormal(mu, std)


if __name__ == "__main__":
    b1 = torch.FloatTensor([[[0.3, 0.4], [0.1, 0.2]]])
    b2 = torch.FloatTensor([[[0.1, 0.2], [0.2, 0.5]]])
    a = FixedNormal(b1, b2)
    action = a.sample()
    print(action)
    print(a.log_prob(action))
    print(a.entropy())
    #print(a.entropy().unsqueeze(-1))
    #print(torch.cat([b1, b2], dim=-1))
    #print(action.mean())
    a1 = [np.arange(3, 7, 2)]
    a2 = [np.arange(3,8,2)]
    print(np.concatenate([a1, a2], axis=-1)[..., 0:9])
    c = np.random.rand(3,2)
    c1 = np.zeros_like(c)
    c2 = np.array([1,2,3,4])

    c1[..., 1] = c2[0:3]
    print(c1)