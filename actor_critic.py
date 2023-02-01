import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99, tau=1.0):
        super(ActorCritic, self).__init__()

        self.gamma = gamma
        self.tau = tau
        
        self.input_layer = nn.Linear(input_dims, 256)
        self.hidden_layer_1 = nn.Linear(256, 128)

        self.gru = nn.GRUCell(128, 256) # Explicacion 1
        self.actor_output = nn.Linear(256, n_actions)
        self.critic_output = nn.Linear(256, 1)

    def forward(self, state, hx):
        conv = F.elu(self.input_layer(state))
        conv = F.elu(self.hidden_layer_1(conv))

        conv_state = conv.view(128) # torch.Size([1, 128]) -------> torch.Size([128])
        hx = hx.view(256)

        hx = self.gru(conv_state, (hx))

        pi = self.actor_output(hx).unsqueeze(0)
        v = self.critic_output(hx)

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.numpy()[0], v, log_prob, hx

    def calculate_discounted_reward(self, done, rewards, values):
        values = T.cat(values).squeeze()

        if len(values.size()) == 1: 
            R = values[-1]*(1-int(done))
        elif len(values.size()) == 0:
            R = values*(1-int(done))

        discounted_reward = []
        for reward in rewards[::-1]:
            R = reward + self.gamma * R
            discounted_reward.append(R)
        discounted_reward.reverse()
        discounted_reward = T.tensor(discounted_reward, dtype=T.float).reshape(values.size())
        
        return discounted_reward

    def calculate_generalized_advantage_estimate(self, rewards, values):
        delta_t = rewards + self.gamma * values[1:] - values[:-1]
        n_steps = len(delta_t)
        gae = np.zeros(n_steps)
        for t in range(n_steps):
            for k in range(0, n_steps-t):
                temp = (self.gamma*self.tau)**k * delta_t[t+k]
                gae[t] += temp
        gae = T.tensor(gae, dtype=T.float)
        return gae


    def calc_cost(self, new_state, hx, done, rewards, values, log_probs, intrinsic_reward):

        rewards += intrinsic_reward.detach().numpy()

        returns = self.calculate_discounted_reward(done, rewards, values)

        next_value = T.zeros(1, 1)[0] if done else self.forward(T.tensor([new_state], dtype=T.float), hx)[1]

        values.append(next_value.detach())
        values = T.cat(values).squeeze()

        log_probs = T.cat(log_probs)
        rewards = T.tensor(rewards)

        gae = self.calculate_generalized_advantage_estimate(rewards, values)

        actor_loss = -(log_probs * gae).sum()
        critic_loss = F.mse_loss(values[:-1].squeeze(), returns)
        entropy_loss = (-log_probs * T.exp(log_probs)).sum()
        
        total_loss = actor_loss + critic_loss - 0.01 * entropy_loss
        
        return total_loss

'''
Explicacion 1:
Gated Recurrent Unit (GRU) es un variante de la Recurrent Neural Network (RNN) que usa 
mecanismos de puertas para manejar el flujo de la informacion entre las celdas de la red.

GRU permite capturar dependencias desde largas sequencias de data sin descartar las que se 
encuentra al principio. Esto se logra atraves de las unidades de activacion.

Estas unidades son las responsables de mantener y descartar la informacion

Las GRUs tienen un estado oculto pasado entre pasos de tiempo. Dicho estado oculto es capas
de mantener las dependencias de corto y largo plazo.

El GRU tiene dos puertas, la "puerta de actualizacion" y la "puerta de reinicio", las cuales 
se encargan de filtrar la informacion que es relevante.

Estas puertas son escencialmente vectores que contienen valores entre 0 y 1 los cuales son 
multiplicados con la data de entrada  y/o el estado oculto. Un valor 0 en la puerta indica
que la data correspondiente en la entrada o en el estado oculto es prescindible y por lo tanto,
retornara cero. Por el otro lado, un valor 1 en el vector puerta significa que corresponde con data 
que es importante y por lo tanto sera usada.
'''