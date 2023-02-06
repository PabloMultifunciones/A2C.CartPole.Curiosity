import torch as T
import torch.nn as nn
import torch.nn.functional as F


class ICM(nn.Module):
    def __init__(self, input_dims, n_actions, alpha=0.1, beta=0.2):
        super(ICM, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.input_layer = nn.Linear(input_dims, 256)
        self.hidden_layer_1 = nn.Linear(256, 256)

        self.phi = nn.Linear(256, 128)
        self.phi_hat_new = nn.Linear(256, 128)

        self.inverse = nn.Linear(128*2, 256)
        self.pi_logits = nn.Linear(256, n_actions)

        self.dense1 = nn.Linear(128+1, 256)

        device = T.device('cpu')
        self.to(device)

    def forward(self, state, new_state, action):
        conv = F.elu(self.input_layer(state))
        conv = F.elu(self.hidden_layer_1(conv))
        phi = self.phi(conv)

        conv_new = F.elu(self.input_layer(new_state))
        conv_new = F.elu(self.hidden_layer_1(conv_new))
        phi_new = self.phi(conv_new)

        phi = phi.view(phi.size()[0], -1).to(T.float)
        phi_new = phi_new.view(phi_new.size()[0], -1).to(T.float)

        inverse = self.inverse(T.cat([phi, phi_new], dim=1))
        pi_logits = self.pi_logits(inverse)

        action = action.reshape((action.size()[0], 1))

        forward_input = T.cat([phi, action], dim=1)

        dense = self.dense1(forward_input)
        phi_hat_new = self.phi_hat_new(dense)

        return phi_new, pi_logits, phi_hat_new

    def calc_loss(self, states, new_states, actions):
        actions = T.tensor(actions, dtype=T.float)

        states = T.tensor([item.cpu().detach().numpy() for item in states])
        new_states = T.tensor([item.cpu().detach().numpy() for item in new_states])

        phi_new, pi_logits, phi_hat_new = self.forward(states, new_states, actions)
        
        inverse_loss = nn.CrossEntropyLoss()

        L_I = (1 - self.beta) * inverse_loss(pi_logits, actions.to(T.long))
        
        forward_loss = nn.MSELoss()
        L_F = self.beta * forward_loss(phi_hat_new, phi_new)

        intrinsic_reward = self.alpha*0.5*((phi_hat_new-phi_new).pow(2)).mean(dim=1)
        return intrinsic_reward, L_I, L_F
