import numpy as np
import torch as T
from memory import Memory


def worker(agent, agent_optimizer, icm, icm_optimizer, env):
    T_MAX = 20

    memory = Memory()

    episode, max_steps, t_steps, scores = 0, 5e5, 0, []

    while episode < max_steps:
        state, _ = env.reset()

        score, done, ep_steps = 0, False, 0
        hx = T.zeros(1, 256)
        while not done:
            state = T.tensor(state, dtype=T.float, device="cpu").unsqueeze(0) # Imagen con forma(210, 160) pasa a ser un tensor torch.Size([1, 210, 160])

            action, value, log_prob, hx = agent(state, hx)
            next_state, reward, done, _, _ = env.step(action)
            auxiliar = next_state

            next_state = T.tensor(next_state, dtype=T.float, device="cpu").unsqueeze(0) # Imagen con forma(210, 160) pasa a ser un tensor torch.Size([1, 210, 160])

            memory.remember(state, action, next_state, reward, value, log_prob)
           
            score += reward
            state = auxiliar

            ep_steps += 1
            t_steps += 1

            if ep_steps % T_MAX == 0 or done:
                states, actions, new_states, rewards, values, log_probs = memory.sample_memory()
                
                intrinsic_reward, L_I, L_F = icm.calc_loss(states, new_states, actions)

                loss = agent.calc_cost(state, hx, done, rewards, values, log_probs, intrinsic_reward)
                
                agent_optimizer.zero_grad()

                hx = hx.detach()

                icm_optimizer.zero_grad()

                (L_I + L_F).backward()
                loss.backward()
                
                T.nn.utils.clip_grad_norm_(agent.parameters(), 40)

                agent_optimizer.step()
                icm_optimizer.step()

                memory.clear_memory()
        episode += 1
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_score_5000 = np.mean(scores[max(0, episode-5000): episode+1])
        print('ICM episode {} steps {:.2f}M score {:.2f} avg score (100) (5000) {:.2f} {:.2f}'.format(episode, t_steps/1e6, score, avg_score, avg_score_5000))


'''
Torch.Unqueeze(dim)
Devuelve un nuevo tensor de dimensión uno insertado en la posición especificada.
El tensor devuelto comparte los mismos datos subyacentes con este tensor.

Ejemplo:
x = T.tensor([[1,2], [2,4], [3,6], [4,6]])

print(x.unsqueeze(0)) 

tensor([[[1, 2],
         [2, 4],
         [3, 6],
         [4, 6]]])

print(x.unsqueeze(1))

tensor([[[1, 2]],

        [[2, 4]],

        [[3, 6]],

        [[4, 6]]])

print(x.unsqueeze(2))

tensor([[[1],
         [2]],

        [[2],
         [4]],

        [[3],
         [6]],

        [[4],
         [6]]])

'''