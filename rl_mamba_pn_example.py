import gym
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Model, GPT2Tokenizer, MambaForCausalLM

# Definir el modelo de política utilizando GPT-2
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.state_encoder = nn.Linear(4, 768)  # CartPole tiene 4 estados
        # self.llm = GPT2Model.from_pretrained('gpt2')
        self.llm = MambaForCausalLM.from_pretrained('state-spaces/mamba-370m-hf')
        self.fc = nn.Linear(self.llm.config.hidden_size, 2)  # CartPole tiene 2 acciones posibles

    def forward(self, x):
        x = self.state_encoder(x)
        x = x.unsqueeze(0)  # Añadir dimensión de batch
        outputs = self.llm(inputs_embeds=x)
        logits = self.fc(outputs.last_hidden_state[:, -1, :])
        return logits

# Función para seleccionar una acción basada en la política
def select_action(state, policy_net):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    logits = policy_net(state)
    action_prob = torch.softmax(logits, dim=-1)
    action = torch.multinomial(action_prob, 1).item()
    return action, action_prob

# Entrenamiento del agente
def train(env, policy_net, optimizer, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()[0]
        episode_reward = 0
        done = False

        while not done:
            action, action_prob = select_action(state, policy_net)
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward

            # Actualización de la política
            optimizer.zero_grad()
            loss = -torch.log(action_prob[0, action]) * reward
            loss.backward()
            optimizer.step()

            state = next_state

        print(f'Episodio {episode + 1}: Recompensa total = {episode_reward}')

# Configuración del entorno y el agente
env = gym.make('CartPole-v1')
policy_net = PolicyNetwork()
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

# Entrenar el agente
train(env, policy_net, optimizer)
