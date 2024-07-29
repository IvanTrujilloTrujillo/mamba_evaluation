import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2Model

class ActorCriticModel(nn.Module):
    def __init__(self, gpt_model_name):
        super(ActorCriticModel, self).__init__()
        self.gpt = GPT2Model.from_pretrained(gpt_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
        self.actor = nn.Linear(self.gpt.config.hidden_size, action_space_size)
        self.critic = nn.Linear(self.gpt.config.hidden_size, 1)
    
    def forward(self, text_input):
        inputs = self.tokenizer(text_input, return_tensors='pt')
        outputs = self.gpt(**inputs)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, -1, :]  # Utiliza el último token [CLS] para resumen
        action_logits = self.actor(pooled_output)
        value = self.critic(pooled_output)
        return action_logits, value

# Definición del espacio de acción y otros hiperparámetros
action_space_size = 10  # Por ejemplo
learning_rate = 0.01

# Inicialización del modelo y optimizador
model = ActorCriticModel('gpt2')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Ejemplo de bucle de entrenamiento
for episode in range(100):
    state = "Descripción del estado inicial"
    done = False
    while not done:
        action_logits, value = model(state)
        action_probabilities = torch.softmax(action_logits, dim=-1)
        action = torch.multinomial(action_probabilities, 1).item()
        
        # Interactúa con el entorno
        next_state, reward, done = environment.step(action)
        
        # Calcula la pérdida y realiza el backpropagation
        _, next_value = model(next_state)
        td_target = reward + 0.99 * next_value * (1 - done)
        td_error = td_target - value
        actor_loss = -torch.log(action_probabilities[action]) * td_error
        critic_loss = td_error.pow(2)
        loss = actor_loss + critic_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = next_state