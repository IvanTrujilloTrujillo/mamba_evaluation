import gym
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import mse_loss

# Configurar el entorno de RL
env = gym.make("CartPole-v1")

# Configurar el modelo LLaMA (suponiendo que el modelo LLaMA está disponible en Hugging Face)
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Optimizer para fine-tuning
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# Función para convertir observaciones a formato de entrada del modelo
def preprocess_observation(observation):
    obs_str = ','.join([str(x) for x in observation])
    input_ids = tokenizer.encode(obs_str, return_tensors='pt')
    return input_ids

# Función para obtener la acción del modelo
def get_action(input_ids):
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits
    action = torch.argmax(logits[:, -1, :]).item()
    return action % 2  # Para asegurarse de que sea 0 o 1

# Función de ajuste del modelo
def fine_tune(model, input_ids, action, reward):
    model.train()
    
    # Forward pass
    outputs = model(input_ids=input_ids, labels=input_ids)
    logits = outputs.logits
    
    # Calculate loss
    predicted_action = torch.argmax(logits[:, -1, :])
    loss = mse_loss(predicted_action.float(), torch.tensor([action], dtype=torch.float)) * reward
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Entrenamiento RL
total_episodes = 50
for episode in range(total_episodes):
    observation = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Preprocesar la observación y obtener la acción
        input_ids = preprocess_observation(observation)
        action = get_action(input_ids)
        
        # Tomar la acción en el entorno
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        
        # Fine-tuning del modelo basado en la recompensa
        loss = fine_tune(model, input_ids, action, reward)
        
        if done:
            print(f"Episode {episode + 1}: Total Reward: {total_reward}, Loss: {loss}")
            break

env.close()