import gym
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Cargar el entorno CartPole de OpenAI Gym
env = gym.make('CartPole-v1')

# Cargar el modelo de lenguaje preentrenado de Hugging Face (BERT, en este caso)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Configurar el modelo en modo evaluación
model.eval()

# Función para calcular la recompensa ajustada utilizando el LLM
def adjust_reward_with_llm(state, action, original_reward):
    # Convertir el estado y la acción en una representación de texto
    state_text = "State: " + " ".join(map(str, state))
    action_text = "Action: " + str(action)
    
    # Concatenar el estado y la acción para formar la entrada del modelo
    input_text = state_text + " " + action_text
    
    # Tokenizar la entrada y pasarla por el modelo
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Obtener el valor del logits para la clase "positiva" (ajuste de recompensa)
    logits = outputs.logits[:, 1]
    reward_adjustment = torch.sigmoid(logits).item()
    
    # Ajustar la recompensa original con el valor obtenido del modelo
    adjusted_reward = original_reward * reward_adjustment
    
    return adjusted_reward

# Parámetros del entrenamiento
num_episodes = 100
max_steps_per_episode = 200

# Entrenamiento del agente con recompensas ajustadas
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for step in range(max_steps_per_episode):
        # Renderizar el entorno (opcional, útil para visualizar)
        # env.render()

        # Seleccionar una acción aleatoria (0 o 1)
        action = env.action_space.sample()

        # Ejecutar la acción en el entorno
        next_state, original_reward, done, _, _ = env.step(action)

        # Ajustar la recompensa usando el LLM
        adjusted_reward = adjust_reward_with_llm(state, action, original_reward)

        # Acumular la recompensa total
        total_reward += adjusted_reward

        # Actualizar el estado actual
        state = next_state

        # Si el episodio termina, salir del bucle
        if done:
            break

    print(f"Episodio {episode + 1}: Recompensa total ajustada = {total_reward:.2f}")

# Cerrar el entorno
env.close()
