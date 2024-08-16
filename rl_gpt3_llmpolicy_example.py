import openai
import gym
import numpy as np

# Configura tu clave de API de OpenAI
client = openai.OpenAI(api_key="")

# Función para obtener una acción del modelo GPT-3.5 Turbo
def get_gpt_action(observation):
    prompt = f"CartPole observation: {observation}\nAction (0 or 1):"
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI trained to play CartPole. You must to specify an action (0 or 1) based on the given observation. 0 means push cart to the left, 1 means push cart to the right."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1,
        n=1,
        stop=None,
        temperature=0.5,
    )
    
    action_str = response.choices[0].message.content
    
    try:
        action = int(action_str)
    except ValueError:
        print(f"Invalid action received from GPT-3.5 Turbo: {action_str}")
        action = 0  # Default action
    
    return action

# Crear el entorno CartPole
env = gym.make("CartPole-v1")

# Interactuar con el entorno utilizando GPT-3.5 Turbo como política
total_episodes = 50
for episode in range(total_episodes):
    observation = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Obtener acción del modelo GPT-3.5 Turbo
        action = get_gpt_action(observation)
        
        # Tomar la acción en el entorno
        observation, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        if done:
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")
            break

env.close()