import retro
from transformers import DecisionTransformerModel
import torch






"""Tamaño de la entrada:
    224  320  3
    height, width, channels
    
número de acciones posibles:
    8
    [izq,der]
"""


if __name__ == '__main__':
    env = retro.make(game="MortalKombat3-Genesis",
                    state="Level1.ShangTsungVsLiuKang")
    obs = env.reset()
    height, width, channels = env.observation_space.shape
    print("", height, "", width, "", channels)
    actions = env.action_space.n
    
    #dqn.fit(env, nb_steps=1000000, visualize=True, verbose=2)fit entrena el modelo

    while True:
        print("", height, "", width, "", channels)
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
            env.close()


    """
    este código es una de las formas en las que podemos conseguir guardar
    los píxeles de la pantalla en variables.
    
    alto, ancho, canales = obs.shape
    
    acciones = env.action_space.n
    
    
    para conocer las acciones disponibles en el juego:
    env.unwrapped.get_action_meanings()
    """
    
    """
while True:
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
        env.close()
"""
# Function that gets an action from the model using autoregressive prediction
# with a window of the previous 20 timesteps.


def get_action(model, states, actions, rewards, returns_to_go, timesteps):
    # This implementation does not condition on past rewards

    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    # The prediction is conditioned on up to 20 previous time-steps
    states = states[:, -model.config.max_length:]
    actions = actions[:, -model.config.max_length:]
    returns_to_go = returns_to_go[:, -model.config.max_length:]
    timesteps = timesteps[:, -model.config.max_length:]

    # pad all tokens to sequence length, this is required if we process batches
    padding = model.config.max_length - states.shape[1]
    attention_mask = torch.cat(
        [torch.zeros(padding), torch.ones(states.shape[1])])
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
    states = torch.cat(
        [torch.zeros((1, padding, state_dim)), states], dim=1).float()
    actions = torch.cat(
        [torch.zeros((1, padding, act_dim)), actions], dim=1).float()
    returns_to_go = torch.cat(
        [torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
    timesteps = torch.cat(
        [torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)

    # perform the prediction
    state_preds, action_preds, return_preds = model(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,)
    return action_preds[0, -1]
