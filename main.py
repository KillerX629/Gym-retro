import retro
#from agent import *





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
