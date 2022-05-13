import retro
from agent import *







"""Tamaño de la entrada:
    224  320  3
    height, width, channels
    
número de acciones posibles:
    8
    [izq,der]
"""


if __name__ == '__main__':
    env = retro.make(game="MortalKombat3-Genesis",
                    state="Level1.ShangTsungVsLiuKang",
                    scenario="scenario")
    obs = env.reset()
    height, width, channels = env.observation_space.shape
    #print("", height, "", width, "", channels)
    actions = env.action_space.n
    agent = build_agent(build_model(actions),actions)
    try:
        agent.load_weights("mkiii.h5")
    except:
        FileNotFoundError("")
    agent.compile(optimizer=adam_v2.Adam(lr=1e-3), metrics=['mae'])
    
    agent.fit(env=env, nb_steps=1000000, visualize=True, verbose=2)

"""
while True:
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
        env.close()
        agent.save_weights("mkiii.h5", overwrite=True)
"""









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
