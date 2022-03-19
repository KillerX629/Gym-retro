import retro

env = retro.make(game="MortalKombat3-Genesis", state="Level1.ShangTsungVsLiuKang")
obs = env.reset()

while True:
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
        env.close()