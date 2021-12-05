import retro
from DQNAgent import DQNAgent
from SonicDiscretizer import SonicDiscretizer

EPISODES = 200000

if __name__ == "__main__":
    env = retro.make(game="SonicTheHedgehog-Genesis", state="GreenHillZone.Act2")
    env = SonicDiscretizer(env)
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, env)
    #agent.load("sonic-dqn1.h5")
    done = False
    batch_size = 500
    for e in range(1, EPISODES):
        state = env.reset()
        _ = {"lives": 3, "x": 0}
        for time_t in range(1, 3000):
            action = agent.act(state / 255.0)
            x = float(_["x"])
            lives = _["lives"]
            for i in range(5):

                if agent.epsilon <= 0.01 and i % 10 == 0:
                    env.render()
                    agent.epsilon_decay = 0.9995

                next_state, reward, done, _ = env.step(action)
            if done or lives > _["lives"]:
                done = True
                reward = -1
                for i in range(10):
                    agent.remember(state / 255.0, action, reward, next_state / 255.0, done)
            else:
                reward = (float(_["x"]) - x) / 50.0
            agent.remember(state / 255.0, action, reward, next_state / 255.0, done)
            state = next_state
        print("episode: {}/{}, e: {:.2}, distance: {}".format(e, EPISODES, agent.epsilon, _["x"]))
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            agent.save("sonic-dqn1.h5")
