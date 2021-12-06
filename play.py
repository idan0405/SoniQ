import retro
import numpy as np
from DQNAgent import DQNAgent
from SonicDiscretizer import SonicDiscretizer


if __name__ == "__main__":
    env = retro.make(game="SonicTheHedgehog-Genesis", state="GreenHillZone.Act2")
    env = SonicDiscretizer(env)
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, env)
    agent.load("sonic-dqn1.h5")
    agent.epsilon = 0
    done = False
    while True:
        state = env.reset()
        _ = {"lives": 3, "x": 0}
        while not done:
            action = np.argmax(agent.model.predict(np.array([state/255.0]))[0])
            for i in range(5):
                env.render()
                next_state, reward, done, _ = env.step(action)
            state = next_state

