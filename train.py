import random
import gym
import retro
import numpy as np
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam

EPISODES = 200000
retro.data.list_games()
sys_details = tf.sysconfig.get_build_info()
cuda_version = sys_details["cuda_version"]


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.8
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.99
        self.learning_rate = 1e-8
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(512, kernel_size=(7, 7),
                         activation='relu',
                         input_shape=state_size))
        model.add(MaxPooling2D(pool_size=(5, 5)))
        model.add(Dropout(0.25))
        model.add(Conv2D(512, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(5, 5)))
        model.add(Flatten())
        model.add(Dropout(0.25))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        act_values = self.model.predict(np.array([state]))
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = []
            times = 1
            if not done:
                for i in range(self.action_size):
                    if action == i:
                        target.append(reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0]))
                    else:
                        target.append(self.model.predict(np.array([state]))[0][i])


            else:
                times = 1
                for i in range(self.action_size):
                    if action == i:
                        target.append(reward)
                    else:
                        target.append(self.model.predict(np.array([state]))[0][i])
            self.model.fit(np.array([state]), np.array([target]), epochs=times, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """

    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()


if __name__ == "__main__":
    env = retro.make(game="SonicTheHedgehog-Genesis", state="GreenHillZone.Act2")
    env = SonicDiscretizer(env)
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    print(state_size)
    print(action_size)
    agent = DQNAgent(state_size, action_size)
    agent.model.built = True
    # agent.load("sanic-dqn1.h5")
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
            agent.save("sanic-dqn1.h5")
