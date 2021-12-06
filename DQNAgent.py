import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size, env):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.8
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.99
        self.learning_rate = 1e-7
        self.env = env
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(512, kernel_size=(7, 7),
                         activation='relu',
                         input_shape=self.state_size))
        model.add(MaxPooling2D(pool_size=(6, 6)))
        model.add(Dropout(0.25))
        model.add(Conv2D(1024, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Flatten())
        model.add(Dropout(0.25))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
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
