import numpy as np
import keras

from AC import Actor, Critic
from replayBuffer import ReplayBuffer
from ouNoise import OUNoise



class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, 
        env, # learning environment
        task, # environment parameters
        actor_setting, # actor network parameters
        critic_setting, # actor network parameters
        noise, # noise parameters
        memory, # memory parameters
        gamma, # discount factor
        tau, # soft update parameter
        num_training, # number of training episodes
        log_path, 
        actor_save_path, # actor model save path
        critic_save_path # critic model save path
        ):
        self.env = env
        self.state_size = task["state_size"]
        self.action_size = task["action_size"]
        # self.action_range = task["action_range"] # {dX: [-1, 1], dY: [-1, 1], dZ: [-1, 1], dA: [-4, 4]}
        self.action_low = task["action_low"]
        self.action_high = task["action_high"]
        # initially train the model using range = 1
        # self.action_high = task["action_high"]

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high, actor_setting)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high, actor_setting)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size, critic_setting)
        self.critic_target = Critic(self.state_size, self.action_size, critic_setting)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = noise["mu"]
        self.exploration_theta = noise["theta"]
        self.exploration_sigma = noise["sigma"]
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = memory["buffer_size"]
        self.batch_size = memory["batch_size"]
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = gamma  # discount factor
        self.tau = tau  # for soft update of target parameters
        self.training_episodes = num_training # number of training episodes

        self.rewards_ls = [] # store training rewards
        self.log_path = log_path
        self.actor_save_path = actor_save_path
        self.critic_save_path = critic_save_path

        self.last_state = None

    def reset_episode(self):
        self.noise.reset()
        self.env.reset()
        state = self.env.get_feature_vec_observation()
        state = np.concatenate([np.array(state[:9]), keras.utils.to_categorical(state[-1], num_classes=9)]) # 
        self.last_state = state
        # print("reset to starting state: ", state)
        return state

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

    def train(self):
        with open(self.log_path, "w") as f:
            f.write("episodes, reward\n")
            episode = 1
            episode_rewards_p100 = 0
            while episode <= self.training_episodes:
                episode_reward = 0
                done = False
                state = self.reset_episode()
                while not done:
                    action = self.act(self.last_state)
                    next_state, reward, done, info = self.env.step(action)
                    next_state = self.env.get_feature_vec_observation()
                    next_state = np.concatenate([np.array(next_state[:9]), 
                        keras.utils.to_categorical(next_state[-1], num_classes=9)])
                    # agent save eperiences, and update networks
                    self.step(action, reward, next_state, done)
                    self.rewards_ls.append(reward)

                    episode_rewards_p100 += reward
                    episode_reward += reward

                metrice = "%d, %f\n"%(episode, episode_reward)
                f.write(metrice)
                if not episode % 100:
                    print("episode: {}, ave_reward: {}".format(episode, episode_rewards_p100 / 100))
                    episode_rewards_p100 = 0
                episode += 1

                if not episode % 1000:
                    self.critic_target.model.save_weights(self.critic_save_path)
                    self.actor_target.model.save_weights(self.actor_save_path)

        return self.rewards_ls



