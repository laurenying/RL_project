import numpy as np
from keras import layers, models, optimizers
from keras import backend as K

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high, net_setting):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        # self.num_hlayers = net_setting["num_hlayers"]
        self.num_hlayers = 2
        self.num_hunits = net_setting["num_hunits"]
        self.activation = "relu"
        self.lr = net_setting["lr"]

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = layers.Dense(units=self.num_hunits, kernel_regularizer=layers.regularizers.l2(1e-6))(states)
        net = layers.BatchNormalization()(net)
        net = layers.Activation(self.activation)(net)
        net = layers.Dense(units=self.num_hunits, kernel_regularizer=layers.regularizers.l2(1e-6))(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation(self.activation)(net)

        # net = layers.Dense(units=self.num_hunits, activation=self.activation, 
        #                    kernel_regularizer=layers.regularizers.l2(1e-6))(states)
        # for lay in range(self.num_hlayers - 1):
        #     net = layers.Dense(units=self.num_hunits, activation=self.activation, 
        #                        kernel_regularizer=layers.regularizers.l2(1e-6))(net)


        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='tanh', 
            name='raw_actions', kernel_initializer=layers.initializers.RandomUniform(minval=-3e-3, maxval=3e-3))(net)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=raw_actions)

        # Scale [0, 1] output for each action dimension to proper range
        # actions = layers.Lambda(lambda x: x * self.action_range + self.action_low, name='actions')(raw_actions)
        # action_low = K.constant(self.action_low)
        # action_range = K.constant(self.action_range)
        # actions = self.model.output * action_range

        # Define loss function using action value (Q value) gradients
        Q_gradients = K.placeholder(shape=(self.action_size,), name="Q_gradient_wrt_action")
        loss = K.mean(-Q_gradients * raw_actions, axis=0)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=self.lr)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, Q_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)


class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, net_setting):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_hlayers = 2
        self.num_hunits = net_setting["num_hunits"]
        self.activation = "relu"
        self.lr = net_setting["lr"]

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=self.num_hunits, kernel_regularizer=layers.regularizers.l2(1e-6))(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation("relu")(net_states)
        net_states = layers.Dense(units=self.num_hunits, kernel_regularizer=layers.regularizers.l2(1e-6))(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=self.num_hunits, kernel_regularizer=layers.regularizers.l2(1e-6))(actions)

        # net_states = layers.Dense(units=self.num_hunits, activation=self.activation, 
        #     kernel_regularizer=layers.regularizers.l2(1e-6))(states)
        # net_states = layers.Dense(units=self.num_hunits // 2, kernel_regularizer=layers.regularizers.l2(1e-6))(net_states)
        # net_actions = layers.Dense(units=self.num_hunits // 2, kernel_regularizer=layers.regularizers.l2(1e-6))(actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation(self.activation)(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values', 
            kernel_initializer=layers.initializers.RandomUniform(minval=-3e-3, maxval=3e-3))(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.lr)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
        