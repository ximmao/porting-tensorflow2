"""
reimplementation of the code for Bayesian Action Decoder with just tf 2.0
modifed from BAD paper colab  https://bit.ly/2P3YOyd
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as K

def repeat_tensor(tensor, repetion):
    """
       tensor is one dimensional (num_t,)
       repetion is a list e.g. [2], to repeat the entire input tensor,
                along extra dimension from dim > 1

    RETURN:
       repeated tensor (*repetion * num_t, )

    EXAMPLE:
       a = tf.constant([1,2,3])
       b = repeat_tensor(a, [2])
       print(b)
       # <tf.Tensor: id=10, shape=(6,), dtype=int32, numpy=array([1, 1, 2, 2, 3, 3], dtype=int32)>
    """

    exp_tensor = tf.expand_dims(tensor, axis=-1)
    tensor_t = tf.tile(exp_tensor, [1]+repetion)
    tensor_r = tf.reshape(tensor_t, repetion * tf.shape(tensor))
    return tensor_r

def get_unique_name(name):
    name_count = K.backend.get_uid(name)
    if name_count > 1:
        name = name + '_' + str(name_count)
    return name

def create_mlp(input_dim, output_dim, num_hidden, activation='relu', name='mlp'):
    """ Create a simple one-hidden layer MLP """
    x = K.Input(input_dim)
    y = K.layers.Dense(units=num_hidden, activation=activation)(x)
    y = K.layers.Dense(units=output_dim, activation=activation)(y)
    return K.Model(x, y, name=get_unique_name(name))

def get_models(bad_mode, payoff_values, batch_size, num_hidden=32):
    """
    """

    # Input is a single number for agent 0 and agent 1.
    # not needed in tf2.0
    #input_0 = tf.placeholder(tf.int32, shape=batch_size)
    #input_1 = tf.placeholder(tf.int32, shape=batch_size)

    # Payoff matrix
    num_cards = payoff_values.shape[0]    # C. python int
    num_actions = payoff_values.shape[-1] # A.
    # use tf constant
    #num_cards = tf.shape(payoff_values)[0]     # C.
    #num_actions = tf.shape(payoff_values)[-1]  # A.
    payoff_tensor = tf.constant(payoff_values)

    # Agent 0
    #with tf.variable_scope('agent_0'):
        #weights_0 = tf.get_variable('weights_0', shape=(num_cards, num_actions))
        #baseline_0_mlp = snt.nets.MLP([num_hidden, 1])

    # initialization of model done outside of the training loop
    # should be glorot uniform

    # policy weights: for one-hot encoding of card, get the action dim=3
    weights_0 = tf.Variable(tf.random.uniform(
                shape=(num_cards, num_actions)),
                name="weights_0")

    net_value_0 = create_mlp(input_dim=num_cards*2, output_dim=1, num_hidden=num_hidden)

    # Agent 1
    if bad_mode == 2:
        num_input_1 = num_cards * 2 + num_actions
    else:
        num_input_1 = num_cards * 3 + num_actions
    net_policy_1 = create_mlp(input_dim=num_input_1 - num_cards, output_dim=num_actions, num_hidden=num_hidden)
    net_value_1 = create_mlp(input_dim=num_input_1, output_dim=1, num_hidden=num_hidden)

    print(net_policy_1.summary())
    print(net_value_1.summary())

    # These are the 'counterfactual inputs', i.e., all possible cards.
    # all_inputs = tf.placeholder(tf.int32, shape=(1, num_cards))

    # Optimizer
    # use the default parameters the same as tf.train.AdamOptimizer in v1
    opt_value_0 = K.optimizers.Adam(epsilon=1e-8)
    opt_policy_0 = K.optimizers.Adam(epsilon=1e-8)
    opt_value_1 = K.optimizers.Adam(epsilon=1e-8)
    opt_policy_1 = K.optimizers.Adam(epsilon=1e-8)

    Param_dict = {}
    Param_dict['opt_value_0'] = opt_value_0
    Param_dict['opt_policy_0'] = opt_policy_0
    Param_dict['opt_value_1'] = opt_value_1
    Param_dict['opt_policy_1'] = opt_policy_1
    Param_dict['weight_policy_0'] = weights_0
    Param_dict['net_value_0'] = net_value_0
    Param_dict['net_policy_1'] = net_policy_1
    Param_dict['net_value_1'] = net_value_1

    return Param_dict

def train(bad_mode, batch_size=32, num_runs=1, num_episodes=5000,
          num_readings=100, seed=42, debug=False):

    payoff_values = np.asarray([
        [
            [[10, 0, 0], [4, 8, 4], [10, 0, 0]],
            [[0, 0, 10], [4, 8, 4], [0, 0, 10]]
        ],
        [
            [[0, 0, 10], [4, 8, 4], [0, 0, 0]],
            [[10, 0, 0], [4, 8, 4], [10, 0, 0]]
        ]
    ], dtype=np.float32)
    num_cards = payoff_values.shape[0] # C.
    num_actions = payoff_values.shape[-1] # A.

    # Set random number generator seeds for reproducibility.
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Get Learnable Parameter & Optimizer
    Param_dict = get_models(bad_mode, payoff_values, batch_size)

    opt_value_0 = Param_dict['opt_value_0']
    opt_policy_0 = Param_dict['opt_policy_0']
    opt_value_1 = Param_dict['opt_value_1']
    opt_policy_1 = Param_dict['opt_policy_1']
    weights_0 = Param_dict['weight_policy_0']
    net_value_0 = Param_dict['net_value_0']
    net_policy_1 = Param_dict['net_policy_1']
    net_value_1 = Param_dict['net_value_1']

    @tf.function
    def train_epoch(input_0, input_1, payoff_tensors):
        with tf.GradientTape(persistent=True) as tape:
            # counterfactual actions
            # ###########
            # All cards, counterfactual inputs

            all_cards = np.zeros((1, num_cards))
            for i in range(num_cards):
                all_cards[0, i] = i
            all_cards_tensor = tf.constant(all_cards, dtype=tf.int32)

            # Repeat for each batch, [bs*num_cards,]
            repeated_in = tf.reshape(
                        repeat_tensor(all_cards_tensor, [batch_size, 1]),
                        [-1])

            # one hot encoding for all the cards: output dim: [bs*num_cards, num_cards]
            one_hot_batch_in = tf.one_hot(repeated_in, num_cards)

            # log softmax of action for for each state (holding one specific card)
            # log_po: [bs*num_cards, num_actions]
            log_p0 = tf.nn.log_softmax(tf.matmul(one_hot_batch_in,
                                                weights_0))

            # after sampling [bs*num_cards, 1]
            # after squeeze [bs*num_cards,], in the last axis it is the action taken, range: 0...num_action
            # action for each cards in each batch
            cf_action = tf.cast(tf.squeeze(tf.random.categorical(log_p0, num_samples=1)),
                                dtype=tf.int32)

            # get log_prob of sampled action [bs*num_cards,], last axis is log_prob of action
            log_cf = tf.reduce_sum(log_p0 * tf.one_hot(cf_action, num_actions), axis=-1)

            # reshaping
            repeated_in = tf.reshape(repeated_in, [batch_size, num_cards])
            cf_action = tf.reshape(cf_action, [batch_size, num_cards])
            log_cf = tf.reshape(log_cf, [batch_size, num_cards])

            # Agent 0
            ############
            # agent0 choose action based on already sampled counterfactual action
            u0 = tf.reduce_sum(cf_action * tf.one_hot(input_0, num_cards, dtype=tf.int32), axis=-1)
            # same for log_prob
            log_p0 = tf.reduce_sum(log_cf * tf.one_hot(input_0, num_cards), axis=-1)

            # sum everything together for joint-action:
            # for each batch with all possible state, all possible action / log-likelihood
            joint_log_p0 = tf.reduce_sum(log_cf, axis=-1)

            # Repeat action for agent 0 through duplicate column (duplicate num_cards times)
            repeated_actions = repeat_tensor(tf.reshape(u0, [batch_size, 1]), [1, num_cards])

            # for each batch, find the state that could result in the observed agent 0 action
            weights = tf.cast(tf.equal(cf_action, repeated_actions), tf.float32)

            # normalized, the probability of state that could lead to agent 0's action
            # as agent 0's state is private, only action is publicly available
            # infer the state based on the actions
            # [batchsize, num_cards]
            # f^a|
            beliefs = tf.cast(tf.divide(weights, tf.reduce_sum(weights, axis=-1, keepdims=True)), tf.float32)

            beliefs = tf.stop_gradient(beliefs)

            # Agent 1 receives beliefs + private ops for agent 1, unless it's
            # the pure policy gradient version.
            if bad_mode == 2:
                joint_in1 = tf.concat([
                tf.one_hot(u0, num_actions, dtype=tf.float32),
                tf.one_hot(input_1, num_cards, dtype=tf.float32),
            ], axis=1)
            else:
                joint_in1 = tf.concat([
                tf.one_hot(u0, num_actions, dtype=tf.float32),
                beliefs,
                tf.one_hot(input_1, num_cards, dtype=tf.float32),
            ], axis=1)
            joint_in1 = tf.reshape(joint_in1, [batch_size, -1])

            baseline_0_input = tf.concat(
                [tf.one_hot(input_0, num_cards),
                tf.one_hot(input_1, num_cards)], axis=1)
            baseline_1_input = tf.concat(
                [tf.one_hot(input_0, num_cards),
                joint_in1], axis=1)

            # Calculate baselines.
            baseline_0 = net_value_0(baseline_0_input, training=True)
            baseline_1 = net_value_1(baseline_1_input, training=True)

            # ensure the size of beliefs
            beliefs = tf.reshape(beliefs, [batch_size, num_cards])

            # calculate log policy for agent 1
            # 1. with policy network, calculate softmax distribution of agent 1's action
            # 2. sample action
            # 3. get the log-likelihood for sampled action
            log_p1_dist = tf.cast(tf.nn.log_softmax(net_policy_1(joint_in1, training=True)), tf.float32)
            u1 = tf.cast(tf.squeeze(tf.random.categorical(log_p1_dist, num_samples=1)),
                                dtype=tf.int32)
            log_p1 = tf.reduce_sum(log_p1_dist * tf.one_hot(u1, num_actions), axis=-1)

            # Getting the rewards is just indexing into the payout matrix for all
            # elements in the batch.
            # This is a fully observable game, so P(f^a) is unnecessary
            rewards = tf.stack([
                payoff_tensors[input_0[i], input_1[i], tf.cast(u0[i], tf.int64), tf.cast(u1[i], tf.int64)]
                for i in range(batch_size)
                ], axis=0)

            # Log-prob used for learning.
            if bad_mode == 1:
                # with counterfactual actions: sum of all possible action log-like
                log_p0_train = joint_log_p0
            else:
                # without counterfactual actions: the sampled action log-like
                log_p0_train = log_p0
            log_p1_train = log_p1

            # calculated Loss
            # Policy-gradient loss.
            pg_final = tf.reduce_mean(
                    (rewards - tf.stop_gradient(baseline_0)) * log_p0_train)
            pg_final += tf.reduce_mean(
                    (rewards - tf.stop_gradient(baseline_1)) * log_p1_train)

            # Baseline loss.
            total_baseline_loss = tf.reduce_mean(tf.square(rewards - baseline_0))
            total_baseline_loss += tf.reduce_mean(tf.square(rewards - baseline_1))

        policy_grad_0 = tape.gradient(pg_final, weights_0)
        policy_grad_1 = tape.gradient(pg_final, net_policy_1.trainable_variables)
        value_grad_0 = tape.gradient(total_baseline_loss, net_value_0.trainable_variables)
        value_grad_1 = tape.gradient(total_baseline_loss, net_value_1.trainable_variables)

        #TODO this not yet works
        opt_policy_0.apply_gradients(
                    zip(policy_grad_0, tf.reshape(weights_0, [-1])))
        opt_policy_1.apply_gradients(
                    zip(policy_grad_1, net_policy_1.trainable_variables))
        opt_value_0.apply_gradients(
                    zip(value_grad_0, net_value_0.trainable_variables))
        opt_value_1.apply_gradients(
                    zip(value_grad_1, net_value_1.trainable_variables))
        del tape

        return rewards


    rewards = np.zeros((num_runs, num_readings + 1))
    interval = num_episodes // num_readings
    for run_id in np.arange(num_runs):
        if run_id % max(num_runs // 10, 1) == 0:
            print('Run {}/{} ...'.format(run_id+1, num_runs))

        for episode_id in np.arange(num_episodes + 1):
            cards_0 = np.random.choice(num_cards, size=batch_size)
            cards_1 = np.random.choice(num_cards, size=batch_size)

            reward = train_epoch(cards_0, cards_1, payoff_values)
            reward = tf.reduce_mean(reward)

            # Maybe save.
            if episode_id % interval == 0:
                rewards[run_id, episode_id // interval] = reward

            # Maybe log.
            if debug and episode_id % (num_episodes // 5) == 0:
                print(episode_id, 'reward:', reward)
    return rewards

if __name__ == "__main__":
    #@title Train.  {vertical-output: true}

    mode_labels = [
        'BAD, no CF gradient',
        'BAD, with CF gradient',
        'Vanilla PG',
    ]

    # Set debug = True to get a faster run (roughly 5 mins) and more printouts.
    debug = False

    if debug:
        num_runs = 3
        num_episodes = 5000
    else:
        num_runs = 30
        num_episodes = 15000
        num_readings = 100

    rewards_by_bad_mode = {}
    for bad_mode in range(3):
        print('Running', mode_labels[bad_mode])
        rewards_by_bad_mode[bad_mode] = train(bad_mode,
                                            num_runs=num_runs,
                                            num_episodes=num_episodes,
                                            num_readings=num_readings,
                                            debug=debug)
        print('')

    #@title Plot training curves.  {vertical-output: true}

    plt.figure(figsize=(10, 5))

    save_every = num_episodes // num_readings
    steps = np.arange(num_readings + 1) * save_every
    for bad_mode in range(3):
        rewards = rewards_by_bad_mode[bad_mode]
        mean = rewards.mean(axis=0)
        sem = rewards.std(axis=0) / np.sqrt(num_runs)
        plt.plot(steps, mean, label=mode_labels[bad_mode])
        plt.fill_between(steps, mean - sem, mean + sem, alpha=0.3)
    plt.ylim(8, 9.7)
    plt.legend()

    pass
