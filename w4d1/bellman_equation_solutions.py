import numpy as np
Arr = np.ndarray

class Environment:
    def __init__(self, num_states: int, num_actions: int, start=0, terminal=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.start = start
        if terminal is None:
            self.terminal = np.array([], dtype=int)
        else:
            self.terminal = terminal
        (self.T, self.R) = self.build()

    def build(self):
        '''
        Constructs the T and R tensors from the dynamics of the environment.
        Outputs:
            T : (num_states, num_actions, num_states) State transition probabilities
            R : (num_states, num_actions, num_states) Reward function
        '''
        num_states = self.num_states
        num_actions = self.num_actions
        T = np.zeros((num_states, num_actions, num_states))
        R = np.zeros((num_states, num_actions, num_states))
        for s in range(num_states):
            for a in range(num_actions):
                (states, rewards, probs) = self.dynamics(s, a)
                (all_s, all_r, all_p) = self.out_pad(states, rewards, probs)
                T[s, a, all_s] = all_p
                R[s, a, all_s] = all_r
        return (T, R)

    def dynamics(self, state: int, action: int) -> tuple[Arr, Arr, Arr]:
        '''
        Computes the distribution over possible outcomes for a given state
        and action.
        Inputs:
            state : int (index of state)
            action : int (index of action)
        Outputs:
            states  : (m,) all the possible next states
            rewards : (m,) rewards for each next state transition
            probs   : (m,) likelihood of each state-reward pair
        '''
        raise NotImplementedError

    def render(pi: Arr):
        '''
        Takes a policy pi, and draws an image of the behavior of that policy,
        if applicable.
        Inputs:
            pi : (num_actions,) a policy
        Outputs:
            None
        '''
        raise NotImplementedError

    def out_pad(self, states: Arr, rewards: Arr, probs: Arr):
        '''
        Inputs:
            states  : (m,) all the possible next states
            rewards : (m,) rewards for each next state transition
            probs   : (m,) likelihood of each state-reward pair
        Outputs:
            states  : (num_states,) all the next states
            rewards : (num_states,) rewards for each next state transition
            probs   : (num_states,) likelihood of each state-reward pair (including
                           probability zero outcomes.)
        '''
        out_s = np.arange(self.num_states)
        out_r = np.zeros(self.num_states)
        out_p = np.zeros(self.num_states)
        for i in range(len(states)):
            idx = states[i]
            out_r[idx] += rewards[i]
            out_p[idx] += probs[i]
        return (out_s, out_r, out_p)

def policy_eval_exact(env: Environment, pi: Arr, gamma=0.99) -> Arr:
    N = env.num_states
    idx = np.arange(N)
    p_i_j = env.T[idx,pi[idx],:]
    r_i_j = env.R[idx,pi[idx],:]
    r_i_pi = (p_i_j @ r_i_j.T).diagonal()
    I = np.eye(N)

    return np.linalg.inv(I - gamma*p_i_j) @ r_i_pi


def policy_improvement(env: Environment, V: Arr, gamma=0.99) -> Arr:
    '''
    Inputs:
        env: Environment
        V  : (num_states,) value of each state following some policy pi
    Outputs:
        pi_better : vector (num_states,) of actions representing a new policy obtained via policy iteration
    '''
    return (env.T * (env.R + V)).sum(axis = 2).argmax(axis = 1)

def find_optimal_policy(env: Environment, gamma=0.99):
    '''
    Inputs:
        env: environment
    Outputs:
        pi : (num_states,) int, of actions represeting an optimal policy
    '''
    old_pi = np.random.randint(0, env.num_actions, env.num_states)
    new_pi = np.random.randint(0, env.num_actions, env.num_states)
    while (old_pi != new_pi).any():
        old_pi = new_pi
        val = policy_eval_exact(env, new_pi, gamma)
        new_pi = policy_improvement(env, val, gamma)

    return new_pi
