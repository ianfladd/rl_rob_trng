"""
SARSA Agent Implementation for Robot Trajectory Optimization.
"""

import numpy as np
import pickle
from typing import Dict, Tuple, Optional
from collections import defaultdict

from .state_action import StateSpace, ActionSpace, SimplifiedActionSpace, RobotState


class SARSAAgent:
    """
    SARSA (State-Action-Reward-State-Action) agent for tabular RL.

    SARSA is an on-policy TD control algorithm that updates Q-values
    based on the actual action taken in the next state.

    Update rule:
    Q(s,a) <- Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]
    """

    def __init__(
        self,
        state_space: StateSpace,
        action_space: ActionSpace,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        self.state_space = state_space
        self.action_space = action_space

        # Hyperparameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: maps (discrete_state, action) -> Q-value
        # Using defaultdict for sparse representation
        self.q_table: Dict[Tuple, float] = defaultdict(float)

        # Statistics
        self.episode_count = 0
        self.total_steps = 0
        self.episode_rewards = []

    def get_discrete_state(self, state: RobotState) -> Tuple[int, ...]:
        """Convert continuous state to discrete representation."""
        return self.state_space.discretize(state)

    def select_action(self, state: RobotState, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current robot state
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            Action index
        """
        discrete_state = self.get_discrete_state(state)

        # Epsilon-greedy action selection
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return self.action_space.sample_random()
        else:
            # Exploit: best known action
            return self._get_best_action(discrete_state)

    def _get_best_action(self, discrete_state: Tuple[int, ...]) -> int:
        """Get the action with highest Q-value for given state."""
        best_action = 0
        best_value = float('-inf')

        for action_idx in range(self.action_space.num_actions):
            q_value = self.q_table[(discrete_state, action_idx)]
            if q_value > best_value:
                best_value = q_value
                best_action = action_idx

        return best_action

    def update(
        self,
        state: RobotState,
        action: int,
        reward: float,
        next_state: RobotState,
        next_action: int,
        done: bool
    ):
        """
        SARSA update rule.

        Q(s,a) <- Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]
        """
        discrete_state = self.get_discrete_state(state)
        discrete_next_state = self.get_discrete_state(next_state)

        # Current Q-value
        current_q = self.q_table[(discrete_state, action)]

        # Target Q-value
        if done:
            target_q = reward
        else:
            next_q = self.q_table[(discrete_next_state, next_action)]
            target_q = reward + self.gamma * next_q

        # TD error
        td_error = target_q - current_q

        # Update Q-value
        self.q_table[(discrete_state, action)] = current_q + self.alpha * td_error

        self.total_steps += 1

    def decay_epsilon(self):
        """Decay exploration rate after episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_count += 1

    def get_q_value(self, state: RobotState, action: int) -> float:
        """Get Q-value for state-action pair."""
        discrete_state = self.get_discrete_state(state)
        return self.q_table[(discrete_state, action)]

    def save(self, filepath: str):
        """Save agent to file."""
        data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'episode_rewards': self.episode_rewards,
            'alpha': self.alpha,
            'gamma': self.gamma,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath: str):
        """Load agent from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.q_table = defaultdict(float, data['q_table'])
        self.epsilon = data['epsilon']
        self.episode_count = data['episode_count']
        self.total_steps = data['total_steps']
        self.episode_rewards = data.get('episode_rewards', [])

    def get_stats(self) -> Dict:
        """Get training statistics."""
        return {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'avg_reward_last_100': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
        }


class SARSALambdaAgent(SARSAAgent):
    """
    SARSA(λ) agent with eligibility traces.

    Eligibility traces provide a bridge between TD and Monte Carlo methods,
    allowing for faster credit assignment.
    """

    def __init__(
        self,
        state_space: StateSpace,
        action_space: ActionSpace,
        lambda_: float = 0.9,
        **kwargs
    ):
        super().__init__(state_space, action_space, **kwargs)
        self.lambda_ = lambda_

        # Eligibility traces
        self.e_traces: Dict[Tuple, float] = defaultdict(float)

    def update(
        self,
        state: RobotState,
        action: int,
        reward: float,
        next_state: RobotState,
        next_action: int,
        done: bool
    ):
        """SARSA(λ) update with eligibility traces."""
        discrete_state = self.get_discrete_state(state)
        discrete_next_state = self.get_discrete_state(next_state)

        # TD error
        current_q = self.q_table[(discrete_state, action)]
        if done:
            target_q = reward
        else:
            next_q = self.q_table[(discrete_next_state, next_action)]
            target_q = reward + self.gamma * next_q

        td_error = target_q - current_q

        # Update eligibility trace for current state-action
        self.e_traces[(discrete_state, action)] += 1.0

        # Update all state-action pairs
        keys_to_delete = []
        for key, trace in self.e_traces.items():
            # Update Q-value
            self.q_table[key] += self.alpha * td_error * trace

            # Decay trace
            new_trace = self.gamma * self.lambda_ * trace
            if new_trace < 0.001:
                keys_to_delete.append(key)
            else:
                self.e_traces[key] = new_trace

        # Remove negligible traces
        for key in keys_to_delete:
            del self.e_traces[key]

        self.total_steps += 1

    def reset_traces(self):
        """Reset eligibility traces at episode start."""
        self.e_traces.clear()


class ExpectedSARSAAgent(SARSAAgent):
    """
    Expected SARSA agent.

    Instead of using the actual next action, uses the expected value
    over all possible next actions weighted by their policy probabilities.
    """

    def update(
        self,
        state: RobotState,
        action: int,
        reward: float,
        next_state: RobotState,
        next_action: int,  # Not used in Expected SARSA
        done: bool
    ):
        """Expected SARSA update rule."""
        discrete_state = self.get_discrete_state(state)
        discrete_next_state = self.get_discrete_state(next_state)

        current_q = self.q_table[(discrete_state, action)]

        if done:
            target_q = reward
        else:
            # Compute expected Q-value under epsilon-greedy policy
            expected_q = self._compute_expected_q(discrete_next_state)
            target_q = reward + self.gamma * expected_q

        td_error = target_q - current_q
        self.q_table[(discrete_state, action)] = current_q + self.alpha * td_error

        self.total_steps += 1

    def _compute_expected_q(self, discrete_state: Tuple[int, ...]) -> float:
        """Compute expected Q-value under epsilon-greedy policy."""
        num_actions = self.action_space.num_actions

        # Get Q-values for all actions
        q_values = np.array([
            self.q_table[(discrete_state, a)] for a in range(num_actions)
        ])

        # Best action probability: (1 - epsilon) + epsilon/num_actions
        best_action = np.argmax(q_values)
        best_prob = (1 - self.epsilon) + self.epsilon / num_actions

        # Other actions probability: epsilon/num_actions
        other_prob = self.epsilon / num_actions

        # Compute expected value
        expected_q = 0.0
        for a in range(num_actions):
            prob = best_prob if a == best_action else other_prob
            expected_q += prob * q_values[a]

        return expected_q
