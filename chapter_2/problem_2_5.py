# demo of sample average method for action value estimation
# in a 10-armed bandit problem with non-stationary rewards
import numpy as np
from matplotlib import pyplot as plt

class TenArmBandit():
    def __init__(self, n_arms=10):
        self.n_arms = n_arms
        self.q_true = np.zeros(n_arms)
        self.q_history = []

    def step(self, action):
        rewards = np.random.randn() + self.q_true[action]  # Reward with noise
        self.q_history.append(self.q_true.copy())
        self.q_true += 0.01 * np.random.randn(self.n_arms)
        return rewards
    
    def plot_reward_history(self):
        q_history = np.array(self.q_history)
        for i in range(self.n_arms):
            plt.plot(q_history[:, i], label=f'Action {i}')
        plt.xlabel('Steps')
        plt.ylabel('Estimated Action Value')
        plt.grid(True)
        plt.savefig('./problem_2_5_reward.png', dpi=300, bbox_inches='tight')
        plt.close()
    
        
class Agent():
    def __init__(self, environment, mode='sample_average'):
        self.environment = environment
        self.n_states = environment.n_arms
        self.q_est = np.zeros(self.n_states)  # Optimal action values
        self.action_counts = np.zeros(self.n_states)  # Count of actions taken
        self.expected_rewards = [] # Store rewards
        self.mode = mode
        
    def update_estimates(self, action, reward):
        # sample average method
        self.action_counts[action] += 1
        self.q_est[action] += 1 / self.action_counts[action] * (reward - self.q_est[action])
        
    def update_estimates_exponential(self, action, reward, alpha=0.1):
        # exponential recency-weighted average method
        self.q_est[action] += alpha * (reward - self.q_est[action])
        
    def select_action(self, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_states)  # Explore
        else:
            return np.argmax(self.q_est)  # Exploit

    def record_expected_reward(self, epsilon):
        # Record expected rewards for plotting
        rewards = []
        for _ in range(100):
            action = self.select_action(epsilon)
            reward = self.environment.step(action)
            rewards.append(reward)
        self.expected_rewards.append(np.mean(rewards))
        
    def run(self, n_steps=10000, epsilon=0.1):
        self.record_expected_reward(epsilon)
        # run the agent 
        for _ in range(n_steps):
            action = self.select_action(epsilon)
            reward = self.environment.step(action)
            if self.mode == 'sample_average':
                self.update_estimates(action, reward)
            elif self.mode == 'exponential':
                self.update_estimates_exponential(action, reward)
            self.record_expected_reward(epsilon)
        return self.expected_rewards
        
        
if __name__ == "__main__":
    environment = TenArmBandit(n_arms=10)
    agent = Agent(environment, mode='sample_average')
    rewards = agent.run(n_steps=10000, epsilon=0.1)
    environment.plot_reward_history()

    plt.plot(rewards, label='Sample Average')
    
    environment = TenArmBandit(n_arms=10)
    agent = Agent(environment, mode='exponential')
    rewards = agent.run(n_steps=10000, epsilon=0.1)

    plt.plot(rewards, label='Exponential Recency')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.title('Average Reward Over Time')
    plt.savefig('./problem_2_5.png')
    plt.close()

