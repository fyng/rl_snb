# demo of epsilon-greedy action selection
# in a 10-armed bandit problem with stationary rewards
# using sample average method for action value estimation

import numpy as np
from matplotlib import pyplot as plt

class TenArmBandit():
    def __init__(self, n_arms=10):
        self.n_arms = n_arms
        self.q_true = np.random.randn(n_arms)  # True action values

    def step(self, action):
        return np.random.randn() + self.q_true[action]  # Reward with noise
    
    def plot_reward_distribution(self):
        n_samples = 1000
        reward_data = []
        action_labels = []
    
        for action in range(self.n_arms):
            rewards = np.random.randn(n_samples) + self.q_true[action]
            reward_data.extend(rewards)
            action_labels.extend([f'Action {action}'] * n_samples)
        
        # Prepare data for violin plot
        data_by_action = [np.random.randn(n_samples) + self.q_true[i] for i in range(self.n_arms)]
        parts = plt.violinplot(data_by_action, positions=range(self.n_arms), showmeans=True, showmedians=True)

        plt.xlabel('Action')
        plt.ylabel('Reward Distribution')
        plt.title('Reward Distribution for 10-Armed Bandit')
        plt.xticks(range(self.n_arms))
        plt.grid(True, alpha=0.3)        
        plt.scatter(range(self.n_arms), self.q_true, color='red', s=50, zorder=3, label='True Values')
        plt.legend()        
        plt.savefig('./problem_2_3_reward_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

class Agent():
    def __init__(self, environment):
        self.environment = environment
        self.n_states = environment.n_arms
        self.q_est = np.zeros(self.n_states)  # Optimal action values
        self.action_counts = np.zeros(self.n_states)  # Count of actions taken
        self.expected_rewards = [] # Store rewards
        
    def update_estimates(self, action, reward):
        # sample average method
        self.action_counts[action] += 1
        self.q_est[action] += 1 / self.action_counts[action] * (reward - self.q_est[action])

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
        
    def run(self, n_steps=1000, epsilon=0.1):
        self.record_expected_reward(epsilon)

        # run the agent 
        for _ in range(n_steps):
            action = self.select_action(epsilon)
            reward = self.environment.step(action)
            self.update_estimates(action, reward)
            self.record_expected_reward(epsilon)
        return self.expected_rewards
        
        
if __name__ == "__main__":
    environment = TenArmBandit(n_arms=10)
    environment.plot_reward_distribution()
    
    for eps in [0.0, 0.1, 0.01]:
        agent = Agent(environment)
        rewards = agent.run(n_steps=1000, epsilon=eps)
        plt.plot(rewards, label=f'Epsilon: {eps}')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.title('Average Reward Over Time')
    plt.savefig('./problem_2_3.png')
    plt.close()

