"""
rl_agent.py
Contextual bandit agent for adaptive decision threshold tuning.
Uses epsilon-greedy exploration over a discretized threshold action space.
Reward function penalizes false negatives more heavily than false positives,
reflecting the asymmetry of the misinformation detection task.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

THRESHOLDS   = np.round(np.arange(0.1, 1.0, 0.1), 2)
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")


def get_reward(pred_label, true_label, threshold, max_prob):
    """
    Asymmetric reward function.
    Correct prediction      : +1.0
    False negative          : -2.0  (missed misinformation)
    False positive          : -0.5
    Low confidence abstain  : -0.1
    """
    if max_prob < threshold:
        return -0.1
    if pred_label == true_label:
        return 1.0
    if true_label in [0, 1] and pred_label not in [0, 1]:
        return -2.0
    return -0.5


class ContextualBandit:
    def __init__(self, n_actions, epsilon=0.1):
        self.n_actions     = n_actions
        self.epsilon       = epsilon
        self.q_values      = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)

    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_values)

    def update(self, action, reward):
        self.action_counts[action] += 1
        self.q_values[action] += (
            (reward - self.q_values[action]) /
            self.action_counts[action]
        )


def train_bandit(probs, true_labels, n_episodes=500, epsilon=0.1, seed=42):
    """
    Train the contextual bandit agent.
    probs       : array of shape (n_samples, n_classes) — softmax outputs
    true_labels : array of shape (n_samples,)
    """
    np.random.seed(seed)
    n_samples = len(true_labels)
    agent     = ContextualBandit(n_actions=len(THRESHOLDS), epsilon=epsilon)

    cumulative_rewards = []
    running_reward     = 0

    for episode in range(n_episodes):
        idx        = np.random.randint(n_samples)
        prob       = probs[idx]
        true_label = true_labels[idx]
        action     = agent.select_action()
        threshold  = THRESHOLDS[action]
        pred_label = np.argmax(prob)
        max_prob   = np.max(prob)
        reward     = get_reward(pred_label, true_label, threshold, max_prob)
        agent.update(action, reward)
        running_reward += reward
        cumulative_rewards.append(running_reward)

    return agent, cumulative_rewards


def save_results(agent, cumulative_rewards, fixed_rewards):
    """Save bandit results to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {
        "model": "Contextual Bandit (epsilon-greedy)",
        "n_episodes": len(cumulative_rewards),
        "epsilon": 0.1,
        "best_threshold": float(THRESHOLDS[np.argmax(agent.q_values)]),
        "q_values": {
            str(round(t, 1)): round(q, 4)
            for t, q in zip(THRESHOLDS, agent.q_values)
        },
        "action_counts": {
            str(round(t, 1)): int(c)
            for t, c in zip(THRESHOLDS, agent.action_counts)
        },
        "note": "note": "Bandit agent trained on simulated softmax outputs sampled from "
                        "a Dirichlet distribution. Re-evaluation with real DistilBERT "
                        "outputs is left for future work."
    }
    with open(os.path.join(RESULTS_DIR, 'bandit_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("Bandit results saved!")


if __name__ == "__main__":
    print("Running bandit agent with simulated probabilities...")
    n_samples   = 1267
    sim_probs   = np.random.dirichlet(np.ones(6), size=n_samples)
    true_labels = np.random.randint(0, 6, size=n_samples)

    agent, cumulative_rewards = train_bandit(sim_probs, true_labels)

    fixed_rewards  = []
    running_fixed  = 0
    for _ in range(500):
        idx        = np.random.randint(n_samples)
        prob       = sim_probs[idx]
        true_label = true_labels[idx]
        reward     = get_reward(np.argmax(prob), true_label, 0.5, np.max(prob))
        running_fixed += reward
        fixed_rewards.append(running_fixed)

    best = THRESHOLDS[np.argmax(agent.q_values)]
    print(f"Best learned threshold: {best}")

    plt.figure(figsize=(10, 4))
    plt.plot(cumulative_rewards, label='Bandit Agent', color='steelblue')
    plt.plot(fixed_rewards, label='Fixed Threshold (0.5)',
             color='darkorange', linestyle='--')
    plt.title('Contextual Bandit vs Fixed Threshold')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'bandit_learning_curves.png'))
    plt.show()

    save_results(agent, cumulative_rewards, fixed_rewards)
