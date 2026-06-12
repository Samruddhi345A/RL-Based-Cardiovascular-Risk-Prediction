import pandas as pd

df = pd.read_csv('heart.csv')  # upload in Colab
df.head()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Convert text → numbers
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Features & target
X = df.drop('target', axis=1)
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9. RL ENVIRONMENT
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Train environment model
env_model = RandomForestClassifier(n_estimators=100, random_state=42)
env_model.fit(X_train, y_train)

n_states = len(X_train)
n_actions = 2

# Initialize Q-table
q_table = np.zeros((n_states, n_actions))

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2

# ENVIRONMENT FUNCTION
def step(state_idx, action):
    patient = X_train[state_idx].reshape(1, -1)

    predicted = env_model.predict(patient)[0]

    # Reward logic
    if action == predicted:
        reward = +10
    else:
        reward = -10

    # Next state (random transition)
    next_state = np.random.randint(0, n_states)

    return next_state, reward

# 10. Q-LEARNING TRAINING
episode_rewards = []

for episode in range(300):
    state = np.random.randint(0, n_states)
    total_reward = 0

    for step_count in range(10):

        # Action selection (epsilon-greedy)
        if np.random.rand() < epsilon:
            action = np.random.randint(0, n_actions)
        else:
            action = np.argmax(q_table[state])

        # Environment step
        next_state, reward = step(state, action)
        total_reward += reward

        # Q-learning update
        q_table[state, action] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )

        state = next_state

    episode_rewards.append(total_reward)

print("\nQ-learning completed!")

# 11. SAMPLE POLICY OUTPUT
print("\nSample Learned Actions (first 10 states):")
for i in range(10):
    print(f"State {i} → Best Action: {np.argmax(q_table[i])}")

# 12. KNN-BASED RL INFERENCE (k=1)
from sklearn.metrics import classification_report

y_pred_rl = []

for i in range(len(X_test)):
    test_sample = X_test[i]

    # Compute Euclidean distance to every training sample
    distances = np.sqrt(np.sum((X_train - test_sample) ** 2, axis=1))

    # Find the nearest training state (k=1)
    nearest_state = np.argmin(distances)

    # Pick the best action from Q-table at that nearest state
    action = np.argmax(q_table[nearest_state])
    y_pred_rl.append(action)

rl_accuracy = accuracy_score(y_test, y_pred_rl)
print("\nRL Accuracy (KNN-based inference):", rl_accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_rl, target_names=["No Disease", "Disease"]))

# PRINT Q-TABLE (first 10 + last 5 states)
selected_states = list(range(10)) + list(range(n_states - 5, n_states))

print("\nQ-Table:")
rows = []
for i in selected_states:
    rows.append([round(q_table[i][0], 8), round(q_table[i][1], 8)])

print(np.array(rows))

# 13. REWARD vs EPISODES GRAPH

plt.figure()
plt.plot(episode_rewards)

plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("RL Learning Curve (Reward vs Episodes)")

plt.show()

# 14. SMOOTHED GRAPH (BETTER VISUAL)

window = 10
smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')

plt.figure()
plt.plot(smoothed)

plt.xlabel("Episodes")
plt.ylabel("Smoothed Reward")
plt.title("Smoothed RL Learning Curve")

plt.show()