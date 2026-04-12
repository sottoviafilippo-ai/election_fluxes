import numpy as np
from PSO import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

def simulate_voting_district(election1: np.ndarray, transfer_matrix: np.ndarray, precision = 500):
    """ Simulates election2 results given election1 and the transfer matrix, with some uncertaninty controlled by precision"""
    """ For a single voting district"""
    targets = np.dot(transfer_matrix, election1)
    return np.random.dirichlet(targets * precision)

def simulate_election_1(average_percents: np.ndarray, nb_districts = 100, precision = 100):
    """ Randomly draw election results in nb_districts districts"""
    """ For the moment for simplicity's sake all districts have the same size"""

    return np.random.dirichlet(average_percents * precision, size = nb_districts) 
    
def simulate_election_2_given_election_1(election1, transfer_matrix: np.ndarray, precision = 500):
    return np.array([simulate_voting_district(el1, transfer_matrix, precision=precision) for el1 in election1])

# first number: abstention. second round between first two candidates in first round (and abstention!)
means1 = np.array([0.4, 0.3, 0.2, 0.1])
mat = np.array([[0.8, 0.03, 0.01, 0.5],[0.15, 0.95, 0.01, 0.3],[0.05, 0.02, 0.98, 0.2]])
mat2 = np.array([[0.2, 0.53, 0.01, 0.5],[0.75, 0.45, 0.01, 0.3],[0.05, 0.02, 0.98, 0.2]])

ele_1 = simulate_election_1(means1, nb_districts=100, precision=50)
ele_2 = simulate_election_2_given_election_1(ele_1, mat, precision=100)

"""
print(ele_1[7])
plt.hist(ele_1[:][0], bins=30, edgecolor='black')
plt.title("Histogram of the first component ele1")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

plt.hist(ele_2[:][0], bins=30, edgecolor='black')
plt.title("Histogram of the first component ele2")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
"""

print(mat)
print(ele_1)
print(ele_2)
# test particle swarm and genetic
# print(PSO_optimization(ele_1, ele_2, nb_particles = 64))
# print(genetic_optimization(ele_1, ele_2, nb_particles = 16, nb_iterations=1000, sig = 0.8))


# none of both methods works...

# TO DO: try with a neural network using pytorch 
# dando percentuali al 100% per un candidato all'elezione 1 dovrebbe dare i trasferimenti di voti

# 1. Data Setup (Example: 100 samples of 3x2 inputs and 2x2 outputs)
# Replace these with your actual lists of arrays
list_a = [torch.randn(3, 2) for _ in range(100)]
list_b = [torch.randn(2, 2) for _ in range(100)]

# Convert lists to single tensors: (Batch, Height, Width)
X = torch.stack(list_a)
Y = torch.stack(list_b)

# Flatten the dimensions for a standard Linear layer
# X_flat becomes (100, 6) | Y_flat becomes (100, 4)
X_flat = X.view(X.size(0), -1)
Y_flat = Y.view(Y.size(0), -1)

# 2. Define Model
model = nn.Sequential(
    nn.Linear(X_flat.size(1), 32),
    nn.ReLU(),
    nn.Linear(32, Y_flat.size(1))
)

# 3. Training setup
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 4. Simple Training Loop
for epoch in range(200):
    optimizer.zero_grad()
    prediction = model(X_flat)
    loss = criterion(prediction, Y_flat)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 5. Usage: Predict and reshape back to original target dimensions
test_input = X_flat[0]
predicted_flat = model(test_input)
predicted_array = predicted_flat.view(2, 2) # Reshape back to target dimensions
print("\nPredicted Array Shape:", predicted_array.shape)