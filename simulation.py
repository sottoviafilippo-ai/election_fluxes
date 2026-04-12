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
# test particle swarm and genetic
# print(PSO_optimization(ele_1, ele_2, nb_particles = 64))
# print(genetic_optimization(ele_1, ele_2, nb_particles = 16, nb_iterations=1000, sig = 0.8))


# none of both methods works...



# ANOTHER APPROACH: try with a neural network using pytorch 
# dando percentuali al 100% per un candidato all'elezione 1 dovrebbe dare i trasferimenti di voti

# Convert lists to single tensors: (Batch, Height, Width)
X = torch.stack([torch.from_numpy(a) for a in ele_1]).float()
Y = torch.stack([torch.from_numpy(a) for a in ele_2]).float()

# Flatten the dimensions for a standard Linear layer
X_flat = X.view(X.size(0), -1)
Y_flat = Y.view(Y.size(0), -1)

# Define Model
model = nn.Sequential(
    nn.Linear(X_flat.size(1), 32),
    nn.ReLU(),
    nn.Linear(32, Y_flat.size(1))
)

# Training setup
# Adam: adaptive moment estimation. lr: initial step size
optimizer = optim.Adam(model.parameters(), lr=0.001)
# the model enters the optimizer by model.parameters()
criterion = nn.MSELoss() # use a mean squared error loss function
loss_history = []
# Simple Training Loop
for epoch in range(100):
    optimizer.zero_grad()
    prediction = model(X_flat)
    loss = criterion(prediction, Y_flat.float())
    loss.backward() # backpropagation
    optimizer.step() # parameter update, new_param = old_param - eta*gradient 
    # note that this works because prediction is a node in a dynamic computational graph
    # prediction thus carries a reference to grad_fn, which is passed over to loss via the criterion
    loss_history.append(loss.item())

# Now test what would happen with 100% for a condidate at a time
test_input = torch.tensor([[1,0,0,0]]).float()
predicted_flat = model(test_input)
predicted_array = predicted_flat.view(-1) # Reshape back to target dimensions
print(predicted_array.detach().numpy())

test_input = torch.tensor([[0,1,0,0]]).float()
predicted_flat = model(test_input)
predicted_array = predicted_flat.view(-1) # Reshape back to target dimensions
print(predicted_array.detach().numpy())

test_input = torch.tensor([[0,0,1,0]]).float()
predicted_flat = model(test_input)
predicted_array = predicted_flat.view(-1) # Reshape back to target dimensions
print(predicted_array.detach().numpy())

test_input = torch.tensor([[0,0,0,1]]).float()
predicted_flat = model(test_input)
predicted_array = predicted_flat.view(-1) # Reshape back to target dimensions
print(predicted_array.detach().numpy())

# does not work, even gives negatives results


"""plt.figure()
plt.plot(loss_history)
plt.title("Loss history (adam)")
plt.show()"""

# Gemini suggestion : have you considered a Softmax output layer with KLDivLoss (Kullback–Leibler divergence) instead of MSE

# TO DO: EXPLORE KL

# SEARCH FOR THE CLASSICAL ALGOS TO SOLVE THIS PROBLEM... 