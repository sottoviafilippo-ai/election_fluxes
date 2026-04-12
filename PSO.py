import numpy as np
from numpy.linalg import norm

""" Particle Swarm Algortihm applied to the analysis of electoral fluxes"""

def election_flux_rest(election1: np.ndarray, election2: np.ndarray, transfer_matrix: np.ndarray):
    """ election1 and election2 are the data for 2 different elections in the same individual unit (province, municipality, etc)"""

    return norm(election2 - np.dot(transfer_matrix, election1))

def election_flux_rest_all_districts(ele1: np.ndarray, ele2: np.ndarray, transfer_matrix: np.ndarray):
    """ election1 and election2 are lists of election data in all considered districts"""

    return np.sum([election_flux_rest(ele1[k], ele2[k], transfer_matrix) for k in range(len(ele1))])

def generate_random_transfer_matrix(n: int, m: int) -> np.ndarray:
    """
    Generate an n x m matrix where:
    - All entries are between 0 and 1
    - The sum of each column equals 1
    """
    # Draw random positive values, then normalize each column
    matrix = np.random.uniform(0, 1, size=(n, m))
    col_sums = matrix.sum(axis=0)          # shape (m,)
    matrix = matrix / col_sums[np.newaxis, :]  # broadcast-divide each column
    return matrix

def PSO_optimization(ele1, ele2, nb_particles = 16, nb_iterations = 5, w = 0.5, psi_p = 0.25, psi_g = 0.25):
    """ particle swarm optimization for families of nb_particles indiividal solutions"""

    # first initialize the solutions by random drawing
    current_family = [generate_random_transfer_matrix(len(ele2[0]),len(ele1[0])) for k in range(nb_particles)]
    current_rests  = np.array([election_flux_rest_all_districts(ele1, ele2, matt) for matt in current_family])
    best_rests = current_rests
    individual_best_positions = current_family
    overall_best = current_family[np.argmin(current_rests)]
    overall_best_rest = best_rests[np.argmin(current_rests)]
    # initialize the velocities
    velocities = [np.random.uniform(-1.0, 1.0, size=(len(ele2[0]), len(ele1[0]))) for k in range(nb_particles)]

    for k in range(nb_iterations):
        if k>0 and k%10==0:
            print(k)

        for j in range(nb_particles):

            # update the velocities
            velocities[j] = w * velocities[j] 
            velocities[j] = velocities[j] + psi_p * np.random.uniform(0, 1, size=(len(ele2[0]),len(ele1[0]))) * (individual_best_positions[j] - current_family[j])
            velocities[j] = velocities[j] + psi_g * np.random.uniform(0, 1, size=(len(ele2[0]),len(ele1[0]))) * (overall_best - current_family[j])

            # print(norm(velocities[j]))
            # update the particle. use np.clip to avoid negztive entries (or too large)
            current_family[j] = np.clip(current_family[j] + velocities[j], 0, 2)
            sumcol = current_family[j].sum(axis=0)
            current_family[j] = current_family[j] / sumcol[np.newaxis, :] # poor man's way of implementing unitarity

            current_rests[j]  = election_flux_rest_all_districts(ele1, ele2, current_family[j])

            if current_rests[j] < best_rests[j]:
                individual_best_positions[j] = current_family[j]
                best_rests[j] = current_rests[j]
                print(k, current_rests[j])
                if current_rests[j] < best_rests[j]:
                    overall_best = current_family[j]
                    overall_best_rest = best_rests[j]
                    print(k, overall_best_rest)
    print(overall_best_rest)

    return overall_best

#eventual issue: high dimensionality of transfer matrices. to be checked
# non converge, non si aggiornano mai le singole soluzioni. anche eliminando clip e la n rinormalizzazione. capire


def genetic_optimization(ele1, ele2, nb_particles = 24, nb_iterations = 500, sig = 0.2):
    """ genetic optimization for families of nb_particles indiividal solutions"""

    # first initialize the solutions by random drawing
    current_family = np.array([generate_random_transfer_matrix(len(ele2[0]),len(ele1[0])) for k in range(nb_particles)])
    
  
    for k in range(nb_iterations):
        for j in range(nb_particles):
            # generate a new solution close to the existing one
            newsol = current_family[j] + np.random.normal(loc=0, scale=sig, size=(len(ele2[0]),len(ele1[0])))
            sumcol = newsol.sum(axis=0)
            newsol = newsol / sumcol[np.newaxis, :] # poor man's way of enforcing unitarity
            np.append(current_family, newsol)

        rests = np.array([election_flux_rest_all_districts(ele1, ele2, matt) for matt in current_family])
        # put old and new solutions together and keep the best half of the solution set 
        indices = rests.argsort()
        current_family = current_family[indices]
        current_family = current_family[:nb_particles]


    return current_family