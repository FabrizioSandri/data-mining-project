import lsh
import similarity_measures as sim
import evaluate as ev

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

#### VARIABLES
utility_df = pd.read_csv("tests/Dataset/dataFolder/utilityMatrixDataset.csv")
# utility_df = pd.read_csv("utility.csv", header=None, index_col=False)
utility = utility_df.to_numpy()

# fill some random values of the utility with 0. based on the sparsity amount
utility_sparsity = 0.3
mask = np.random.choice([True, False], utility.shape , p=[utility_sparsity,1-utility_sparsity])
utility[mask] = 0


################################################################################
# The following two snippets of code runs LSH with minHash and LSH with simHash.
# They are there just as a reference to be used for testing.

### LSH with minHash
# signature_matrix = lsh.minHash(utility_matrix=utility, k=200, T=50)
# similar_items = lsh.lsh(signature_matrix, bands=50)


### LSH with simHash
# signature_matrix = lsh.simHash(utility_matrix=utility, hyperplanes=400)
# similar_items = lsh.lsh(signature_matrix, rows_per_band=15)

################################################################################
# In this part we plot the time to run LSH as well as the number of correctly 
# estimated similar queries using LSH(compared to the algorithm run without LSH)
# and we also print the average number of candidate pairs(potential similar
# items) that LSH finds.
#
# As a general rule the average number of candidate pairs should be small in 
# order to find the smallest number of candidate pairs and make the algorithm 
# faster. At the same time the number of correctly estimated similar queries 
# must be the close as possible to the total amount of queries in the dataset:
# meaning that the predicted similar items using LSH are exactly the ones found
# without LSH


# correctly_estimated, time_to_run, avg_candidates = ev.plot_combined_curve(utility, "cosine")

# i = range(len(correctly_estimated))
# plt.plot(i,avg_candidates, label="Avg amount of similar candidates")
# plt.plot(i,correctly_estimated, label="Number of correctly estimated similarities")
# plt.plot(i,time_to_run, label="Time to find the candidate similar items")

# plt.xlabel("Number of hyperplanes for SimHash")
# plt.legend()
# plt.grid()
# plt.show()

################################################################################
# This part plots the time complexity of the four algorithms(jaccard similarity
# without LSH, cosine similarity without LSH, LSH with minHash and LSH with 
# simHash). 
num_queries_j = np.arange(start=100, stop=1100, step=100)
num_users = np.arange(start=100, stop=600, step=100)

time_jaccard_q, _, time_minhash_q, _ = ev.measure_time_increase(utility, num_queries=num_queries_j, algorithms="jaccard")
time_jaccard_u, _, time_minhash_u, _ = ev.measure_time_increase(utility, num_users=num_users, num_queries=[1000], algorithms="jaccard")

num_queries_c = np.arange(start=300, stop=3000, step=200) # test cosine similarity and simhash with more queries
_, time_cosine_q, _, time_simhash_q = ev.measure_time_increase(utility, num_queries=num_queries_c, algorithms="cosine")
_, time_cosine_u, _, time_simhash_u = ev.measure_time_increase(utility, num_users=num_users, algorithms="cosine")

fig,ax = plt.subplots(2,2)
ax = ax.flatten()

ax[0].plot(num_queries_j, time_jaccard_q, label="Baseline solution Jaccard")
ax[0].plot(num_queries_j, time_minhash_q, label="LSH solution Minhash")
ax[0].set_xlabel("Number of queries (utility matrix columns)")
ax[0].set_ylabel("Time (seconds)")
ax[0].legend()
ax[0].grid()
ax[0].title.set_text("LSH for Jaccard - increasing the number of queries")

ax[2].plot(num_queries_c, time_cosine_q, label="Baseline solution Cosine")
ax[2].plot(num_queries_c, time_simhash_q, label="LSH solution Simhash")
ax[2].set_xlabel("Number of queries (utility matrix columns)")
ax[2].set_ylabel("Time (seconds)")
ax[2].legend()
ax[2].grid()
ax[2].title.set_text("LSH for Cosine - increasing the number of queries")

ax[1].plot(num_users, time_jaccard_u, label="Baseline solution Jaccard")
ax[1].plot(num_users, time_minhash_u, label="LSH solution Minhash")
ax[1].set_xlabel("Number of users (utility matrix rows)")
ax[1].set_ylabel("Time (seconds)")
ax[1].legend()
ax[1].grid()
ax[1].title.set_text("LSH for Jaccard - increasing the number of users")

ax[3].plot(num_users, time_cosine_u, label="Baseline solution Cosine")
ax[3].plot(num_users, time_simhash_u, label="LSH solution Simhash")
ax[3].set_xlabel("Number of users (utility matrix rows)")
ax[3].set_ylabel("Time (seconds)")
ax[3].legend()
ax[3].grid()
ax[3].title.set_text("LSH for Cosine - increasing the number of users")

plt.show()