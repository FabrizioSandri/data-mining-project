import lsh
import similarity_measures as sim
import evaluate as ev

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

#### VARIABLES
# utility_df = pd.read_csv("tests/Dataset/dataFolder/utilityMatrixDataset.csv")
utility_df = pd.read_csv("utility.csv", header=None, index_col=False)
utility = utility_df.to_numpy()

# fill some random values of the utility with 0. based on the sparsity amount
utility_sparsity = 0.3
mask = np.random.choice([True, False], utility.shape , p=[utility_sparsity,1-utility_sparsity])
utility[mask] = 0

print("============ Utility matrix:")
print(utility)

### LSH with minHash
# signature_matrix = lsh.minHash(utility_matrix=utility, k=200, T=50)
# similar_items = lsh.lsh(signature_matrix, bands=50)


### LSH with simHash
# signature_matrix = lsh.simHash(utility_matrix=utility, hyperplanes=400)
# similar_items = lsh.lsh(signature_matrix, rows_per_band=15)

# correctly_estimated, time_to_run, avg_candidates = ev.plot_combined_curve(utility, "cosine")

# i = range(len(correctly_estimated))
# plt.plot(i,avg_candidates, label="Avg amount of similar candidates")
# plt.plot(i,correctly_estimated, label="Number of correctly estimated similarities")
# plt.plot(i,time_to_run, label="Time to find the candidate similar items")

# plt.xlabel("Number of hyperplanes for SimHash")
# plt.legend()
# plt.grid()
# plt.show()


num_queries = np.arange(start=100, stop=1100, step=100)
num_users = np.arange(start=100, stop=600, step=100)

time_jaccard_q, time_cosine_q, time_minhash_q, time_simhash_q = ev.measure_time_increase(utility, num_queries=num_queries)
time_jaccard_u, time_cosine_u, time_minhash_u, time_simhash_u = ev.measure_time_increase(utility, num_users=num_users)


fig,ax = plt.subplots(2,2)
ax = ax.flatten()

ax[0].plot(num_queries, time_jaccard_q, label="Baseline solution Jaccard")
ax[0].plot(num_queries, time_minhash_q, label="LSH solution Minhash")
ax[0].set_xlabel("Number of queries (utility matrix columns)")
ax[0].set_ylabel("Time (seconds)")
ax[0].legend()
ax[0].grid()

ax[2].plot(num_queries, time_cosine_q, label="Baseline solution Cosine")
ax[2].plot(num_queries, time_simhash_q, label="LSH solution Simhash")
ax[2].set_xlabel("Number of queries (utility matrix columns)")
ax[2].set_ylabel("Time (seconds)")
ax[2].legend()
ax[2].grid()

ax[1].plot(num_users, time_jaccard_u, label="Baseline solution Jaccard")
ax[1].plot(num_users, time_minhash_u, label="LSH solution Minhash")
ax[1].set_xlabel("Number of users (utility matrix rows)")
ax[1].set_ylabel("Time (seconds)")
ax[1].legend()
ax[1].grid()

ax[3].plot(num_users, time_cosine_u, label="Baseline solution Cosine")
ax[3].plot(num_users, time_simhash_u, label="LSH solution Simhash")
ax[3].set_xlabel("Number of users (utility matrix rows)")
ax[3].set_ylabel("Time (seconds)")
ax[3].legend()
ax[3].grid()

plt.show()