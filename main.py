import lsh
import similarity_measures as sim
import evaluate as ev

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

#### VARIABLES
utility_df = pd.read_csv("tests/Dataset/dataFolder/UtilityDataset.csv")
# utility_df = pd.read_csv("utility.csv", header=None, index_col=False)

#tranform utility dataset from pandas dataframe to numpy matrix, 
#user iDs strings tranformed in their integer part
utility_df = utility_df.replace('usr', '', regex=True)
utility_df['Usr'] = utility_df['Usr'].astype(int)
original_utility = utility_df.to_numpy()

# fill some random values of the utility with 0. based on the sparsity amount
utility_sparsity = 0.3
mask = np.random.choice([True, False], original_utility.shape , p=[utility_sparsity,1-utility_sparsity])
utility = original_utility.copy()
utility[mask] = 0


################################################################################
# The following two snippets of code runs LSH with minHash and LSH with simHash.
# They are there just as a reference to be used for testing.

### LSH with minHash
signature_matrix = lsh.minHash(utility_matrix=utility, k=400, T=50)
similar_items = lsh.lsh(signature_matrix, rows_per_band=10)


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

# num_queries_j = np.arange(start=100, stop=1100, step=100)
# num_queries_c = np.arange(start=300, stop=3000, step=200) # test cosine similarity and simhash with more queries
# num_users = np.arange(start=100, stop=600, step=100)

# ev.plot_time_increase(utility, num_users, num_users, num_queries_j, num_queries_c)


################################################################################
# In this part the recommendation system actually operates and the utility 
# matrix is filled with the missing values, taking the value from the most 
# similar query for a user 
similarity = "cosine"

for query in range(utility.shape[1]):
  print("Predicting missing ratings for query %d" % query)
  for user in range(utility.shape[0]):
    if utility[user, query] == 0:

      prefs = []
      for i in similar_items[query]:
        if similarity == "jaccard":
          prefs.append(sim.jaccard_threshold(utility[:,query], utility[:,i], T=50))
        else:
          prefs.append(sim.cosine_similarity(utility[:,query], utility[:,i]))
      
      most_similar_query = None
      if len(similar_items[query]) > 0: # at least one similar query found
        most_similar_query_i = np.argmax(prefs)
        most_similar_query = list(similar_items[query])[most_similar_query_i]

      if most_similar_query:
        utility[user, query] = utility[user, most_similar_query]


# dump the predicted utility matrix to file
pd.DataFrame(utility).to_csv("predicted.csv", header=False, index=False)

# measure the average difference between the predicted ratings and the originals
avg = ev.evaluate_prediction(original_utility, utility, mask)
print("The predicted ratings differ from the real ones by an average of %f" % np.mean(avg))


# compare the results with random assignment of the ratings
fake_utility = np.random.randint(1,101, utility.shape) # utility with random values
avg_fake = ev.evaluate_prediction(fake_utility, utility, mask)
print("The predicted ratings differ from the randomly generated predictions by an average of %f" % np.mean(avg_fake))
