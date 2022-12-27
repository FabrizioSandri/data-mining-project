import lsh
import similarity_measures as sim
import evaluate as ev
import recommendation as rec

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


import logging
import sys

#Creating and Configuring Logger

Log_Format = "%(levelname)s %(asctime)s - %(message)s"

logging.basicConfig(stream = sys.stdout, 
                    filemode = "w",
                    format = Log_Format, 
                    level = logging.INFO)

logger = logging.getLogger()

#### VARIABLES

logger.info("Importing Datasets")

utility_df = pd.read_csv("Dataset/dataFolder/UtilityDataset_Synthetic.csv", index_col=0)
utility_df.fillna(0, inplace=True)

query_set = pd.read_csv("Dataset/dataFolder/QueriesDataset_Syntethic.csv", index_col=0, header=None)
relational_table = pd.read_csv("Dataset/dataFolder/RelationaTable_make_blobs.csv")
relational_table = relational_table.convert_dtypes()


utility = utility_df.to_numpy()
original_utility = utility.copy()

logger.info("Imports DONE")

################################################################################
# The following two snippets of code runs LSH with minHash and LSH with simHash
# on the utility matrix to extract the candidate similar queries of each query.

### LSH with minHash
# signature_matrix = lsh.minHash(utility_matrix=utility, k=400, T=50)
# similar_items = lsh.lsh(signature_matrix, rows_per_band=7)

### LSH with simHash
logger.info("Building signature matrix")
signature_matrix = lsh.simHash(utility_matrix=utility, hyperplanes=400)
logger.info("Finding Similar items")
similar_items = lsh.lsh(signature_matrix, rows_per_band=15)

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

# logger.info("Start estimation and plots")
# correctly_estimated, time_to_run, avg_candidates = ev.plot_combined_curve(utility, "cosine")

# i = range(len(correctly_estimated))
# plt.plot(i,avg_candidates, label="Avg amount of similar candidates")
# plt.plot(i,correctly_estimated, label="Number of correctly estimated similarities")
# plt.plot(i,time_to_run, label="Time to find the candidate similar items")

# plt.xlabel("Number of hyperplanes for SimHash")
# plt.legend()
# plt.grid()
# plt.show()
# logger.info("Done with estimation")
################################################################################
# This part plots the time complexity of the four algorithms(jaccard similarity
# without LSH, cosine similarity without LSH, LSH with minHash and LSH with 
# simHash). 

# logger.info("Start plotting algorithms time complexity")
# num_queries_j = np.arange(start=100, stop=1100, step=100)
# num_queries_c = np.arange(start=300, stop=3000, step=200) # test cosine similarity and simhash with more queries
# num_users = np.arange(start=100, stop=600, step=100)

# ev.plot_time_increase(utility, num_users, num_users, num_queries_j, num_queries_c)

# logger.info("Done with time complexity evaluation")

################################################################################
# In this part the recommendation system actually operates and the utility 
# matrix is filled with the missing values, taking the value from the K most 
# similar query for a user 


# find the K-most similar queries for each query
K = 20 

logger.info("Retrieving k most similar")
k_most_similar = rec.get_k_most_similar_queries_utility(K, utility, similar_items, "cosine")


logger.info("Retrivieng utility prediction")
predicted_utility_k = rec.predictAsAverage(utility, k_most_similar)
predicted_utility_content = rec.hybridRecommendation(utility, relational_table, query_set, k_most_similar)


# dump the predicted utility matrix to file
# dump the predicted utility matrix to file
predicted_utility_df = pd.DataFrame(predicted_utility_k)
predicted_utility_df.columns = utility_df.keys()
predicted_utility_df.index = utility_df.index
predicted_utility_df.to_csv("predicted_LSH_only.csv")

predicted_utility_df = pd.DataFrame(predicted_utility_content)
predicted_utility_df.columns = utility_df.keys()
predicted_utility_df.index = utility_df.index
predicted_utility_df.to_csv("predicted_LSH_content.csv")


################################################################################
# In this part we evaluate the results found by the recommendation system 
logger.info("Starting with the evaluation")

avg_error_only_lsh, avg_error_lsh_content, avg_error_random = ev.evaluate_prediction(original_utility, 0.01, "cosine", relational_table, query_set)

print("The predicted ratings using only LSH differ from the real ones by an average of %f" % avg_error_only_lsh)
print("The predicted ratings using LSH + content based differ from the real ones by an average of %f" % avg_error_lsh_content)
print("The predicted ratings differ from the randomly generated predictions by an average of %f" % avg_error_random)
