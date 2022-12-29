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

################################################################################

#Creating and Configuring Logger
Log_Format = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(stream = sys.stdout, filemode = "w", format = Log_Format, level = logging.INFO)
logger = logging.getLogger()

#### Import the datasets
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


'''
This function plots the time complexity of the four algorithms(jaccard
similarity without LSH, cosine similarity without LSH, LSH with minHash and LSH
with simHash). 
'''
def plotTimeComplexityCurve():
  logger.info("Start plotting LSH algorithms time complexity")
  num_queries_j = np.arange(start=100, stop=1100, step=100)
  num_queries_c = np.arange(start=200, stop=2000, step=100) # test cosine similarity and simhash with more queries
  num_users = np.arange(start=100, stop=600, step=100)

  return ev.plot_time_increase(utility, num_users, num_users, num_queries_j, num_queries_c, logger)

################################################################################

'''
This function fills the blanks of the utility matrix using the LSH method
specified and using content_based combined with collaborative filtering(hybrid
recommendation) based on the hybrid parameter.

Arguments:
  LSH_method: the LSH technique to user, either 'minhash' or 'simhash'
  hybrid: run content based on top of collaborative filtering
'''
def predictUtilityMatrix(LSH_method="simhash", hybrid=True):
  
  ### Run LSH
  logger.info("Running LSH")
  if LSH_method=="minhash":
    # LSH with minHash
    signature_matrix = lsh.minHash(utility_matrix=utility, k=400, T=50)
    similar_items = lsh.lsh(signature_matrix, rows_per_band=7)
  else:
    # LSH with simHash
    signature_matrix = lsh.simHash(utility_matrix=utility, hyperplanes=400)
    similar_items = lsh.lsh(signature_matrix, rows_per_band=15)


  ### Run Collaborative filtering
  logger.info("Running collaborative filtering")

  K = 20 
  k_most_similar = rec.get_k_most_similar_queries_utility(K, utility, similar_items, "cosine")

  ### Run Hybrid recommendation
  if hybrid: # run alsocontent based
    logger.info("Running content based")
    predicted_utility = rec.hybridRecommendation(utility, relational_table, query_set, k_most_similar)
  else: # run only collaborative filtering taking the average of the k-most similar queries
    predicted_utility = rec.predictAsAverage(utility, k_most_similar)


  ### dump the predicted utility matrix to file
  logger.info("Saving the predicted utility matrix as file predicted_utility.csv")
  predicted_utility_df = pd.DataFrame(predicted_utility)
  predicted_utility_df.columns = utility_df.keys()
  predicted_utility_df.index = utility_df.index
  predicted_utility_df.to_csv("predicted_utility.csv")

################################################################################


'''
This function evaluates the results found by the recommendation system 
'''
def evaluatePredictions(test_size):
  
  logger.info("Starting with the evaluation of the results")
  avg_error_only_lsh, avg_error_lsh_content, avg_error_random = ev.evaluate_prediction(original_utility, test_size, "cosine", relational_table, query_set)

  print("The predicted ratings using only LSH differ from the real ones by an average of %f" % avg_error_only_lsh)
  print("The predicted ratings using LSH + content based differ from the real ones by an average of %f" % avg_error_lsh_content)
  print("The predicted ratings differ from the randomly generated predictions by an average of %f" % avg_error_random)


################################################################################

if __name__=='__main__':

  logger.info("Starting DataSets Generation Phase")
  command = input("Select the operation you want to do:\n[1] Fill the blanks of the utility matrix using the Hybrid recommendation system(LSH + CF + Content based)\n[2] Evaluate the performance of LSH wrt the performance of running the algorithm without LSH\n[3] Compare the RMSE of running the algorithm using all the following methods\n\ta. Collaborative filtering with LSH(LSH + CF)\n\tb. Hybrid recommendation system with LSH(LSH + CF + content based)\n\tc. random ratings prediction\n Choice: ")
  if command == '1':
    predictUtilityMatrix(LSH_method="simhash", hybrid=True)
  elif command == '2':
    time_complexity_vals = plotTimeComplexityCurve()
  elif command=='3':
    evaluatePredictions(test_size=0.01)
  else:
    logger.error("Wrong Input, try again!")


