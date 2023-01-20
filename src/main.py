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

# either "synthetic" or "real"
DATASET_TYPE = "real"

################################################################################

# Creating and Configuring Logger
Log_Format = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(stream = sys.stdout, filemode = "w", format = Log_Format, level = logging.INFO)
logger = logging.getLogger()

#### Import the datasets
logger.info("Importing Datasets")

utility_df = pd.read_csv("data/" +  DATASET_TYPE +"/utility_matrix.csv", index_col=0)
utility_df.fillna(0, inplace=True)

user_set = pd.read_csv("data/" +  DATASET_TYPE +"/user_set.csv", header=None)
query_set = pd.read_csv("data/" +  DATASET_TYPE +"/query_set.csv", index_col=0, header=None)
relational_table = pd.read_csv("data/" +  DATASET_TYPE +"/relational_table.csv")
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
  num_queries_c = np.arange(start=200, stop=2200, step=200) # test cosine similarity and simhash with more queries
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
  if hybrid: # run also content based
    logger.info("Running content based")
    predicted_utility = rec.hybridRecommendation(utility, relational_table, query_set, k_most_similar)
  else: # run only collaborative filtering taking the average of the k-most similar queries
    predicted_utility = rec.predictAsAverage(utility, k_most_similar)


  ### dump the predicted utility matrix to file
  logger.info("Saving the predicted utility matrix as file predicted_utility.csv")
  predicted_utility_df = pd.DataFrame(predicted_utility)
  predicted_utility_df.columns = utility_df.keys()
  predicted_utility_df.index = utility_df.index
  predicted_utility_df = predicted_utility_df.convert_dtypes()
  predicted_utility_df.to_csv("data/" +  DATASET_TYPE + "/predicted_utility.csv")

################################################################################


'''
This function evaluates the results found by the recommendation system 
'''
def evaluatePredictions(num_folds):
  
  logger.info("Starting with the evaluation of the results")
  errors, fold_rmse, fold_mae = ev.kfold_cross_validation(num_folds, original_utility, relational_table, query_set)

  avg_error_only_lsh_minhash, avg_error_only_lsh_simhash, avg_error_lsh_content, avg_error_random = errors["rmse"]
  print("RMSE = %f when using LSH + CF(MinHash)" % avg_error_only_lsh_minhash)
  print("RMSE = %f when using LSH + CF(SimHash)" % avg_error_only_lsh_simhash)
  print("RMSE = %f when using LSH + CF + Content based(Hybrid rec. system)" % avg_error_lsh_content)
  print("RMSE = %f when using random ratings" % avg_error_random)

  avg_error_only_lsh_minhash, avg_error_only_lsh_simhash, avg_error_lsh_content, avg_error_random = errors["mae"]
  print("MAE = %f when using LSH + CF(MinHash)" % avg_error_only_lsh_minhash)
  print("MAE = %f when using LSH + CF(SimHash)" % avg_error_only_lsh_simhash)
  print("MAE = %f when using LSH + CF + Content based(Hybrid rec. system)" % avg_error_lsh_content)
  print("MAE = %f when using random ratings" % avg_error_random)

  return fold_rmse, fold_mae

################################################################################

if __name__=='__main__':

  command = input("\n[1] Fill the blanks of the utility matrix \n[2] Compare the time performance of CF + LSH wrt CF (without LSH)\n[3] Evaluate the accuracy(RMSE and MAE) of the following algorithms\n\t- Collaborative filtering with LSH(LSH + CF)\n\t- Hybrid recommendation system with LSH(LSH + CF + content based)\n\t- random ratings prediction\n\n[4] Measure time performance and error rate by increasing the signature matrix size (CF + LSH)\n[5] Measure time performance and error rate by increasing the number of rows per band of LSH (CF + LSH)\n\nSelect one option: ")  
  if command == '1':
    
    method = input("\nSelect the algorithm to use for filling the blanks of the utility matrix: \n\t[1] Collaborative filtering with LSH-MinHash(LSH + CF) \n\t[2] Collaborative filtering with LSH-SimHash(LSH + CF) \n\t[3] Hybrid recommendation system(LSH + CF + Content based)\n\nSelect one option: ")
    if method == '1':
      predictUtilityMatrix(LSH_method="minhash", hybrid=False)
    elif method == '2':
      predictUtilityMatrix(LSH_method="simhash", hybrid=False)
    elif method == '3':
      predictUtilityMatrix(LSH_method="simhash", hybrid=True)
    else:
      logger.error("Wrong Input, try again!")

  elif command == '2':
    time_complexity_vals = plotTimeComplexityCurve()
  elif command == '3':
    fold_rmse, fold_mae = evaluatePredictions(num_folds=10)
  elif command=='4':
    rows, error_rate, correctly_estimated, time_to_run, avg_candidates = ev.plot_rows_curve(utility, "cosine")
  elif command=='5':
    rows_per_band, error_rate, correctly_estimated, time_to_run, avg_candidates = ev.plot_bands_curve(utility, "cosine")
  else:
    logger.error("Wrong Input, try again!")

