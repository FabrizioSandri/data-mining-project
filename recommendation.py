import similarity_measures as sim
import numpy as np
import pandas as pd


########################### COLLABORATIVE FILTERING ############################

'''
This function returns the K most similar queries for each of the queries in the 
utility matrix.

Arguments:
  K: the number of most similar queries to find for each query
  utility: the utility matrix
  similar_items: a dictionary containing the similar items found by LSH(the 
    output of the LSH procedure)
  similarity: the similarity measure to use for comparing the queries, either 
    "jaccard" or "cosine" 
  
Returns:
  A dictionary with the K most similar queries
'''
def get_k_most_similar_queries_utility(K, utility, similar_items, similarity="cosine"):
  k_most_similar = {}
  for query in range(utility.shape[1]):
    prefs = []
    for i in similar_items[query]:
      if similarity == "jaccard":
        prefs.append(sim.jaccard_threshold(utility[:,query], utility[:,i], T=50))
      else:
        prefs.append(sim.cosine_similarity(utility[:,query], utility[:,i]))
    
    k_most_similar[query] = []
    if len(similar_items[query]) > 0: # at least one similar query found
      most_similar_query_i = np.argsort(prefs)[-min(K,len(prefs)-1):]
      k_most_similar[query] =  [list(similar_items[query])[i] for i in most_similar_query_i]

  return k_most_similar


############################## CONTENT BASED ###################################

'''
This function simulates a DBMS by returning the ids of the rows of the 
relational table that satisfy the conditions specified by the query given as 
input. Note that the input query is a query id

Arguments:
  query: the id of the query without the "Q" prefix
  relational_table: the pandas dataframe containing the relational table
  query_set: the pandas dataframe containing the description of the queries sent
    by the users

Returns:
  A list containing the row ids that the query has returned. If the query 
  doesn't return anything the result is an empty list: no row has been returned.
'''
def getRowsIds(query, relational_table, query_set):

  result_rows = np.full((relational_table.shape[0]), True)

  query_full_row = query_set.loc["Q"+str(query)]
  query_conditions = query_full_row[~pd.isna(query_full_row)].tolist() # remove NAN conditions
  for condition in query_conditions:
    splitted = condition.split("=")
    cond_var = splitted[0]
    cond_val = splitted[1]

    if pd.api.types.is_numeric_dtype(relational_table[cond_var]):
      result_rows = result_rows & (relational_table[cond_var] == float(cond_val)).to_numpy()
    else:
      result_rows = result_rows & (relational_table[cond_var] == cond_val).to_numpy()

  return np.where(result_rows != False)
  


'''
This function find the top-T similar queries by looking at their content. The 
idea is that two queries A and B should be considered more than A with C if 
A and B have more tuples in common than the tuples in common between A and C

Arguments:
  T: the number of most similar queries to find for each query(this number must
    be smaller than K)
  k_most_similar: the K most similar queries for each query in the utility 
    matrix returned by the function the get_k_most_similar_queries_utility that 
    returns the similar queries according to collaborative filtering + LSH
  relational_table: the pandas dataframe containing the relational table
  query_set: the pandas dataframe containing the description of the queries sent
    by the users
  
Returns:
  A dictionary of the top T most similar queries for each query in the utility
  matrix
'''
def get_t_most_similar_queries_content(T, k_most_similar, relational_table, query_set):
  t_most_similar = {}
  
  # find the rows returned by each query to avoid recomputing them several times
  row_ids = {}
  for q in range(len(k_most_similar)):
    row_ids[q] = getRowsIds(q, relational_table, query_set)

  # iterator on the K most similar queries to extract the T<K most similar 
  # queries based on their content
  for query in range(len(k_most_similar)):
    queries_in_common = []
    rows_of_query = row_ids[query] # get the ids of the rows returned by query

    for i in k_most_similar[query]: 
      rows_of_query_i = row_ids[i]
      queries_in_common.append(len(np.intersect1d(rows_of_query, rows_of_query_i)))

    t_most_similar[query] = []
    if len(k_most_similar[query]) > 0:
      most_similar_query_i = np.argsort(queries_in_common)[-min(T,len(queries_in_common)-1):]
      t_most_similar[query] =  [list(k_most_similar[query])[i] for i in most_similar_query_i]

  return t_most_similar


########################## HYBRID RECOMMENDATION ###############################

'''
This function returns a utility matrix filled with the missing ratings by taking
the average of the most similar queries that a user has rated, according to 

Arguments:
  utility: the utility matrix
  similar_items: a dictionary containing the similar items found by LSH 
    (for collaborative filtering with LSH only) or the similar items found by 
    the combination of collaborative filtering + content based(hybrid)
  
Returns:
  The utility matrix with the filled missing ratings 
'''
def predictAsAverage(utility, most_similar):
  predicted_utility = utility.copy()

  # now the recommendation system computes the missing values as the average of 
  # the K most similar queries 
  for query in range(predicted_utility.shape[1]):
    print("Predicting missing ratings for query %d" % query)
    for user in range(predicted_utility.shape[0]):
      if predicted_utility[user, query] == 0:
        if len(most_similar[query]) > 0:
          similar_ratings = [predicted_utility[user,j] for j in most_similar[query]]
          predicted_utility[user, query] = round(np.mean(similar_ratings))
        else:
          predicted_utility[user, query] = np.random.randint(1,101) # if no similar query is found predict with a random value

  return predicted_utility
