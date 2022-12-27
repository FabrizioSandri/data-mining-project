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
    for user in range(predicted_utility.shape[0]):
      if predicted_utility[user, query] == 0:
        if len(most_similar[query]) > 0:
          similar_ratings = [predicted_utility[user,j] for j in most_similar[query]]
          predicted_utility[user, query] = round(np.mean(similar_ratings))
        else:
          predicted_utility[user, query] = np.random.randint(1,101) # if no similar query is found predict with a random value

  return predicted_utility

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

  row_ids = np.where(result_rows != False)
  return row_ids[0]
  

'''
This function generates the item profile for each query, which is a vector with
the size of the number of rows of the relational table, filled with 1 in 
correspondence with the rows that were returned by the query specified

Arguments:
  query: the id of the query without the "Q" prefix
  relational_table: the pandas dataframe containing the relational table
  query_set: the pandas dataframe containing the description of the queries sent
    by the users

Returns:
  An item profile for the query given as input
'''
def getItemProfile(query, relational_table, query_set):
  num_rows = relational_table.shape[0]
  item_profile = np.asarray([0] * num_rows)

  returned_row_ids = getRowsIds(query, relational_table, query_set)
  item_profile[returned_row_ids] = 1

  return item_profile


'''
This fun
'''
def getUserProfile(userId, utility, relational_table, query_set, queries_returning_row):
  num_rows = relational_table.shape[0]
  num_queries = query_set.shape[0]
  user_profile = np.asarray([0] * num_rows)

  userRating = []
  for rating in utility[userId]:
    if rating != 0 :
      userRating.append(rating)

  avgUserRating = int(np.round(np.mean(userRating)))

  for row_i in range(num_rows):
    row_ratings = [] # ratings of the specified user to the row `row_i`
    queries_returning_row_i = queries_returning_row[row_i]

    for query in queries_returning_row_i:
      if utility[userId,query] != 0: # the user has rated the query
        row_ratings.append(utility[userId,query])

    row_ratings = np.asarray(row_ratings) - avgUserRating # subtract the average rating of the user

    if len(row_ratings) == 0:
      avg_row_rating = 0
    else:
      avg_row_rating = np.mean(row_ratings)

    user_profile[row_i] = avg_row_rating

  return user_profile

'''
This function returns a dictionary where each key of the dictionary is a row of 
the relational table and the value is a list of queries that returned that row

Arguments:
  relational_table: the pandas dataframe containing the relational table
  query_set: the pandas dataframe containing the description of the queries sent
    by the users

Returns:
  A dictionary as described above
'''
def getQueriesReturningRow(relational_table, query_set):
  num_rows = relational_table.shape[0]
  num_queries = query_set.shape[0]
  
  row_queries = {} # dictionary returning for each row of the relational table the queries that returned it

  for row in range(num_rows):
    row_queries[row] = []

  # find the rows returned by each query to avoid recomputing them several times

  for q in range(num_queries):
    row_ids = getRowsIds(q, relational_table, query_set)
    for row in row_ids:
      row_queries[row].append(q)

  return row_queries


########################## HYBRID RECOMMENDATION ###############################



def hybridRecommendation(utility, relational_table, query_set, most_similar):
  predicted_utility = utility.copy()
  num_queries = query_set.shape[0]
  num_users = predicted_utility.shape[0]

  item_profile = {}
  user_profile = {}

  print("Computing the item profiles(query profiles)")
  for query in range(num_queries):
    item_profile[query] = getItemProfile(query, relational_table, query_set)

  print("Computing the user profiles")
  queries_returning_row = getQueriesReturningRow(relational_table, query_set)
  for user in range(num_users):
    print(user)
    user_profile[user] = getUserProfile(user, utility, relational_table, query_set, queries_returning_row)

  print("Finished to compute the profiles")

  for user in range(num_users):  # iterate over all the cell of the utility matrix that are empty
    print("Predicting missing ratings for user%d" % (user+1))
    for query in range(num_queries):    
      if predicted_utility[user, query] == 0:
        
        most_similar_query_sim = []
        for similar_query in most_similar[query]:
          most_similar_query_sim.append(sim.cosine_similarity(user_profile[user], item_profile[similar_query]))
        
        # find the most similar query that should be recommended to the user
        # using content based recommendation combined with collaborative 
        # filtering. The idea is to compare the user profile with the item 
        # profile only with the potentially similar query found by collaborative 
        # filtering with LSH, avoiding in this way to compare all the item 
        # profiles with all the user profiles
        most_similar_query_i = np.argsort(most_similar_query_sim)
        K = len(most_similar[query])
        for ki in range(K):
          most_similar_i = K - 1 - ki
          most_similar_query = list(most_similar[query])[most_similar_query_i[most_similar_i]]
          if utility[user, most_similar_query] != 0:
            predicted_utility[user, query] = utility[user, most_similar_query]
            break

        if predicted_utility[user, query] == 0:
          predicted_utility[user, query] = np.random.randint(1,101) # if no similar query is found predict with a random value
          
  return predicted_utility

