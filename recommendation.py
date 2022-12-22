import similarity_measures as sim
import numpy as np

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
def get_k_most_similar_queries(K, utility, similar_items, similarity="cosine"):
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
the average of the K most similar queries that a user has rated

Arguments:
  utility: the utility matrix
  similar_items: a dictionary containing the similar items found by LSH(the 
    output of the LSH procedure)
  K: the number of most similar queries to find for each query
  similarity: the similarity measure to use for comparing the queries, either 
    "jaccard" or "cosine" 
  
Returns:
  The utility matrix with the filled missing ratings 
'''
def predictAsTopKAverage(utility, similar_items, K, similarity):
  predicted_utility = utility.copy()

  # find the K-most similar queries for each query
  k_most_similar = get_k_most_similar_queries(K, utility, similar_items, similarity)


  # now the recommendation system computes the missing values as the average of 
  # the K most similar queries 
  for query in range(predicted_utility.shape[1]):
    print("Predicting missing ratings for query %d" % query)
    for user in range(predicted_utility.shape[0]):
      if predicted_utility[user, query] == 0 and len(k_most_similar[query]) > 0:
        similar_ratings = [predicted_utility[user,j] for j in k_most_similar[query]]
        predicted_utility[user, query] = round(np.mean(similar_ratings))

  return predicted_utility