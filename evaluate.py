import numpy as np
import similarity_measures as sim
import lsh
import evaluate as ev
import time
import matplotlib.pyplot as plt

'''
Finds the most similar query to the one specified as input "q", both in the 
original utility matrix by comparing "q" with all the other queries and then by
comparing "q" with the candidate similar queries found by LSH

Arguments:
  q: a vector representing a query in the utility matrix
  similarity: the similarity measure to use, either "jaccard" or "cosine"
  utility: the utility matrix
  similar_items: the list of similar items found using LSH
Returns:
  The query most similar to "q" in the original utility matrix and using LSH
'''
def most_similar_query(q, similarity, utility, similar_items):

  res = []

  ### 1. Find the best among all the queries (inefficient, need to check all the 
  # combinations with query 1: (1,2), (1,3), (1,4), .. , (1,num_queries))
  prefs = []
  for i in range(utility.shape[1]):
    if similarity == "jaccard":
      prefs.append(sim.jaccard_threshold(utility[:,q], utility[:,i], T=50))
    else:
      prefs.append(sim.cosine_similarity(utility[:,q], utility[:,i]))

  prefs[q] = 0
  most_similar_query = np.argmax(prefs)
  most_similar_query_similarity = prefs[most_similar_query]

  res.append(most_similar_query)
  # print("The query most similar to {} in the original utility matrix is query {}, with a similarity of {} ".format(q, most_similar_query, most_similar_query_similarity))

  ### 2. Find the best among the candidate pairs. Here take as the most similar 
  # the query that has the highest similarity in the utility matrix by only 
  # looking at the candidate pairs
  prefs = []
  for i in similar_items[q]:
    if similarity == "jaccard":
      prefs.append(sim.jaccard_threshold(utility[:,q], utility[:,i], T=50))
    else:
      prefs.append(sim.cosine_similarity(utility[:,q], utility[:,i]))

  if len(similar_items[q]) == 0:
    res.append(-1) # no similar query found
  else:
    most_similar_query = np.argmax(prefs)
    most_similar_query_similarity = prefs[most_similar_query]

    res.append(list(similar_items[q])[most_similar_query])
    # print("The query most similar to {} is query {}, with a similarity of {} ".format(q, list(similar_items[q])[most_similar_query], most_similar_query_similarity))

  return res

'''
This function allows to measure the performance of LSH increasing the number of 
rows per band, i.e. decreasing the number of bands. 

Arguments:
  utility: the utility matrix
  similarity: the similarity measure to use, either "jaccard" or "cosine"
Returns:
  This function returns several performance measures in a tuple of three 
  elements(three lists):
  1. the number of items for which the estimated most similar item with LSH is 
    the same as the one found without LSH (correctly_estimated)
  2. the time to run LSH (time_to_run)
  3. the average number of candidate pairs found for each item (avg_candidates)
'''
def plot_bands_curve(utility, similarity):
  correctly_estimated = []
  time_to_run = []
  avg_candidates = []
  for r in range(1,15):
    start_time = int(time.time())

    # run simHash
    signature_matrix = lsh.simHash(utility_matrix=utility, hyperplanes=200)
    similar_items = lsh.lsh(signature_matrix, rows_per_band=r)

    end_time = int(time.time()) - start_time

    real = []
    estimated = []
    for i in range(1000):
      res = ev.most_similar_query(i, similarity=similarity, utility=utility, similar_items=similar_items)
      real.append(res[0])
      estimated.append(res[1])

    print("Test with rows_per_band=%d" % r)

    real = np.asarray(real)
    estimated = np.asarray(estimated)
    print("Number of correctly predicted similar items: %d with a total time of %d" % (np.sum((real - estimated ) == 0), end_time))

    lengths = [len(i) for i in similar_items.values()]
    avg_candidates.append(0 if len(lengths) == 0 else (float(sum(lengths)) / len(lengths)))

    correctly_estimated.append(np.sum((real - estimated ) == 0))
    time_to_run.append(end_time)

  return correctly_estimated, time_to_run, avg_candidates


'''
This function allows to measure the performance of LSH increasing the number of 
rows of the signature matrix. 

Arguments:
  utility: the utility matrix
  similarity: the similarity measure to use, either "jaccard" or "cosine"
Returns:
  This function returns several performance measures in a tuple of three 
  elements(three lists):
  1. the number of items for which the estimated most similar item with LSH is 
    the same as the one found without LSH (correctly_estimated)
  2. the time to run LSH (time_to_run)
  3. the average number of candidate pairs found for each item (avg_candidates)
'''
def plot_rows_curve(utility, similarity):
  correctly_estimated = []
  time_to_run = []
  avg_candidates = []
  for h in np.arange(50, 500, 50):
    start_time = int(time.time())

    # run simHash
    signature_matrix = lsh.simHash(utility_matrix=utility, hyperplanes=h)
    similar_items = lsh.lsh(signature_matrix, rows_per_band=11)

    end_time = int(time.time()) - start_time

    real = []
    estimated = []
    for i in range(1000):
      res = ev.most_similar_query(i, similarity=similarity, utility=utility, similar_items=similar_items)
      real.append(res[0])
      estimated.append(res[1])

    print("Test with number of hyperplanes=%d" % h)

    real = np.asarray(real)
    estimated = np.asarray(estimated)
    print("Number of correctly predicted similar items: %d with a total time of %d" % (np.sum((real - estimated ) == 0), end_time))

    lengths = [len(i) for i in similar_items.values()]
    avg_candidates.append(0 if len(lengths) == 0 else (float(sum(lengths)) / len(lengths)))

    correctly_estimated.append(np.sum((real - estimated ) == 0))
    time_to_run.append(end_time)

  return correctly_estimated, time_to_run, avg_candidates


'''
This function allows to measure the performance of LSH increasing the number of 
rows of the signature matrix and increasing the number of rows per band

Arguments:
  utility: the utility matrix
  similarity: the similarity measure to use, either "jaccard" or "cosine"
Returns:
  This function returns several performance measures in a tuple of three 
  elements(three lists):
  1. the number of items for which the estimated most similar item with LSH is 
    the same as the one found without LSH (correctly_estimated)
  2. the time to run LSH (time_to_run)
  3. the average number of candidate pairs found for each item (avg_candidates)
'''
def plot_combined_curve(utility, similarity):
  correctly_estimated = []
  time_to_run = []
  avg_candidates = []
  for h in np.arange(50, 500, 50):
    for r in range(5,20):

      start_time = int(time.time())

      # run simHash
      signature_matrix = lsh.simHash(utility_matrix=utility, hyperplanes=h)
      similar_items = lsh.lsh(signature_matrix, rows_per_band=r)

      end_time = int(time.time()) - start_time

      real = []
      estimated = []
      for i in range(1000):
        res = ev.most_similar_query(i, similarity=similarity, utility=utility, similar_items=similar_items)
        real.append(res[0])
        estimated.append(res[1])

      print("Test with number of hyperplanes=%d" % h)
      print("Test with rows_per_band=%d" % r)

      real = np.asarray(real)
      estimated = np.asarray(estimated)
      print("Number of correctly predicted similar items: %d with a total time of %d" % (np.sum((real - estimated ) == 0), end_time))

      lengths = [len(i) for i in similar_items.values()]
      avg_candidates.append(0 if len(lengths) == 0 else (float(sum(lengths)) / len(lengths)))

      correctly_estimated.append(np.sum((real - estimated ) == 0))
      time_to_run.append(end_time)

  print("Parameters:")
  for h in np.arange(50, 500, 50):
    for r in range(5,20):
      print("h: %d r: %d" % (h,r))

  return correctly_estimated, time_to_run, avg_candidates



'''
This function allows to measure the performance four algorithms to find the most 
similar items:
1. Standard way of making all the comparison O(n^2) using the modified version 
  of the Jaccard similarity
2. Standard way of making all the comparison O(n^2) using the cosine similarity
3. Using LSH with minHash and the Jaccard similarity
4. Using LSH with simHash and the Cosine similarity
Arguments:
  utility: the utility matrix
  algorithms: specify which algorithms to run either "all", "jaccard", "cosine"
Returns:
  A tuple of four elements where:
  1. the first element is the time(in seconds) needed to find similar items 
    without LSH, computing the Jaccard similarity between all the possible 
    combinations of queries
  2. the second element is the time(in seconds) needed to find similar items 
    without LSH, computing the Cosine similarity between all the possible 
    combinations of queries
  3. the third element is the time(in seconds) needed to find similar items 
    using minHash with LSH for the Jaccard similarity
  4. the fourth element is the time(in seconds) needed to find similar items 
    using simHash with LSH for the Cosine similarity
'''
def measure_time_performance(utility, algorithms="all"):
  rows, cols = utility.shape

  time_baseline_jaccard = None
  time_baseline_cosine = None
  time_minhash = None
  time_simhash = None

  
  # 1. find similar items without LSH(Jacccard similarity)
  if (algorithms == "all" or algorithms=="jaccard"):
    start_time = int(time.time())
    most_similar = [] # list containing in the ith position the query that is  
                      # most similar to query i

    for i in range(cols):
      similarities = []
      for j in range(cols):
        similarities.append(sim.jaccard_threshold(utility[:,i], utility[:,j], 50))

      most_similar_query = -1 if len(similarities)==0 else np.argmax(similarities)
      most_similar.append(most_similar_query)

    end_time = int(time.time()) - start_time
    time_baseline_jaccard = end_time

  # 2. find similar items without LSH(Cosine similarity)
  if (algorithms == "all" or algorithms=="cosine"):
    start_time = int(time.time())
    most_similar = [] # list containing in the ith position the query that is 
                      # most similar to query i

    for i in range(cols):
      similarities = []
      for j in range(cols):
        similarities.append(sim.cosine_similarity(utility[:,i], utility[:,j]))

      most_similar_query = -1 if len(similarities)==0 else np.argmax(similarities)
      most_similar.append(most_similar_query)

    end_time = int(time.time()) - start_time
    time_baseline_cosine = end_time

  # 3. find similar items with LSH: minHash
  if (algorithms == "all" or algorithms=="jaccard"):
    start_time = int(time.time())
    most_similar = []

    signature_matrix = lsh.minHash(utility_matrix=utility, k=200, T=50)
    similar_items = lsh.lsh(signature_matrix, rows_per_band=6)

    for i in range(cols):
      similarities = []
      for j in range(len(similar_items[i])):
        similarities.append(sim.jaccard_threshold(utility[:,i], utility[:,list(similar_items[i])[j]], 50))

      most_similar_query =  -1 if len(similarities)==0 else list(similar_items[i])[np.argmax(similarities)]
      most_similar.append(most_similar_query)

    end_time = int(time.time()) - start_time
    time_minhash = end_time

  # 4. find similar items with LSH: simHash
  if (algorithms == "all" or algorithms=="cosine"):
    start_time = int(time.time())
    most_similar = []

    signature_matrix = lsh.simHash(utility_matrix=utility, hyperplanes=200)
    similar_items = lsh.lsh(signature_matrix, rows_per_band=15)

    for i in range(cols):
      similarities = []
      for j in range(len(similar_items[i])):
        similarities.append(sim.cosine_similarity(utility[:,i], utility[:,list(similar_items[i])[j]]))

      most_similar_query =  -1 if len(similarities)==0 else list(similar_items[i])[np.argmax(similarities)]
      most_similar.append(most_similar_query)

    end_time = int(time.time()) - start_time
    time_simhash = end_time

  return time_baseline_jaccard, time_baseline_cosine, time_minhash, time_simhash


'''
This function measures different time performances of the algorithms(baseline 
without LSH and the ones with LSH) with different utility matrices sizes: for 
example increasing the number of users or increasing the number of queries.

Arguments:
  utility: the utility matrix
  num_users: the number of users to use for the test. Default: number of rows of
    the utility matrix
  num_queries: the number of queries to use for the test. Default: number of 
    columns of the utility matrix
  algorithms: specify which algorithms to run either "all", "jaccard", "cosine"
Returns:
  The function returns several performance measures in a tuple of four elements:
  1. the first element is a list of times(in seconds) needed to find similar 
    items without LSH, computing the Jaccard similarity between all the possible 
    combinations of queries
  2. the second element is a list of times(in seconds) needed to find similar 
    items without LSH, computing the Cosine similarity between all the possible 
    combinations of queries
  3. the third element is a list of times(in seconds) needed to find similar 
    items using minHash with LSH for the Jaccard similarity
  4. the fourth element is a list of times(in seconds) needed to find similar 
    items using simHash with LSH for the Cosine similarity
'''
def measure_time_increase(utility, num_users=[], num_queries=[], algorithms="all"):
  rows, cols = utility.shape
  num_users = num_users if len(num_users)>0 else [rows]
  num_queries = num_queries if len(num_queries)>0 else [cols]

  if (np.asarray(num_users) > rows + 1).any():
    print("Warning: you specified a number of users that is bigger than the ones available in the utility matrix")
    num_users = [rows]

  if (np.asarray(num_queries) > cols + 1).any():
    print("Warning: you specified a number of queries that is bigger than the ones available in the utility matrix")
    num_queries = [cols]
    

  time_baseline_jaccard = []
  time_baseline_cosine = []
  time_minhash = []
  time_simhash = []

  for users in num_users:
    for queries in num_queries:
      print("Measuring time performance with a utility matrix of size %d x %d (users x queries)" % (users, queries))
      # run the time performance measure on the sliced version of the utility matrix
      x, y, z, w = measure_time_performance(utility[:users-1,:queries-1], algorithms) 

      time_baseline_jaccard.append(x)
      time_baseline_cosine.append(y)
      time_minhash.append(z)
      time_simhash.append(w)

  return time_baseline_jaccard, time_baseline_cosine, time_minhash, time_simhash

'''
This function creates 4 subplots: 
1. the first shows the time performance of LSH with minHash compared to the 
  baseline solution with the Jaccard similarity, while increasing the size of 
  the utility matrix in term of number of queries.
2. the second shows the time performance of LSH with minHash compared to the 
  baseline solution with the Jaccard similarity, while increasing the size of 
  the utility matrix in term of number of users.
3. the first shows the time performance of LSH with simHash compared to the 
  baseline solution with the Cosine similarity, while increasing the size of 
  the utility matrix in term of number of queries.
4. the first shows the time performance of LSH with simHash compared to the 
  baseline solution with the Cosine similarity, while increasing the size of 
  the utility matrix in term of number of users.

Arguments:
  utility: the utility matrix
  num_users_j: a list containing the sizes of the utility matrix in terms of 
    users to be tested. This parameter is used for the minHash test.
  num_users_c: a list containing the sizes of the utility matrix in terms of 
    users to be tested. This parameter is used for the simHash test.
  num_queries_j: a list containing the sizes of the utility matrix in terms of 
    queries to be tested. This parameter is used for the minHash test.
  num_queries_c: a list containing the sizes of the utility matrix in terms of 
    queries to be tested. This parameter is used for the simHash test.
Returns:
  Plot the four aforementioned subplots
'''
def plot_time_increase(utility, num_users_j, num_users_c, num_queries_j, num_queries_c):

  time_jaccard_q, _, time_minhash_q, _ = ev.measure_time_increase(utility, num_queries=num_queries_j, algorithms="jaccard")
  time_jaccard_u, _, time_minhash_u, _ = ev.measure_time_increase(utility, num_users=num_users_j, num_queries=[1000], algorithms="jaccard")

  _, time_cosine_q, _, time_simhash_q = ev.measure_time_increase(utility, num_queries=num_queries_c, algorithms="cosine")
  _, time_cosine_u, _, time_simhash_u = ev.measure_time_increase(utility, num_users=num_users_c, algorithms="cosine")

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

  ax[1].plot(num_users_j, time_jaccard_u, label="Baseline solution Jaccard")
  ax[1].plot(num_users_j, time_minhash_u, label="LSH solution Minhash")
  ax[1].set_xlabel("Number of users (utility matrix rows)")
  ax[1].set_ylabel("Time (seconds)")
  ax[1].legend()
  ax[1].grid()
  ax[1].title.set_text("LSH for Jaccard - increasing the number of users")

  ax[3].plot(num_users_c, time_cosine_u, label="Baseline solution Cosine")
  ax[3].plot(num_users_c, time_simhash_u, label="LSH solution Simhash")
  ax[3].set_xlabel("Number of users (utility matrix rows)")
  ax[3].set_ylabel("Time (seconds)")
  ax[3].legend()
  ax[3].grid()
  ax[3].title.set_text("LSH for Cosine - increasing the number of users")

  plt.show()


'''
This function allows to measure the average error between the predicted ratings 
and the real ones 

Arguments:
  original_utility: the original utility matrix without sparsity
  predicted_utility: the utility matrix predicted using the recommendation 
    system
  sparsity_mask: the sparsity mask used to add sparsity to the original utility
Returns:
  This function returns the average difference between the real values values of 
  the utility matrix and the predicted ones: this is a value between 0 and 99
'''
def evaluate_prediction(original_utility, predicted_utility, sparsity_mask):
  new_mask = (predicted_utility!=0 & sparsity_mask) # remove from the mask the cells that were not predicted(no similar items found)
  predicted_diff = np.abs(original_utility[new_mask] - predicted_utility[new_mask])
  return predicted_diff