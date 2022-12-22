import numpy as np
from collections import defaultdict
from itertools import combinations
import similarity_measures as sim

'''
Hash function that defines a permutation for minHash

Arguments:
  x: integer number corresponding to the input to the hash function, i.e. the
    original permutation index(1..n) 
  a: integer number defining the first component of the linear equation
  b: integer number defining the second component of the linear equation 
  c: integer number defining the third component of the linear equation, 
    this parameter guarantees that the result of the hash function is less 
    than c. Generally this number should be a prime number to obtain more 
    evenly distributed permutations, however we generally set this to the 
    number of rows/cols in the utility matrix.
Returns:
  A random permutation of the index x in the range 0-c
'''
# hash function that defines a permutation
def h(x, a, b, c):
  return (a*x+b) % c


'''
This is a modified version of the minHash algorithm adapted to work with 
matrices of integers(not only 0s and 1s). This works by using a modified 
version of the Jaccard similarity that treats a value >= T as 1, 0 otherwise.

Arguments:
  utility_matrix: the utility matrix as a numpy matrix
  k: the number of hash functions for MinHash, corresponding to the number of 
    rows in the signature matrix
  T: the threshold above which the ith entry in the set is considered 1, 
    otherwise 0
Returns:
  The signature matrix with k rows
'''
def minHash(utility_matrix, k, T):
  rows, cols = utility_matrix.shape
  signature_matrix = np.full((k, cols), 0)

  # these hash functions simply generate random perturbations: instead of manually
  # creating random vectors of perturbation it is better to use hash functions 
  # that returns the perturbation in the rows (See chapter 3.3.5 of the book)
  hash_funs = np.random.randint(1, rows, (k,2)) 

  for item in range(cols):
    for hash_i in range(k):
      for user in range(rows):
        random_index = h(user, hash_funs[hash_i][0], hash_funs[hash_i][1], rows)
        if utility_matrix[random_index,item] >= T:
          signature_matrix[hash_i,item] = user+1
          break
  
  return signature_matrix


'''
This is the simHash algorithm implementation.

Arguments:
  utility_matrix: the utility matrix as a numpy matrix
  hyperplanes: the number of random hyperplanes to generate with simHash. This 
    number corresponds to the number of rows that the signature matrix will 
    have
Returns:
  The signature matrix with "hyperplanes" rows
'''
def simHash(utility_matrix, hyperplanes):
  utility_processed = utility_matrix - 50
  #utility[utility==-51] = 0   # -51 corresponds to the missing values

  rows, cols = utility_matrix.shape
  signature_matrix = np.full((hyperplanes, cols), 0)

  # plane's orthogonal vector with components -0.9,0,0.9
  # random_hyperplanes = np.random.choice([-0.9,0,0.9], size=(hyperplanes, rows)) 
  random_hyperplanes = np.random.uniform(low=-1, high=1, size=(hyperplanes, rows)) 

  for item in range(cols):
    for hyperplane in range(hyperplanes):
      dot_product = np.dot(utility_matrix[:,item], random_hyperplanes[hyperplane])
      if dot_product >= 0:
        signature_matrix[hyperplane, item] = 1
      else:
        signature_matrix[hyperplane, item] = 0
  
  return signature_matrix

'''
This function runs LSH on the signature matrix to find the similar items

Arguments:
  signature_matrix: the signature matrix as a numpy matrix
  bands: the number of bands into which to divide the signature matrix rows(you
    can specify wither "band" or rows_per_band)
  rows_per_band: the number of rows for each band(you can specify wither "band" 
    or rows_per_band)
  
Returns:
  The signature matrix with k rows
'''
def lsh(signature_matrix, bands=None, rows_per_band=None):
  rows, cols = signature_matrix.shape
  if bands:
    r = int(rows/bands)
  else:
    r = rows_per_band

  hashbuckets = defaultdict(set)

  for band_i in range(int(rows/r)): # for each band
    for query in range(cols):
      sliced_query = signature_matrix[band_i*r:(band_i+1)*r, query]
      band_id = tuple(sliced_query.tolist()+[str(band_i)])
      hashbuckets[band_id].add(query)

  # compute the similar items
  similar_items = defaultdict(set)
  for bucket in hashbuckets.values():
    if len(bucket) > 1:
      for pair in combinations(bucket, 2):
        similar_items[pair[0]].add(pair[1])
        similar_items[pair[1]].add(pair[0])

  return similar_items


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
      most_similar_query_i = np.argsort(prefs)[min(-K,len(prefs)):]
      k_most_similar[query] =  [list(similar_items[query])[i] for i in most_similar_query_i]

  return k_most_similar