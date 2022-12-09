import numpy as np
import time
from collections import defaultdict
from itertools import combinations

#### VARIABLES
num_users = 1000 # how many users in the random utility matrix
num_queries = 1000 # how many queries in the random utility matrix
utility_sparsity = 0.3 # sparsity of the random utility matrix(in percentage)

k = 300 # number of hash functions for MinHash
T = 50 # threshold value for consider a query as liked

r = 5 # rows per band in LSH, the number of bands is equal to b=k/r

'''
Standard version of Jaccard similarity 
'''
def jaccard(x,y):
  union = 0
  intersection = 0
  for x_elem, y_elem in zip(x,y):
    if x_elem != 0 or y_elem != 0:
      union += 1
      if (x_elem == y_elem ):
        intersection += 1
  
  return intersection/union

'''
Threshold version of Jaccard similarity where two queries ratings intersect if 
both are above T. For example T=50 and two queries match if their rating is both
above 50.
'''
def jaccard_threshold(x,y,T): 
  union = 0
  intersection = 0
  for x_elem, y_elem in zip(x,y):
    if x_elem != 0 or y_elem != 0:
      union += 1
      if (x_elem>=T and y_elem>=T): # they match if they are both above the threshold
        intersection += 1
  
  return intersection/union


## Similar queries:
# 0-1-4
# 2-6-8
# 3-5
# 7-9
# utility = np.array([ 
#   [ 25,  33,  93,  86,  15,  71,  96,   3,  90,  5],
#   [  6,   4,  33,  87,  22,  85,  38,  21,  30,  38],
#   [ 37,  40,  67,  63,  18,  58,  66,  31,  61,  15],
#   [  4,   5,  64,   4,  13,  15,  56,  77,  67,  88],
#   [ 88,  75,  67,  33,  79,  39,  70,  69,  65,  75],
#   [100,  95,   0,   6,  95,   3,   2,  49,  4,   39],
#   [ 72,  80,  79,  47,  75,  43,  65,  92,  67,  89],
#   [ 22,  24,   8,  32,  20,  26,  10,  16,   8,  30],
#   [ 93,  98,  12,  72,  87,  62,  21,  18,  16,   1],
#   [ 67,  60,   6,  29,  65,  39,   4,  75,   2,  88]
# ])


####################### RANDOM UTILITY MATRIX GENERATION #######################
utility = np.random.randint(0,101, (num_users, num_queries) )

# create some random similarities by copying columns
utility[:,57] = utility[:,900] 
utility[:,2] = utility[:,356] 
utility[:,797] = utility[:,156] 
utility[:,246] = utility[:,352] 

# fill some random values of the utility with 0. based on the sparsity amount
mask = np.random.choice([True, False], (num_users, num_queries) , p=[utility_sparsity,1-utility_sparsity])
utility[mask] = 0  

print("============ Utility matrix:")
print(utility)

################################# MIN HASHING ##################################
start_time = int(time.time())

# hash function that defines a permutation
def h(x, a, b, num_users):
  return (a*x+b) % num_users


signature_matrix = np.full((k, num_queries), 0)

# these hash functions simply generate random perturbations: instead of manually
# creating random vectors of perturbation it is better to use hash functions 
# that returns the perturbation in the rows (See chapter 3.3.5 of the book)
hash_funs = np.random.randint(1,num_users,(k,2)) 

for item in range(num_queries):
  for hash_i in range(k):
    for user in range(num_users):
      hashed_row = h(user, hash_funs[hash_i][0], hash_funs[hash_i][1], num_users)
      if utility[hashed_row,item] >= T:
        signature_matrix[hash_i,item] = user+1
        break


print("============ Signature Matrix:")
print(signature_matrix)

print("[Running Time] Time for computing the Signature matrix(Minhash): " + str(int(time.time()) - start_time) + " seconds")

################################## EVAUATION ###################################
# calculate how much time it takes to find similar items without using LSH, i.e.
# compute all the possible combinations of queries. In addition in this part I
# print the Jaccard similarity 
start_time = int(time.time())

real_sim = []
estimated_sim = []
for i in range(num_queries):
  for j in range(i+1, num_queries):
    real_sim.append(jaccard_threshold(utility[:,i], utility[:,j], T))
    estimated_sim.append(jaccard(signature_matrix[:,i], signature_matrix[:,j]))

real_sim = np.asarray(real_sim)
estimated_sim = np.asarray(estimated_sim)

diff = np.abs(real_sim - estimated_sim)
mean_diff = np.mean(diff)

print("Mean difference between the Jaccard similarity on the original utility matrix and the estimated jaccard similarity on the signature matrix: " + str(mean_diff))
print("[Running Time] Time for finding all the possible combinations of similar queries(Inefficient: without using LSH): " + str(int(time.time()) - start_time) + " seconds")

##################################### LSH ######################################
start_time = int(time.time())

hashbuckets = defaultdict(set)
for band_i in range(int(k/r)): # for each band
  for query in range(num_queries):
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

print("[Running Time] Time for finding the similar items using LSH: " + str(int(time.time()) - start_time) + " seconds")


#### EVALUATE THE RESULTS
# In this part I simply take query X and find the most similar query to it first
# using the standard method(computing the similarities of all the possible
# queries with query X that are (X,2), (X,3), (X,4), .. , (X,num_queries).
# In the second part instead I find the most similar queries using LSH; the 
# number of candidate pairs will be smaller with respect to all the combinations
# seen above. 
# Then I compare the similarity between the most similar query found without 
# using LSH(that is the real most similar query, but found in an inefficient 
# way) and I compare it with the best one found by using LSH.


def evaluate_sim(best_similar_to):
  ### 1. Find the best among all the queries (inefficient, need to check all the 
  # combinations with query 1: (1,2), (1,3), (1,4), .. , (1,num_queries))
  prefs = []
  for i in range(num_queries):
      prefs.append(jaccard_threshold(utility[:,best_similar_to], utility[:,i], T))

  prefs[best_similar_to] = 0
  most_similar_query = np.argmax(prefs)
  most_similar_query_similarity = prefs[most_similar_query]

  print("The query most similar to {} in the original utility matrix is query {}, with a similarity of {} ".format(best_similar_to, most_similar_query, most_similar_query_similarity))

  ### 2. Find the best among the candidate pairs. Here take as the most similar 
  # the query that has the highest similarity in the utility matrix by only 
  # looking at the candidate pairs
  prefs = []
  for i in similar_items[best_similar_to]:
      prefs.append(jaccard_threshold(utility[:,best_similar_to], utility[:,i], T))

  most_similar_query = np.argmax(prefs)
  most_similar_query_similarity = prefs[most_similar_query]

  print("The query most similar to {} is query {}, with a similarity of {} ".format(best_similar_to, list(similar_items[best_similar_to])[most_similar_query], most_similar_query_similarity))


evaluate_sim(57)
evaluate_sim(2)