import numpy as np    
import time
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt

#### VARIABLES
num_users = 1000 # how many users in the random utility matrix
num_queries = 1000 # how many queries in the random utility matrix
utility_sparsity = 0.3 # sparsity of the random utility matrix(in percentage)

simhash_hyperplanes = 100 # number of hyperplanes for SimHash

r = 6 # rows per band in LSH, the number of bands is equal to b=k/r

'''
Compute the cosine similarity between two vectors
'''
def cosine_similarity(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    if cos_sim == np.nan:
      return 0
    else:
      return(cos_sim)

'''
Compute the angular similarity between two vectors
'''
def angular_similarity(a,b):
    cos_sim = cosine_similarity(a, b)
    theta = np.arccos(cos_sim)
    return 1.0-(theta/np.pi)

####################### RANDOM UTILITY MATRIX GENERATION #######################

utility = np.random.randint(0,101, (num_users, num_queries))

# add some similarities
utility[:,1] = utility[:,3] 
utility[:,2] = utility[:,5] 
utility[:,10] = utility[:,427] 

# remove some random values of the utility, based on the sparsity amount
mask = np.random.choice([True, False], (num_users, num_queries) , p=[utility_sparsity,1-utility_sparsity])
utility[mask] = -1


start_time = int(time.time())

signature_matrix = np.full((simhash_hyperplanes, num_queries), 0)

# rescale the utility matrix ratings in the range -50,50, setting the missing 
# values to 0

utility -= 50 
utility[utility==-51] = 0 
print("[Running Time] Time to rescale the utility: " + str(int(time.time()) - start_time) + " seconds")


################################# SIM HASHING ##################################
# random_hyperplanes = np.random.uniform(low=-1, high=1, size=(simhash_hyperplanes, num_users)) # plane's orthogonal vector with components that are reals from -1 to 1
random_hyperplanes = np.random.choice([-0.9,0,0.9], size=(simhash_hyperplanes, num_users)) # plane's orthogonal vector with components -0.9,0,0.9

# ## plot the vector defining each plane
# def plot_planes(planes):
#   for hyper in planes:
#     x_plane = [0, hyper[0]*100]
#     y_plane = [0, hyper[1]*100]
#     plt.plot(x_plane, y_plane, c="r")

# plt.scatter(utility[0],utility[1])
# plot_planes(random_hyperplanes)
# plt.show()
# ## end plot

start_time = int(time.time())

for item in range(num_queries):
  for hyperplane in range(simhash_hyperplanes):
    dot_product = np.dot(utility[:,item], random_hyperplanes[hyperplane])
    if dot_product >= 0:
      signature_matrix[hyperplane, item] = 1
    else:
      signature_matrix[hyperplane, item] = 0

print("============ Signature Matrix:")
print(signature_matrix)

print("[Running Time] Time for computing the Signature matrix(Simhash): " + str(int(time.time()) - start_time) + " seconds")


##################################### LSH ######################################
start_time = int(time.time())

hashbuckets = defaultdict(set)
for band_i in range(int(simhash_hyperplanes/r)): # for each band
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

def evaluate_sim(best_similar_to):
  ### 1. Find the best among all the queries (inefficient, need to check all the 
  # combinations with query 1: (1,2), (1,3), (1,4), .. , (1,num_queries))
  prefs = []
  for i in range(num_queries):
      prefs.append(cosine_similarity(utility[:,best_similar_to], utility[:,i]))

  prefs[best_similar_to] = 0
  most_similar_query = np.argmax(prefs)
  most_similar_query_similarity = prefs[most_similar_query]

  print("The query most similar to {} in the original utility matrix is query {}, with a similarity of {} ".format(best_similar_to, most_similar_query, most_similar_query_similarity))

  ### 2. Find the best among the candidate pairs. Here take as the most similar 
  # the query that has the highest similarity in the utility matrix by only 
  # looking at the candidate pairs
  prefs = []
  for i in similar_items[best_similar_to]:
      prefs.append(cosine_similarity(utility[:,best_similar_to], utility[:,i]))

  most_similar_query = np.argmax(prefs)
  most_similar_query_similarity = prefs[most_similar_query]

  print("The query most similar to {} is query {}, with a similarity of {} ".format(best_similar_to, list(similar_items[best_similar_to])[most_similar_query], most_similar_query_similarity))


evaluate_sim(1)
evaluate_sim(2)
evaluate_sim(10)