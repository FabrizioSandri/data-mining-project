import numpy as np
import math

'''
Standard version of Jaccard similarity for sets

Arguments:
  a: set
  b: set
Returns:
  The Jaccard similarity of the two sets
'''
def jaccard(a, b):
  union = 0
  intersection = 0
  for a_i, b_i in zip(a,b):
    if a_i != -1 or b_i != -1:
      union += 1
      if (a_i == b_i ):
        intersection += 1
  
  return intersection/union


'''
Threshold version of Jaccard similarity where two queries ratings intersect if
both are above T. For example T=50 and two queries match if their rating is
both above 50.

Arguments:
  a: vector 
  b: vector 
  T: the threshold above which the ith entry in the set is considered 1, 
    otherwise 0
Returns:
  The Jaccard similarity of the two sets considering 1 the values greater than 
  T and 0 the values smaller than T
'''
def jaccard_threshold(a, b, T): 
  union = 0
  intersection = 0
  for a_i, b_i in zip(a,b):
    if a_i != 0 or b_i != 0:
      union += 1
      if (a_i>=T and b_i>=T): # match if both are above the threshold
        intersection += 1
  
  return intersection/union


'''
Cosine similarity between two vectors

Arguments:
  a: a vector 
  b: a vector 
Returns:
  The Cosine similarity of the two vectors
'''
def cosine_similarity(a, b):
  try:
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    if math.isnan(cos_sim):
      return 0
    else:
      return(cos_sim)
  except:
    print("An exception occurred") 
    print(list(a))
    print(list(b))
    

'''
Angular similarity between two vectors

Arguments:
  a: a vector 
  b: a vector 
Returns:
  The Angular similarity of the two vectors
'''
def angular_similarity(a,b):
    cos_sim = cosine_similarity(a, b)
    theta = np.arccos(cos_sim)
    return 1.0-(theta/np.pi)
