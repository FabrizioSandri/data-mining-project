import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

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
def queryResultsIds(query, relational_table, query_set):
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
This function counts the number of queries in the dataset that returned at least 
one row.

Arguments:
  relational_table: the pandas dataframe containing the relational table
  query_set: the pandas dataframe containing the description of the queries sent
    by the users

Returns:
  The number of queries that returns more than one row as output. 
'''
def countNonEmptyQueries(relational_table, query_set):
  non_zero = 0
  for query_id in range(query_set.shape[0]):
    row_ids = getRowsIds(query_id, relational_table, query_set)
    if len(row_ids) > 0 :
      non_zero += 1

  return non_zero

'''
This function returns the average number of rows returned by all the queries 
that returned at least one row.

Arguments:
  relational_table: the pandas dataframe containing the relational table
  query_set: the pandas dataframe containing the description of the queries sent
    by the users

Returns:
  The average number of rows returned by all the queries  
'''
def averageRowsReturned(relational_table, query_set):
  non_zero = 0
  average = 0
  for query_id in range(query_set.shape[0]):
    row_ids = getRowsIds(query_id, relational_table, query_set)
    if len(row_ids) > 0 :
      non_zero += 1
      average += len(row_ids)

  return average/non_zero



'''
This function runs K-means with K=20 to identify clusters of queries based on 
the rows that they generate.

Arguments:
  relational_table: the pandas dataframe containing the relational table
  query_set: the pandas dataframe containing the description of the queries sent
    by the users

Returns:
  The labels assigned by k-means to each query
'''
def getQueriesClusters(relational_table, query_set):
  K = 20
  queries = []

  for query_id in range(query_set.shape[0]):
    returned_rows_ids = queryResultsIds(query_id, relational_table, query_set)
    result_rows = np.full((relational_table.shape[0]), 0)
    result_rows[returned_rows_ids] = 1
    queries.append(result_rows)

  kmeans = KMeans(n_clusters=K, random_state=42) 
  kmeans.fit(queries)

  # generate a dictionary of query clsuters
  clusters = {}
  for i in range(K):
    clusters[i] = []

  for query_i in range(len(kmeans.labels_)):
    label = kmeans.labels_[query_i]
    clusters[label].append(query_i)

  # generate most_similar dictionary
  most_similar = {}
  for cluster_i in range(K):
    for query_i in clusters[cluster_i]:
      most_similar[query_i] = [query_j for query_j in clusters[cluster_i]]        

  return(most_similar)


'''
This function can be used to tweak the parameters of makeblobs
'''
def showMakeBlobsOutput():
  X, Y = make_blobs(
    n_samples=100,
    n_features=10,
    cluster_std=0.5,
    center_box=(1,20),
    random_state = 89
  )

  rows, columns = X.shape
  for i in range(rows):
    for j in range(columns):
      X[i][j]= abs(int(X[i][j]))

  plt.scatter(X[:,0], X[:,1], c=Y)
  plt.show()


if __name__=='__main__':
  query_set = pd.read_csv("Dataset/dataFolder/QueriesDataset_Syntethic.csv", index_col=0, header=None)
  relational_table = pd.read_csv("Dataset/dataFolder/RelationaTable_make_blobs.csv")
  relational_table = relational_table.convert_dtypes()

  clusters = getQueriesClusters(relational_table, query_set) 
  print(clusters)

  print("The dataset is made of a total of %d queries, of which %d returns at least one row" % (query_set.shape[0], countNonEmptyQueries(relational_table, query_set)))
  print("The dataset queries returns an average of %d rows" % (averageRowsReturned(relational_table, query_set)))
  


  
