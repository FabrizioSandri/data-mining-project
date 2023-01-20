#!/usr/local/bin/python3

import pandas as pd
import numpy as np
import random
import math
from sklearn.cluster import KMeans

from .datasetUtility import csvSaver
from .queries import queryResultsIds, generateQueryDataset

'''
This function generate the dataset of users

Arguments:
  nUser: the number of users of the entire system
  dataset_type: specify whether the generated dataset should be real or 
    synthetic

Returns:
  An array of user ids 
'''
def userGenerator(nUser, dataset_type):
    userArray = []
    for i in range(1, nUser):
        idUsr = "user" + str(i)
        userArray.append(idUsr)
    userDataset = pd.DataFrame(userArray)
    csvSaver(dataName="user_set.csv", dataset=userDataset, header=False, index=False, dataset_type=dataset_type)
    return userArray

'''
Auxiliary function used to add the first line containing the ids of the queries 
to the utility matrix file
'''
def query_header_prepender(filename, queries):
    line = ','.join(queries)
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content) 

'''
This function runs K-means with K=20 to identify clusters of queries based on 
the rows that they generate. 

Arguments:
  relational_table: the pandas dataframe containing the relational table
  query_set: the pandas dataframe containing the description of the queries sent
    by the users

Returns:
  A dictionary with the most similar queries to a given one
'''
def getMostSimilarQueriesClustering(relational_table, query_set):
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
This function returns the number of rows returned by all the queries

Arguments:
  relational_table: the pandas dataframe containing the relational table
  query_set: the pandas dataframe containing the description of the queries sent
    by the users

Returns:
  A list of number of rows returned by each query
'''
def getNumRows(relational_table, query_set):
    num_rows = []
    for query_id in range(query_set.shape[0]):
        rows = queryResultsIds(query_id, relational_table, query_set)
        num_rows.append(len(rows))
    
    return num_rows

'''
This function returns the ratings for a user of a given type for all the queries
in the dataset

Arguments:
  num_reational_table_rows: number of rows in the relational table
  num_queries: the total number of queries in the dataset
  userType: either 0(first 50%), 1(next 40%), 2(users that randomly rates)
  most_similar: dictionary containing the most similar queries to a given one
  num_rows: vector containing for each index i of the vector the number of rows
    returned by query i

Returns:
  A vector of grades given by the user to all the queries
'''
def getUserGrades(num_reational_table_rows, num_queries, userType, most_similar, num_rows):
    user_ratings = [''] * num_queries

    if userType == 0: # rate similar queries
        for i in range(num_queries):
            similar_ratings = []
            for similar_query in most_similar[i]:
                if user_ratings[similar_query] != '': # the user has already rated a similar query
                    similar_ratings.append(user_ratings[similar_query])
            
            if len(similar_ratings) == 0 or num_rows[i]==0: # the user has not rated any similar query or the query doesn't return any output: random rating
                user_ratings[i] = np.random.randint(1,101)
            else:
                user_ratings[i] = int(np.floor(np.mean(similar_ratings) + np.random.uniform(-5,5)))  # assign a rating that differ from the most similar by at most -5 and +5
                user_ratings[i] = user_ratings[i] if user_ratings[i] >= 1 and user_ratings[i] <= 100 else int(np.floor(np.mean(similar_ratings)))
            

    elif userType == 1: # rate based on the number of rows returned
        for i in range(num_queries):
            if num_rows[i] < np.floor(num_reational_table_rows * 0.05):
                user_ratings[i] = np.random.randint(1,25)
            elif num_rows[i] < np.floor(num_reational_table_rows * 0.10):
                user_ratings[i] = np.random.randint(25,50)
            elif num_rows[i] < np.floor(num_reational_table_rows * 0.25):
                user_ratings[i] = np.random.randint(50,75)
            else:
                user_ratings[i] = np.random.randint(75,101)

    else: # random rating users
        for i in range(num_queries):
            user_ratings[i] = np.random.randint(1,101)

    return user_ratings


'''
This function normalize the vector of ratings user_ratings in the range low-high

Arguments:
  low: the lowes grade, corresponding to grade 1
  high: the highest grade, corresponding to grade 100

Returns:
  A vector of normalized ratings
'''
def gradeNormalzation(user_ratings, low, high):
    norm = user_ratings/100
    norm *= (high - low)
    norm += low
    norm = np.round(norm).astype(int)

    return list(norm)


'''
Function that generates the utility matrix based on the query set and according to the grade provided by the getUserGrades function.


Arguments:
  userArray: the array os users that are part of the system
  queryDataset: the query set describing the conditions of the queries
  relational_table: the relational table containing all the tuples
  sparsity: how much of the utility matrix should be made sparse, i.e. the 
    ratings converted to 0(missing rating) 
  dataset_type: specify whether the generated dataset should be real or 
    synthetic

Returns:
  The utility matrix filled with the ratings
'''
def utilityMatrixGenerator(userArray, queryDataset, relational_table, sparsity, dataset_type):
    utilityMatrix = []
    nUsers = len(userArray)

    shuffled_user_array = np.asarray(userArray)
    np.random.shuffle(shuffled_user_array)

    # split the users into three categories:
    # - 60% of users who vote similar queries with approximately the same ratings
    # - 30% of users who rate queries based on the number of rows returned
    # - 10% of users who rate queries randomly
    simGradUsers = int(np.floor(nUsers * 0.6))
    propUsers = int(np.floor(nUsers * 0.3))
    randomUsers = nUsers - propUsers - simGradUsers # the remaining users (aprox 10%)

    # splitting the shuffled user array
    simGradUsersArray = shuffled_user_array[0:simGradUsers]
    propUsersArray = shuffled_user_array[simGradUsers:simGradUsers+propUsers]
    randomUsersArray = shuffled_user_array[propUsers+simGradUsers:nUsers]

    input_rows, input_columns = relational_table.shape
    q_rows, q_columns = queryDataset.shape


    # generate most_similar vector for the first 50% of users, and a vector of 
    # the number of rows returned by each query for the next 40% of the users 
    most_similar_queries = getMostSimilarQueriesClustering(relational_table, queryDataset)
    num_rows_per_query = getNumRows(relational_table, queryDataset)

    ##### PART 1
    for usr in simGradUsersArray:
        userType = 0
        utilityMatrix.append(getUserGrades(input_rows, q_rows, userType, most_similar_queries, num_rows_per_query))
    for usr in propUsersArray:
        userType = 1
        utilityMatrix.append(getUserGrades(input_rows, q_rows, userType, most_similar_queries, num_rows_per_query))
    for usr in randomUsersArray:
        userType = 2
        utilityMatrix.append(getUserGrades(input_rows, q_rows, userType, most_similar_queries, num_rows_per_query))

    ##### PART 2
    scales = [(1,25), (25,50), (50, 75), (75,100), (1,50), (50,100), (1,100)]

    np.random.shuffle(shuffled_user_array) # shuffle the user array again
    user_categories = np.array_split(shuffled_user_array, len(scales))
    for user_category_i in range(len(user_categories)):
        for user in user_categories[user_category_i]:
            user_id = int(user.replace("user", "")) - 1
            utilityMatrix[user_id] = gradeNormalzation(np.asarray(utilityMatrix[user_id]), scales[user_category_i][0], scales[user_category_i][1])

    columns_label = ["Q" + str(i) for i in range(q_rows)] 

    # generate sparsity
    sparsity_amount = int(np.floor(sparsity * nUsers * q_rows))   
    for i in range(sparsity_amount):
        row = np.random.randint(0, nUsers)
        column = np.random.randint(0, q_rows)
        utilityMatrix[row][column] = None

    utilityDataset = pd.DataFrame(utilityMatrix, columns=columns_label, index=userArray)
    csvSaver(dataName="utility_matrix.csv", dataset=utilityDataset, header=False, index=True, dataset_type=dataset_type)
    if dataset_type=="s":
        query_header_prepender("data/synthetic/utility_matrix.csv", columns_label)
    else:
        query_header_prepender("data/real/utility_matrix.csv", columns_label)
    
    return utilityDataset

