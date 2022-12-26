#!/usr/local/bin/python3

import pandas as pd
import numpy as np
import random
import math
from sklearn.cluster import KMeans

from .datasetUtility import csvSaver
from .queries import queryResultsIds, generateQueryDataset, querySimilarityMatrix

'''
This function generate the dataset of users

Arguments:
  nUser: the number of users of the entire system

Returns:
  An array of user ids 
'''
def userGenerator(nUser):
    userArray = []
    for i in range(1, nUser):
        idUsr = "user" + str(i)
        userArray.append(idUsr)
    userDataset = pd.DataFrame(userArray)
    csvSaver(dataName="userDataset.csv", dataset=userDataset, header=False, index=False)
    return userArray

def gradeFunction(userType, x, rel_table_total_rows):
    #this function given a user type decide which
    #would be the grade of a given query result, 
    # x in particular is the query result
    #each user have same randomization to try to avoid identical users
    #
    #currently there are 8 types of users:
    #-> low # results good then exponential decade of the grade as the # increase
    #-> high # results good then exponential decade of the grade as the # decrease
    #-> middle range grade user with an exponential decade with high variance due to high randomization
    #-> math.log(x+math.exp((x+1)/(k*x))) function user, k gives some randomization
    #-> math.log(x+math.exp(k*math.sin(x))) function user, k gives some randomization
    #-> cosine grading user
    #-> tan grading user
    #-> sin grading user
    '''match userType:
        case 0:
            if(x == 0):
                res = random.randint(90,100)
            else:
                k = random.randint(70,99)
                res = int(k-x**1.05)
        case 1:
            if(x == 0):
                res = random.randint(1,15)
            else:
                k = random.randint(1,15)
                res = int(k+x**1.05)
        case 2:
            if(x == 0):
                res = random.randint(40,60)
            else:
                k = random.randint(30,70)
                res = int(k-x**1.05)
        case 3:
            k = random.randint(80,100)  
            res = int(math.log(x+math.exp((x+1)/(k*x))))
        case 4:    
            k = random.randint(80,100)          
            res = abs(int(math.log(x+math.exp(k*math.sin(x)))))
        case 5:    
            k = random.randint(80,100)        
            res = abs(int(k*math.cos(x)))
        case 6:   
            k = random.randint(80,100)         
            res = abs(int(k*math.tan(x)))
        case 7:   
            k = random.randint(80,100)         
            res = abs(int(k*math.sin(x+ math.log(x+math.exp(math.sin(x))))))'''
    res = random.randint(1,100) #to delete, there is a bug and sometimes userType isn't 
    #going inside the match case, when fixed must delete
    match userType:
        case 0: #regarding 40% of proportinal grad users
            if x < int(rel_table_total_rows * 0.05):
                    res = random.randint(1, 25)
            if x < int(rel_table_total_rows * 0.1) and x > int(rel_table_total_rows * 0.05):
                    res = random.randint(25, 50)
            if x < int(rel_table_total_rows * 0.25) and x > int(rel_table_total_rows * 0.10):
                    res = random.randint(50, 75)
            if x > int(rel_table_total_rows * 0.25):
                    res = random.randint(75, 100)
        case 1:
            res = random.randint(1,100)
        case 2:
            res = random.randint(1,100)
        case _:
            res = random.randint(1,100)
    if(res>100):
        res = random.randint(10,100)
    elif(res<1):
        res = random.randint(1,10)
    return res



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
            
            if len(similar_ratings) == 0: # the user has not rated any similar query: random rating
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
                user_ratings[i] = np.random.randint(75,100)

    else: # random rating users
        for i in range(num_queries):
            user_ratings[i] = np.random.randint(1,101)

    return user_ratings

'''
Function that generates the utility matrix based on the query set and according to the grade provided by the getUserGrades function.


Arguments:
  userArray: the array os users that are part of the system
  queryDataset: the query set describing the conditions of the queries
  relational_table: the relational table containing all the tuples
  sparsity: how much of the utility matrix should be made sparse, i.e. the 
    ratings converted to 0(missing rating) 

Returns:
  The utility matrix filled with the ratings
'''
def utilityMatrixGenerator(userArray, queryDataset, relational_table, sparsity = 0.3):
    utilityMatrix = []
    nUsers = len(userArray)

    shuffled_user_array = np.asarray(userArray)
    np.random.shuffle(shuffled_user_array)

    # split the users into three categories:
    # - 50% of users who vote similar queries with approximately the same ratings
    # - 40% of users who rate queries based on the number of rows returned
    # - 10% of users who rate queries randomly
    propUsers = int(np.floor(nUsers * 0.4))
    simGradUsers = int(np.floor(nUsers * 0.5))
    randomUsers = nUsers - propUsers - simGradUsers # the remaining users (aprox 10%)

    # splitting the shuffled user array
    propUsersArray = shuffled_user_array[0:propUsers]
    simGradUsersArray = shuffled_user_array[propUsers:propUsers+simGradUsers]
    randomUsersArray = shuffled_user_array[propUsers+simGradUsers:nUsers]

    input_rows, input_columns = relational_table.shape
    q_rows, q_columns = queryDataset.shape

    # generate most_similar vector for the first 50% of users, and a vector of 
    # the number of rows returned by each query for the next 40% of the users 
    most_similar_queries = getMostSimilarQueriesClustering(relational_table, queryDataset)
    num_rows_per_query = getNumRows(relational_table, queryDataset)

    # for loops for each user type
    for usr in propUsersArray:
        userType = 0
        utilityMatrix.append(getUserGrades(input_rows, q_rows, userType, most_similar_queries, num_rows_per_query))
    for usr in simGradUsersArray:
        userType = 1
        utilityMatrix.append(getUserGrades(input_rows, q_rows, userType, most_similar_queries, num_rows_per_query))
    for usr in randomUsersArray:
        userType = 2
        utilityMatrix.append(getUserGrades(input_rows, q_rows, userType, most_similar_queries, num_rows_per_query))

    columns_label = ["Q" + str(i) for i in range(q_rows)] 

    #generate sparsity
    sparsity_amount = int(np.floor(sparsity * nUsers * q_rows))   
    for i in range(sparsity_amount):
        row = np.random.randint(0, nUsers)
        column = np.random.randint(0, q_rows)
        utilityMatrix[row][column] = ''

    utilityDataset = pd.DataFrame(utilityMatrix, columns=columns_label, index=userArray)
    utilityDataset = utilityDataset.sample(frac=1)

    csvSaver(dataName="UtilityDataset.csv", dataset=utilityDataset, header=True, index=True)

    return utilityDataset

