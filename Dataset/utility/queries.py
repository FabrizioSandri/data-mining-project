#!/usr/local/bin/python3

import pandas as pd
import random
from .datasetUtility import csvSaver


def generateQueryDataset(inputDataset, queryMatrixRows, percentage_of_max_conditions = 0.05, real = False):
    #this function generate the dataset of the queries
    #It takes percentage_of_max_conditions as max conditions per query,
    #the percentage is low in order to have many results with high numbers of
    #answers. 
    #the function choose a random row of the relational table, select few columns
    #to set the conditions, and then save the query as a row of the query matrix.
    # first column of the matrix is teh query iD.
    # others are the columns of the relational table, if a condition is set the 
    # row + column will have the value of the condition, "" blank otherwise

    inputRows, inputColumns = inputDataset.shape
    queryMatrix = []
    for row in range(queryMatrixRows):
        query = []
        query.append("Q"+str(row))
        for i in range(inputColumns):
            query.append("")
        nConditions = random.randint(1, int(percentage_of_max_conditions*inputColumns))
        tableRow = random.randint(1, inputRows-1)
        for j in range(nConditions):
            col = random.randint(1, inputColumns-1)
            query[col] = inputDataset.iloc[tableRow][col]
        queryMatrix.append(query)
    columns_label = ["f"+str(i) for i in range(-1, inputColumns)]#col stand for column
    columns_label[0] = "Q"
    queryDataset_for_utilityMatrix = pd.DataFrame(queryMatrix, columns=columns_label)

    #generate the request querydataset
    q_set = []    
    for i in range(len(queryMatrix)):
        q_set_row = []
        q_set_row.append(queryMatrix[i][0])
        for j in range(1, len(queryMatrix[1])):
            if(queryMatrix[i][j] != ""):
                condition = "F"+str(j-1)+"="+str(queryMatrix[i][j])
                q_set_row.append(condition)
        q_set.append(q_set_row)
    
    queryDataset = pd.DataFrame(q_set)

    if real == True:
        dataName = "QueriesDataset_Real.csv"
    else:
        dataName = "QueriesDataset_Syntethic.csv"
    csvSaver(dataName=dataName, dataset=queryDataset)
    return queryDataset_for_utilityMatrix

def queryResults(inputDataset, query):
    # this function returns the results (the relational table rows)
    # the query would retrieve from the relational table.
    # this part is triviaò as the main feature to set a grade is how many rows are gotten as answer
    tmp = inputDataset
    for i in range(1, len(query)):
        if(query[i] != ""):
            Q = 'F'+str(i)+'=='+str(query[i])
            tmp = tmp.query(Q)
    filteredSet = tmp
    return filteredSet

def querySimilarity(inputDataset, query1, query2, threshold = 0.4):
    pass
    '''q1 = queryResults(inputDataset, query1)
    q2 = queryResults(inputDataset, query2)
    print(q1.columns.values)
    print(q2.columns.values)
    print("start similarity")
    union = q1.compare(q2, keep_equal = True)
    res = 0
    if len(q1) >= int(len(union*threshold)) and len(q2) >= int(len(union*threshold)):
        res = 1
    return res '''

def querySimilarityMatrix(inputdataset, querydataset):
    pass
    '''q_rows, q_columns = querydataset.shape
    sim_matrix = []
    for i in range(q_rows-1):
        sim_q = []
        for j in range(q_rows-1):
            print(querydataset.iloc[j])
            if(i!=j):
                res = querySimilarity(inputdataset, querydataset.iloc[i], querydataset.iloc[j])
            else:
                res = 1
            sim_q.append(res)
        sim_matrix.append(sim_q)
        print(sim_matrix)
    return sim_matrix'''