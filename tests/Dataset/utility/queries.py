#!/usr/local/bin/python3

import pandas as pd
import random
from .datasetUtility import csvSaver


def generateQueryDataset(inputDataset, queryMatrixRows, percentage_of_max_conditions = 0.05):
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
    queryDataset = pd.DataFrame(queryMatrix, columns=columns_label)
    dataName = "QueriesDataset.csv"
    csvSaver(dataName=dataName, dataset=queryDataset)
    return queryDataset

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