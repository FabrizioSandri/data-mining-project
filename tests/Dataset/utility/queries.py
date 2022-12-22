#!/usr/local/bin/python3

import pandas as pd
import random
from .datasetUtility import csvSaver


def generateQueryDataset(inputDataset, queryMatrixRows, percentage_of_max_conditions = 0.05):
    #this function generate the dataset of the queries

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
    tmp = inputDataset
    for i in range(1, len(query)):
        if(query[i] != ""):
            Q = 'F'+str(i)+'=='+str(query[i])
            tmp = tmp.query(Q)
    filteredSet = tmp
    return filteredSet