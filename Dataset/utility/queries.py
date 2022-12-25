#!/usr/local/bin/python3

import pandas as pd
import numpy as np
import random
from .datasetUtility import csvSaver

'''
This function generate the dataset of the queries by taking
percentage_of_max_conditions as max conditions per query, the percentage is low
in order to have many results with high numbers of answers. the function choose
a random row of the relational table, select few columns to set the conditions,
and then save the query as a row of the query matrix. first column of the matrix
is teh query iD. others are the columns of the relational table, if a condition
is set the row + column will have the value of the condition, "" blank otherwise

Arguments:
    relational_table: the pandas dataframe containing the relational table
    queryMatrixRows: how many queries to generate
    percentage_of_max_conditions: ...
    real: specify whether the generated dataset should be real or synthetic

Returns:
    The query set as a pandas dataframe
'''
def generateQueryDataset(relational_table, queryMatrixRows, percentage_of_max_conditions = 0.05, real = False):

    inputRows, inputColumns = relational_table.shape
    queryMatrix = []
    for row in range(queryMatrixRows):
        query = [""] * inputColumns # fill the query feature with inputColumns blanks
        nConditions = random.randint(1, int(percentage_of_max_conditions*inputColumns))
        tableRow = random.randint(1, inputRows-1)
        for j in range(nConditions):
            col = random.randint(1, inputColumns-1)
            query[col] = relational_table.iloc[tableRow][col]
        queryMatrix.append(query)


    # generate the request querydataset
    q_set = []    
    for i in range(len(queryMatrix)):
        q_set_row = []
        for j in range(len(queryMatrix[1])):
            if(queryMatrix[i][j] != ""):
                condition = "F"+str(j-1)+"="+str(queryMatrix[i][j])
                q_set_row.append(condition)
        q_set.append(q_set_row)

    indexes = ["Q" + str(i) for i in range(queryMatrixRows)]
    queryDataset = pd.DataFrame(q_set, index=indexes)
    csv_file_name = "QueriesDataset_Real.csv" if real else "QueriesDataset_Syntethic.csv"
    csvSaver(dataName=csv_file_name, dataset=queryDataset, header=False, index=True)

    return queryDataset


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
  


def querySimilarity(relational_table, query1, query2, threshold = 0.4):
    pass
    '''q1 = queryResults(relational_table, query1)
    q2 = queryResults(relational_table, query2)
    print(q1.columns.values)
    print(q2.columns.values)
    print("start similarity")
    union = q1.compare(q2, keep_equal = True)
    res = 0
    if len(q1) >= int(len(union*threshold)) and len(q2) >= int(len(union*threshold)):
        res = 1
    return res '''

def querySimilarityMatrix(relational_table, querydataset):
    pass
    '''q_rows, q_columns = querydataset.shape
    sim_matrix = []
    for i in range(q_rows-1):
        sim_q = []
        for j in range(q_rows-1):
            print(querydataset.iloc[j])
            if(i!=j):
                res = querySimilarity(relational_table, querydataset.iloc[i], querydataset.iloc[j])
            else:
                res = 1
            sim_q.append(res)
        sim_matrix.append(sim_q)
        print(sim_matrix)
    return sim_matrix'''