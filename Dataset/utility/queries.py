#!/usr/local/bin/python3

import pandas as pd
import numpy as np
from .datasetUtility import csvSaver

'''
This function generate the queries as a set of pseudo-random conditions.

Arguments:
    relational_table: the pandas dataframe containing the relational table
    numQueries: how many queries to generate
    max_conditions: the maximum number of conditions for a single query.
    real: specify whether the generated dataset should be real or synthetic

Returns:
    The query set as a pandas dataframe
'''
def generateQueryDataset(relational_table, numQueries, max_conditions = 20, real = False):
    relational_table = relational_table.convert_dtypes()
    rows, features = relational_table.shape
    q_set = []
    for queries in range(numQueries):
        query_conditions = []
        num_conditions = np.random.randint(0, max_conditions)

        for _ in range(num_conditions): 
            random_feature = relational_table.columns[np.random.randint(0, features)]

            # The value assigned to the feature is randomly chosen from the
            # relational table values for that column with a probability of 99%,
            # and with the remaining probability of 1% the value is chosen
            # randomly(if a numeric feature, a value between 0 an 1000,
            # otherwise a float value)
            if np.random.rand() < 0.99:
                random_feature_value = relational_table.loc[np.random.randint(0,rows), random_feature] 
            else:
                if pd.api.types.is_numeric_dtype(relational_table[random_feature]):
                    random_feature_value = int(np.random.randint(0,1000))
                else:
                    random_feature_value = np.random.rand()

            condition = random_feature + "=" + str(random_feature_value)
            query_conditions.append(condition)
            
        q_set.append(query_conditions)


    indexes = ["Q" + str(i) for i in range(numQueries)]
    queryDataset = pd.DataFrame(q_set, index=indexes)
    csvSaver(dataName="query_set.csv", dataset=queryDataset, header=False, index=True)

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