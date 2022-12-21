#!/usr/local/bin/python3
from utility.users import userGenerator, utilityMatrixGenerator
from utility.datasetUtility import generateRelationalTable
from utility.queries import generateQueryDataset
import logging as log


def main():
    print("Starting DataSets Generation Phase")
    print("User dataset generation")
    userArray, userDataset = userGenerator(200)
    print("Relational Table generation")
    relational_table = generateRelationalTable(10000, 50)
    print("Query DataSets Generation")
    queryDataset = generateQueryDataset(relational_table, 1000)
    print("Utility Matrix DataSets Generation")
    utility_matrix = utilityMatrixGenerator(userArray, queryDataset, relational_table)
    print("Closing Generation Phase")
if __name__=='__main__':
    main()