#!/usr/local/bin/python3
from utility.users import userGenerator, utilityMatrixGenerator
from utility.datasetUtility import generateRelationalTable, importCSV
from utility.queries import generateQueryDataset
import logging
import sys

#Creating and Configuring Logger

Log_Format = "%(levelname)s %(asctime)s - %(message)s"

logging.basicConfig(stream = sys.stdout, 
                    filemode = "w",
                    format = Log_Format, 
                    level = logging.INFO)

logger = logging.getLogger()

sparsity = 0.6

def synthetic():
    logger.info("User dataset generation")
    userArray = userGenerator(500)
    logger.info("Relational Table generation")
    relational_table = generateRelationalTable(10000, 100)
    logger.info("Query DataSets Generation")
    queryDataset = generateQueryDataset(relational_table, 2000, 10, False)
    logger.info("Utility Matrix DataSets Generation")
    utility_matrix = utilityMatrixGenerator(userArray, queryDataset, relational_table, sparsity)
    logger.info("Closing Generation Phase")

def real():
    logger.info("User dataset generation")
    userArray = userGenerator(500)
    logger.info("Relational Table generation")
    relational_table = importCSV("tests/Dataset/dataFolder/madelon_test.csv")
    logger.info("Query DataSets Generation")
    queryDataset = generateQueryDataset(relational_table, 1000, 10, True)
    logger.info("Utility Matrix DataSets Generation")
    utility_matrix = utilityMatrixGenerator(userArray, queryDataset, relational_table, sparsity)
    logger.info("Closing Generation Phase")


def main():
    logger.info("Starting DataSets Generation Phase")
    command = input("Would you like to use synthetic [s] dataset or real dataset [r]?")
    if command == 's':
        synthetic()
    elif command == 'r':
        real()
    else:
        logger.error("Wrong Input, try again!")

if __name__=='__main__':
    main()