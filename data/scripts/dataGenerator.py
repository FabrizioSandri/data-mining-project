#!/usr/local/bin/python3
from utility.users import userGenerator, utilityMatrixGenerator
from utility.datasetUtility import generateRelationalTable, importCSV, csvSaver
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

def main():
    logger.info("Starting DataSets Generation Phase")
    command = input("Would you like to use synthetic [s] dataset or real dataset [r]?")
    
    if command == 's':
        logger.info("Relational Table generation")
        relational_table = generateRelationalTable(10000, 100)
    elif command == 'r':
        logger.info("Importing the real relational table")
        relational_table = importCSV("Dataset/dataFolder/adult.csv")
        csvSaver(dataName="relational_table.csv", dataset=relational_table, header=True, index=False)
    else:
        logger.error("Wrong Input, try again!")

    logger.info("User dataset generation")
    userArray = userGenerator(500)
    logger.info("Query DataSets Generation")
    queryDataset = generateQueryDataset(relational_table, 2000, 10, False)
    logger.info("Utility Matrix DataSets Generation")
    utility_matrix = utilityMatrixGenerator(userArray, queryDataset, relational_table, sparsity)
    logger.info("Closing Generation Phase")


if __name__=='__main__':
    main()