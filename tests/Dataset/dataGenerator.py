#!/usr/local/bin/python3
from utility.users import userGenerator, utilityMatrixGenerator
from utility.datasetUtility import generateRelationalTable
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


def main():
    logger.info("Starting DataSets Generation Phase")
    logger.info("User dataset generation")
    userArray, userDataset = userGenerator(2000)
    logger.info("Relational Table generation")
    relational_table = generateRelationalTable(10000, 150)
    logger.info("Query DataSets Generation")
    queryDataset = generateQueryDataset(relational_table, 2500)
    logger.info("Utility Matrix DataSets Generation")
    utility_matrix = utilityMatrixGenerator(userArray, queryDataset, relational_table)
    logger.info("Closing Generation Phase")
if __name__=='__main__':
    main()