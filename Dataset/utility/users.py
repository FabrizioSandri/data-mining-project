#!/usr/local/bin/python3

import pandas as pd
import numpy as np
import random
import math
from .datasetUtility import csvSaver
from .queries import queryResultsIds, generateQueryDataset, querySimilarityMatrix

def userGenerator(nUser):
    #this function generate the dataset composed by the user iD
    #nUser = number of user to generate
    userArray = []
    for i in range(1, nUser):
        idUsr = "user"+str(i)
        userArray.append(idUsr)
    columns_label = ["Users"]#col stand for column
    userDataset = pd.DataFrame(userArray, columns=columns_label)
    dataName = "userDataset.csv"
    csvSaver(dataName=dataName, dataset=userDataset, header=False, index=False)
    return userArray, userDataset

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

def getUserGrades(queriesResult, userId, rel_table_total_rows, userType):
    #user = userId.replace("usr", "")
    #this function give randomly to each user a user type
    #so then can select the type of function to use
    #and give a grade to each query
    #userType = random.randint(0,8)#randomize the user type
    userArray = []
    userArray.append(userId)
    for i in range(len(queriesResult)):
        userArray.append(gradeFunction(userType, x = queriesResult[i],rel_table_total_rows = rel_table_total_rows))
    return userArray

def utilityMatrixGenerator(userArray, queryDataset, relational_table, sparsity = 0.3, real = False):
    #function which generate utility matrix
    #based on the query matrix and on the getUserGrades
    #return the final utilityMatrix
    #the first column have the user iD
    #the other columns each represent a query and have as values the grade
    #that each user gave to the query
    nUsers = len(userArray)
    #split users type index
    propUsers = int(nUsers * 0.4)
    simGradUsers = int(nUsers * 0.5)
    randomUsers = int(nUsers * 0.1)
    #check if all users got into the splitting
    totUsers = propUsers+simGradUsers+randomUsers
    if nUsers > totUsers:
        randomUsers += (nUsers-totUsers)

    #splitting the user array
    propUsersArray = userArray[:propUsers]
    simGradUsersArray = userArray[propUsers:propUsers+simGradUsers]
    randomUsersArray = userArray[propUsers+simGradUsers:nUsers]

    input_rows, input_columns = relational_table.shape
    q_rows, q_columns = queryDataset.shape
    utilityMatrix = []
    query_answers = []
    for q in range(q_rows):
        results = queryResultsIds(q, relational_table, queryDataset)
        query_answers.append(len(results))

    #k = querySimilarityMatrix(relational_table, queryDataset)
    #for loops for each user type
    for usr in propUsersArray:
        userType = 0
        utilityMatrix.append(getUserGrades(query_answers, usr, input_rows, userType))
    for usr in simGradUsersArray:
        userType = 1
        utilityMatrix.append(getUserGrades(query_answers, usr, input_rows, userType))
    for usr in randomUsersArray:
        userType = 2
        utilityMatrix.append(getUserGrades(query_answers, usr, input_rows, userType))
    columns_label = []
    columns_label = ["Q"+str(i) for i in range(-1, q_rows)]#col stand for column
    columns_label[0] = "Usr"

    #generate sparsity
    sparsity_amount = int(sparsity * (len(utilityMatrix)) * (len(columns_label)-1))   
    for i in range(sparsity_amount):
        row = random.randint(1, (len(utilityMatrix)-1))
        column = random.randint(1, (len(columns_label)-1))
        utilityMatrix[row][column] = ''

    utilityDataset = pd.DataFrame(utilityMatrix, columns=columns_label)
    utilityDataset = utilityDataset.sample(frac=1)
    if real == True:
        dataName = "UtilityDataset_Real.csv"
    else:
        dataName = "UtilityDataset_Synthetic.csv"
    csvSaver(dataName=dataName, dataset=utilityDataset, header=True, index=True)
    return utilityDataset

