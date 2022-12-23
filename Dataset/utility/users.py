#!/usr/local/bin/python3

import pandas as pd
import numpy as np
import random
import math
from .datasetUtility import csvSaver
from .queries import queryResults, generateQueryDataset

def userGenerator(nUser):
    #this function generate the dataset composed by the user iD
    #nUser = number of user to generate
    userArray = []
    for i in range(1, nUser):
        idUsr = "usr"+str(i)
        userArray.append(idUsr)
    columns_label = ["Users"]#col stand for column
    userDataset = pd.DataFrame(userArray, columns=columns_label)
    dataName = "userDataset.csv"
    csvSaver(dataName=dataName, dataset=userDataset)
    return userArray, userDataset

def gradeFunction(userType, x):
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
    match userType:
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
            res = abs(int(k*math.sin(x+ math.log(x+math.exp(math.sin(x))))))
        case _:
            res = int(random.randint(1,100)*random.randint(1,100)/random.randint(1,100))
    if(res>100):
        res = random.randint(10,100)
    elif(res<1):
        res = random.randint(1,10)
    return res

def getUserGrades(queriesResult, userId):
    #user = userId.replace("usr", "")
    #this function give randomly to each user a user type
    #so then can select the type of function to use
    #and give a grade to each query
    userType = random.randint(0,8)#randomize the user type
    userArray = []
    userArray.append(userId)
    for i in range(len(queriesResult)):
        userArray.append(gradeFunction(userType = userType, x = queriesResult[i]))
    return userArray

def utilityMatrixGenerator(userArray, queryDataset, inputdataset, sparsity = 0.3, real = False):
    #function which generate utility matrix
    #based on the query matrix and on the getUserGrades
    #return the final utilityMatrix
    #the first column have the user iD
    #the other columns each represent a query and have as values the grade
    #that each user gave to the query
    q_rows, q_columns = queryDataset.shape
    utilityMatrix = []
    query_answers = []
    for i in range(q_rows):
        q = queryDataset.iloc[i]
        results = queryResults(inputdataset, q)
        query_answers.append(len(results))

    for usr in userArray:
        utilityMatrix.append(getUserGrades(query_answers, usr))
    columns_label = []
    columns_label.append
    columns_label = ["Q"+str(i) for i in range(-1, q_rows)]#col stand for column
    columns_label[0] = "Usr"

    #generate sparsity
    sparsity_amount = int(sparsity * (len(utilityMatrix)) * (len(columns_label)-1))   
    for i in range(sparsity_amount):
        row = random.randint(1, (len(utilityMatrix)-1))
        column = random.randint(1, (len(columns_label)-1))
        utilityMatrix[row][column] = ''

    utilityDataset = pd.DataFrame(utilityMatrix, columns=columns_label)
    if real == True:
        dataName = "UtilityDataset_Real.csv"
    else:
        dataName = "UtilityDataset_Synthetic.csv"
    csvSaver(dataName=dataName, dataset=utilityDataset)
    return utilityDataset

