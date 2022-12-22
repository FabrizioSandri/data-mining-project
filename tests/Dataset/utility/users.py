#!/usr/local/bin/python3

import pandas as pd
import numpy as np
import random
import math
from .datasetUtility import csvSaver
from .queries import queryResults, generateQueryDataset

def userGenerator(nUser):
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
    #would be the grade of a given quer result, 
    # x in particular is the query result
    match userType:
        case 0:
            if(x == 0):
                res = random.randint(80,100)
            else:
                k = random.randint(60,99)
                res = int(k-x**1.05)
        case 1:
            if(x == 0):
                res = random.randint(1,20)
            else:
                k = random.randint(1,20)
                res = int(k+x**1.05)
        case 2:
            if(x == 0):
                res = random.randint(40,60)
            else:
                k = random.randint(30,70)
                res = int(k-x**1.05)
        case 3:
            k = random.randint(50,100)  
            res = int(math.log(x+math.exp((x+1)/(k*x))))
        case 4:    
            k = random.randint(50,100)          
            res = abs(int(math.log(x+math.exp(k*math.sin(x)))))
        case 5:    
            k = random.randint(50,100)        
            res = abs(int(k*math.cos(x)))
        case 6:   
            k = random.randint(50,100)         
            res = abs(int(k*math.tan(x)))
        case 7:   
            k = random.randint(50,100)         
            res = abs(int(k*math.sin(x+ math.log(x+math.exp(math.sin(x))))))
        case _:
            res = int(random.randint(1,100)*random.randint(1,100)/random.randint(1,100))
    if(res>100):
        res = random.randint(95,100)
    elif(res<1):
        res = random.randint(1,5)
    return res

def getUserGrades(queriesResult, userId):
    #user = userId.replace("usr", "")
    userType = random.randint(0,8)#randomize the user type
    userArray = []
    userArray.append(userId)
    for i in range(len(queriesResult)):
        userArray.append(gradeFunction(userType = userType, x = queriesResult[i]))
    return userArray

def utilityMatrixGenerator(userArray, queryDataset, inputdataset):
    #function which generate utility matrix
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
    utilityDataset = pd.DataFrame(utilityMatrix, columns=columns_label)
    dataName = "UtilityDataset.csv"
    csvSaver(dataName=dataName, dataset=utilityDataset)
    return utilityDataset

