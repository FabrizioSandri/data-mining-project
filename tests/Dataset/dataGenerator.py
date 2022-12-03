from sklearn.datasets import make_multilabel_classification, make_gaussian_quantiles
import pandas as pd
import numpy as np
import random

#Testing multi label classification to generate initial dataset

def csvSaver(dataName, dataset):
    path = "Dataset/dataFolder/"+dataName
    dataset.to_csv(path, index=False)

def generateDataMatrix(rows, columns, type = "multilabel_classification"):
    n_Samples = rows
    n_Features = columns
    n_Classes = 5
    n_Labels = 1
    if(type == "multilabel_classification"):
        X, _ = make_multilabel_classification(
                n_samples=n_Samples,
                n_features=n_Features,
                n_classes=n_Classes,
                n_labels=n_Labels,
                allow_unlabeled=True,
                random_state=89
            )
    elif(type == "gaussian"):
        n_Classes = 20
        X, _ = make_gaussian_quantiles(
                n_samples=n_Samples,
                n_features=n_Features,
                n_classes=n_Classes,
                random_state = 89
            )
    rows, columns = X.shape
    columns_label = ["F"+str(i) for i in range(0, columns)] #F stands for feature
    dataset = pd.DataFrame(X, columns=columns_label)
    dataName = "GeneralData_"+type+".csv"
    csvSaver(dataName=dataName, dataset=dataset)
    return dataset

def rowQuery(inputDataset, queryMatrixRows, percentage = 0.01):
    inputRows, inputColumns = inputDataset.shape
    queryMatrix = np.zeros((queryMatrixRows, inputColumns+1))# add 1 column as the 1st colums is the query id
    split = queryMatrixRows*percentage
    for i in range(0,int(queryMatrixRows)):
        queryMatrix[i][0] = i#"Q"+str(i)#first column of the matrix containing Query ids
        n_Conditions = random.randint(1, inputColumns)
        random_array = np.random.randint(0, inputColumns, size = n_Conditions)
        random_row = random.randint(0, inputRows-1)
        for j in random_array: 
            if(i%split == 3):
                random_row = random.randint(0, inputRows-1)        
            queryMatrix[i][j] =  inputDataset.iloc[random_row, j]

    return queryMatrix



def generateQueryMatrix(inputDataset, queryMatrixRows):
    inputRows, inputColumns = inputDataset.shape
    #queryMatrix = np.zeros((1, inputColumns+1))# add 1 column as the 1st colums is the query id

    queryMatrix = rowQuery(inputDataset, int(0.5*queryMatrixRows))

    rows, columns = queryMatrix.shape
    columns_label = ["Col"+str(i) for i in range(0, columns)]#col stand for column
    queryDataset = pd.DataFrame(queryMatrix, columns=columns_label)
    dataName = "QueriesDataset.csv"
    csvSaver(dataName=dataName, dataset=queryDataset)
    return queryDataset, queryMatrix

def getRows(inputDataset, query):
    tmp = inputDataset
    for q in range(len(query)):
        if(query[q] != 0):
            Q = 'F'+str(q)+'=='+str(query[q])
            tmp = tmp.query(Q)
    filteredSet = tmp
    return filteredSet

def gradeFunction(usrType, x): 
    #When a query is useful?
    #-if return results->if not grade=0
    #-return low number of instances, easyto check
    #-return on top the wanted instances
    #-generally if you can find quick and easy the wanted instances
    #-long queries cab be penelized
    #
    #
    #Key aspect is that not all user will say that 0 results = to 0% grade, so 
    #added functtion that randomize user which could like to have 0 results,
    #same thing abput having a lot of results or not, then the case of 
    #people giving only middle range grades
    #then also people giving random values
    #Of course not all possible cases are considered
    match usrType:
        case 0:
            if(x == 0):
                res = random.randint(90,100)
            else:
                res = random.randint(int(89-x**1.5),int(99-x**1.5))
        case 1:
            if(x == 0):
                res = random.randint(0,10)
            else:
                res = random.randint(int(11+x**1.5),int(30+x**1.5))
        case 2:
            if(x == 0):
                res = random.randint(40,60)
            else:
                res = res = random.randint(int(30+x**1.5),int(70+x**1.5))
        case 3:
            res = random.randint(0,100)
        case _:
            res = random.randint(0,100)
    if(res>100):
        res = 100
    elif(res<0):
        res = 0
    return res

def getUserGrades(queriesResult, usrId):
    usrType = usrId % 3
    usrArray = np.zeros(len(queriesResult)+1)
    usrArray[0] = usrId
    for i in range(1, len(queriesResult)):
        usrArray[i] = gradeFunction(usrType=usrType, x = queriesResult[i])
    return usrArray

def generateUser(nUser):
    user = np.zeros(nUser)
    for i in range(len(user)):
        user[i] = i
    columns_label = ["Users"]#col stand for column
    userDataset = pd.DataFrame(user, columns=columns_label)
    dataName = "userDataset.csv"
    csvSaver(dataName=dataName, dataset=userDataset)
    return user, userDataset

    

def generateUtilityMatrix(inputDataset, queryMatrix, userArray):
    #When a query is useful?
    #-if return results->if not grade=0
    #-return low number of instances, easyto check
    #-return on top the wanted instances
    #-generally if you can find quick and easy the wanted instances
    #-long queries cab be penelized
    #print(queryMatrix)
    lengthResults = np.zeros(len(queryMatrix))
    for i in range(0, len(queryMatrix)):
        results = getRows(inputDataset, queryMatrix[i])
        lengthResults[i] = len(results)
    print(lengthResults)
    utility = np.zeros((len(userArray), len(queryMatrix)+1))
    countUsr = 0
    for usr in userArray:
        utility[countUsr] = getUserGrades(lengthResults, usr)
        countUsr +=1
    rows, columns = utility.shape
    
    columns_label = ["Q"+str(i-1) for i in range(0, columns)]#col stand for column
    columns_label[0] = "userId"
    utilityDataset = pd.DataFrame(utility, columns=columns_label)
    dataName = "utilityMatrixDataset.csv"
    csvSaver(dataName=dataName, dataset=utilityDataset)
    return utility, utilityDataset

dataset = generateDataMatrix(500, 20)
query, queryMatrix = generateQueryMatrix(dataset, 70)
userArray, userMatrix = generateUser(100)
_ , utilityDataset = generateUtilityMatrix(dataset, queryMatrix, userArray)
print(utilityDataset)
