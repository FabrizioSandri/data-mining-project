from sklearn.datasets import make_multilabel_classification, make_gaussian_quantiles, make_blobs, make_moons
import pandas as pd
import numpy as np

def importCSV(path):
    df = pd.read_csv(path)
    columns = df.shape
    columns_label = ["F"+str(i) for i in range(0, columns[1])]
    df.columns = columns_label
    return df

def csvSaver(dataName, dataset, header, index):
    path = "Dataset/dataFolder/"+dataName
    dataset = dataset.convert_dtypes()
    dataset.to_csv(path, header=header, index=index)

def generateRelationalTable(rows, columns, typeDataset="make_blobs"):
    #typeDataset take some types available in scikit learn library 
    # and generate a dataset referring to that class
    #availabale are:
    # -> make_blobs
    # -> multilabel_classification
    n_Samples = rows
    n_Features = columns
    n_Classes = 5
    n_Labels = 1
    if(typeDataset == "multilabel_classification"):
        X, _ = make_multilabel_classification(
                n_samples=n_Samples,
                n_features=n_Features,
                n_classes=n_Classes,
                n_labels=n_Labels,
                allow_unlabeled=True,
                random_state=89
            )
    elif(typeDataset == "make_blobs"):
        X, Y = make_blobs(
            n_samples=n_Samples,
            n_features=n_Features,
            cluster_std=0.5,
            center_box=(1,10),
            random_state = 89
        )
        rows, columns = X.shape
        for i in range(rows):
            for j in range(columns):
                X[i][j]= abs(int(X[i][j]))
    rows, columns = X.shape
    columns_label = ["F"+str(i) for i in range(0, columns)] #F stands for feature
    dataset = pd.DataFrame(X, columns=columns_label)
    dataName = "RelationaTable_"+typeDataset+".csv"
    csvSaver(dataName=dataName, dataset=dataset, header=True, index=False)
    return dataset
