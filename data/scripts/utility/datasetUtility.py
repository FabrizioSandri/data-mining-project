from sklearn.datasets import make_multilabel_classification, make_gaussian_quantiles, make_blobs, make_moons
import pandas as pd
import numpy as np

def importCSV(path):
    df = pd.read_csv(path)
    return df

def csvSaver(dataName, dataset, header, index):
    path = "Dataset/dataFolder/"+dataName
    dataset = dataset.convert_dtypes()
    dataset.to_csv(path, header=header, index=index)

def generateRelationalTable(rows, columns):
    n_Samples = rows
    n_Features = columns
    n_Classes = 5
    n_Labels = 1

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

    columns_label = ["F"+str(i) for i in range(0, columns)] #F stands for feature
    dataset = pd.DataFrame(X, columns=columns_label)
    csvSaver(dataName="relational_table.csv", dataset=dataset, header=True, index=False)
    return dataset
