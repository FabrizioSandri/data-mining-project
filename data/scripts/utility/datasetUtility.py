from sklearn.datasets import make_multilabel_classification, make_gaussian_quantiles, make_blobs, make_moons
import pandas as pd
import numpy as np

'''
This is an auxiliary function used to load a csv from file

Arguments:
  path: the full path of the csv file

Returns:
  A pandas dataframe
'''
def importCSV(path):
    df = pd.read_csv(path)
    return df

'''
This is an auxiliary function used to save the datasets as CSV file

Arguments:
  dataName: name of the dataset(prepended to the csv extension)
  dataset: the dataset as a pandas dataframe
  header: boolean values specifying whether to print the header line or not
  index: boolean values specifying whether to add the first column of indexes
  dataset_type: specify whether the generated dataset should be real or 
    synthetic
'''
def csvSaver(dataName, dataset, header, index, dataset_type):
    if dataset_type=="s" or dataset_type=='s':
        path = "data/synthetic/" + dataName
    else:
        path = "data/real/" + dataName

    dataset = dataset.convert_dtypes()
    dataset.to_csv(path, header=header, index=index)


'''
This is function generates a relational table synthetically using the make blobs
function of sklearn

Arguments:
  rows: number of rows of the relational table
  columns: number of columns of the relational table

Returns:
  The relational table as a Pandas dataframe and saves it to file
'''
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
    csvSaver(dataName="relational_table.csv", dataset=dataset, header=True, index=False, dataset_type="s")
    return dataset
