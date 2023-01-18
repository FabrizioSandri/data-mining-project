# Data mining project 2022/2023
## Query recommendation system
In this repository, there is all the material developed to complete the Data Mining Course 2022 project. The course was held at the University of Trento by prof. Yannis Velegrakis.
This Project is developed by **[Fabrizio Sandri](https://github.com/FabrizioSandri)** and **[Erik Nielsen](https://github.com/NielsenErik)**.
The objectives are to develop a sophisticated query recommendation system, then test and evaluate the build algorithm.

## How to run the algorithm
In this section it is explianed how to run the Python script generated to developed the algorithm.
1. > Run on terminal the main.py script
```
python3 main.py
```
2. > The program will show on terminal the following options:
Select the operation you want to do:
```
[1] Fill the blanks of the utility matrix using the Hybrid recommendation system(LSH + CF + Content based)
[2] Fill the blanks of the utility matrix using Collaborative filtering with LSH(LSH + CF)
[3] Evaluate the performance of LSH wrt the performance of running the algorithm without LSH
[4] Compare the RMSE and the MAE of running the algorithm using all the following methods
        a. Collaborative filtering with LSH(LSH + CF)
        b. Hybrid recommendation system with LSH(LSH + CF + content based)
        c. random ratings prediction

[5] Measure time performance and error rate increasing the signature matrix size
[6] Measure time performance and error rate increasing the number of rows per band of LSH
```
3. > Enter the requested option by digiting the appropriate number. 

In each of the aforementioned options it is used a different algorithm and the outcomes are diversified to save some time while testing the program.

Here is a description of what each option returns:

1. Return the complete Utility Matrix by using the Hybrid recommendation system that we propose; it uses LSH first to find candidate pairs, then apply a Collaborative Filtering approach, then the retrieved results are used as input for the Content-Based approach.

2. Return the complete Utility Matrix by using LSH and the Collaborative Filtering approach, this runs only the first part of the Hybrid Recommendation System.

3. Compare the performance between the algorithm if it uses LSH or if it doesn't

4. This evaluates the accuracy of the algorithms by comparing the RMSE and the MAE 
of the outcomes. **USE IT CAREFULLY, it takes a few hours to complete all the evaluations.** It runs 3 types of algorithms which are:
    * Collaborative filtering with LSH(LSH + CF)
    * Hybrid recommendation system with LSH(LSH + CF + content-based)
    * random rating prediction
5. This return the performance measures in term of time and error rate by increasing at each step the signature matrix size
6. This return the performance measures in term of time and error rate by increasing at each step the number of rows in each band of LSH

## Algorithm description and evaluation
The entire description of the algorithm is reported in the paper **HERE ADD DOCUMENTATION FOLDER** explaining the procedure and the development of the algorithm. In the aforementioned paper are described also the tests and the performance evaluations.

## How to generate new datasets
To generate new dataset and new utility matrix, it is necessary to run the `dataGenerator.py` script. Further informations about the Dataset and how to generate new dataset are in the README.md inside the **[Dataset](https://github.com/FabrizioSandri/data-mining-project/tree/main/Dataset)** folder
## Dataset generator description
All the description about the Dataset are in the README.md inside the **[Dataset](https://github.com/FabrizioSandri/data-mining-project/tree/main/Dataset)** folder