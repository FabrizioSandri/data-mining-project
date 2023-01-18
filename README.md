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

In each of the aforementioned option it is used a different algorithm and the outcomes are diversify to save some time while testing the program.

Here a descriprion of what each option returns:

## How to generate new datasets
To generate new dataset and new utility matrix, it is necessary to run the `dataGenerator.py` script. Further informations about the Dataset and how to generate new dataset are in the README.md inside the **[Dataset](https://github.com/FabrizioSandri/data-mining-project/tree/main/Dataset)** folder
## Dataset generator description
All the description about the Dataset are in the README.md inside the **[Dataset](https://github.com/FabrizioSandri/data-mining-project/tree/main/Dataset)** folder