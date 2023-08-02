# Data mining project 2022/2023
[[Paper]](/report.pdf)

This repository contains all the material for the Data Mining Course project.
The course was held at the University of Trento by prof. Yannis Velegrakis. This
Project is developed by **[Fabrizio Sandri](https://github.com/FabrizioSandri)**
and **[Erik Nielsen](https://github.com/NielsenErik)**. The objective is to
develop a sophisticated query recommendation system, testing and evaluate the
solution.

## Requirements
Make sure to install the following Python libraries on your machine before
running the algorithms:
* [matplotlib](https://pypi.org/project/matplotlib/)
* [numpy](https://pypi.org/project/numpy/)
* [pandas](https://pypi.org/project/pandas/)
* [scikit_learn](https://pypi.org/project/scikit-learn/)

## Usage

This repository is structured in this way:
* `data` contains the datasets along with the scripts used to generate the
  datasets
* `doc` contains the latex source code of the paper
* `src` contains the source code of the algorithms devised as solutions for this
  problem

### How to generate a new dataset
The `data/scripts` folder contains all the scripts used to generate the datasets
used for testing the algorithms; please refer to
[data/README.md](data/README.md) for instructions on how to use the scripts.

### How to run the algorithm

You must supply a valid dataset before executing the algorithm; for additional
information on how to do so, go to the preceding section. Otherwise, you can
pick from one of the ones that are already in the `data` folder. In all
instances, after creating the dataset inside the `data` folder and giving it a
name like `datasetX`, you must specify which dataset to use for making
recommendations. To do this, you must change the `DATASET_TYPE` variable in the
first few lines of `src/main.py` and set it to the name of your new dataset. In
the preceding example, if we generated a dataset `data/datasetX`, we would need
to set `DATASET_TYPE="datasetX"` (without the `data/` prefix) in order to use
it. 

Once the dataset is ready to run the algorithm you have to place yourself in the
root of this repository and run the following command:
```shell
python3 src/main.py
```

The program will ask you which action to take: 
```
[1] Fill the blanks of the utility matrix 
[2] Compare the time performance of CF + LSH wrt CF (without LSH)
[3] Evaluate the accuracy(RMSE and MAE) of the following algorithms
	- Collaborative filtering with LSH(LSH + CF)
	- Hybrid recommendation system with LSH(LSH + CF + content based)
	- random ratings prediction

[4] Measure time performance and error rate increasing the signature matrix size
[5] Measure time performance and error rate increasing the number of rows per band of LSH

Select one option:
```
In this case, option `1` executes the algorithms, whereas choices `2` through
`5` do the experiments. Once you select option `1` you will be prompted with
another message asking for the version of the algorithm to run. Since the final
algorithm was built one block at a time, this prompt allows you to select which
version of the algorithm to execute. 

```
Select the algorithm to use for filling the blanks of the utility matrix: 
	[1] Collaborative filtering with LSH-MinHash(LSH + CF) 
	[2] Collaborative filtering with LSH-SimHash(LSH + CF) 
	[3] Hybrid recommendation system(LSH + CF + Content based)

Select one option: 
```
Enter the proper number to select the desired option. Here's a breakdown of which version of the algorithm each option uses: 
1. this option fills the missing ratings of the utility matrix by running
   collaborative filtering(CF) combined with LSH using the **MinHash** technique
   for LSH. This algorithm requires few seconds to complete it's execution.
2. this option fills the missing ratings of the utility matrix by running
   collaborative filtering(CF) combined with LSH using the **SimHash** technique
   for LSH. This algorithm requires few seconds to complete it's execution.
3. this option fills the missing ratings of the utility matrix by running
   collaborative filtering(CF) combined with LSH using the SimHash technique and
   on top of this runs a content based recommendation system to further improve
   the accuracy of the recommendations: this it the **hybrid recommendation
   system**. This algorithm may take several minutes to complete it's execution.
   
   
Once the aforementioned algorithm stops, the utility matrix filled with the
missing ratings will be located in the `data/synthetic` or `data/real` folder
containing the datasets, under the name of `predicted_utility.csv`. If you used
a custom dataset than the file `predicted_utility.csv` will be located inside
that dataset folder.


### Experimental evaluation tests

As previously noted, option `1` provides for the execution of several versions
of the recommendation system, whilst the other choices are all intended to
assess the quality of the recommendation system in terms of both time
performance and recommendation accuracy. Remember the prompt that appeared after
executing `python3 src/main.py`:
```
> python3 src/main.py

[1] Fill the blanks of the utility matrix 
[2] Compare the time performance of CF + LSH wrt CF (without LSH)
[3] Evaluate the accuracy(RMSE and MAE) of the following algorithms
	- Collaborative filtering with LSH(LSH + CF)
	- Hybrid recommendation system with LSH(LSH + CF + content based)
	- random ratings prediction

[4] Measure time performance and error rate by increasing the signature matrix size (CF + LSH)
[5] Measure time performance and error rate by increasing the number of rows per band of LSH (CF + LSH)

Select one option:
```

Here is a description of options `2-5`:

2. Measure and plot the time performance of running the first version of the
   algorithm using collaborative filtering only, by first running it without LSH
   and then applying LSH. The test compares the versions of CF + LSH for both
   MinHash and SimHash 

3. This evaluates the accuracy in terms of RMSE and MAE of the recommendations
   of four different versions of the algorithm:
   * Collaborative filtering with LSH(MinHash)
   * Collaborative filtering with LSH(SimHash)
   * Hybrid recommendation system
   * Random recommendation system: makes random recommendations

Note: Given that the Hybrid recommendation system is relatively slow compared to
the other algorithms, the execution of this test may take 2-3 hours to complete

4. This options allows to measure and plot the time performance and the error
   rate of finding similar queries by increasing at each step the signature
   matrix size in terms of rows(CF + LSH)

5. This options allows to measure and plot the time performance and the error
   rate of finding similar queries by increasing at each step the number of rows
   per band of LSH(CF + LSH)

The last two options(`4` and `5`) are meant to find good parameters for the
number of rows of the signature matrix and the number of rows per band of LSH.

## Algorithm description and evaluation
The entire description of the algorithm is reported in the paper that can be
found inside the [/doc](/doc) folder, explaining the procedure and the
development of the algorithm. The experimental evaluation of the algorithms is
also detailed in the report. 
