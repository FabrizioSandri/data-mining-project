This folder contains two kind of datasets:
* synthetic
* real



## FOLDER STRUCTURE 

This directory contains the following three sub-folders:
* synthetic: synthetic dataset csv file
* real: real dataset csv file
* scripts: the scripts used to generate the datasets

Both the synthetic and real dataset folders follows the following structure:

├── query_set.csv : the query set

├── relational_table.csv : the relational table

├── user_set.csv : the user set

└── utility_matrix.csv : the utility matrix



## HOW TO USE THE SCRIPTS 

The main script is the `dataGenerator.py` script which once is run asks to the
user which kind of dataset to generate, either synthetic or real. The generated
datasets can be found inside the `dataFolder` folder. In order to obtain some
useful information from the datasets we created a `performance.py` script which
allows to measure the number of queries that returns at least one row and the
number of queries that returns no row. In addition this script allows to measure
the average amount of rows returned by the queries.



## DETAILED DESCRIPTION OF THE DATASETS 

The datasets were artificially constructed to as closely resemble a real-world
scenario. In particular, the synthetic datasets were produced starting with a
relational table that was filled with the Scikit-Learn library's `make_blobs`
function, which makes it possible to generate correlated data. The choice of
generating correlated data is motivated by the observation that in a real world
scenario, in a relational table representing for example people, there exist
groups of persons who share characteristics like eye color, height, etc. 

The query set is also built in a realistic manner; precisely, the conditions of
each query are constructed by selecting a random number of features (even zero),
giving each one a value in the feature's column with a chance of 99 percent,
and using the remaining probability to give a generic random value (even not in
the admitted values of that feature). By doing this, the queries are created in
a way that causes a great number of them to return some few rows, others to
return all the rows, and eventually no rows at all. It's important to note that
when a query has no conditions, all of the rows in the relational database are
returned. 

The core of the dataset is the utility matrix and to generate this the idea is
to create categories of users that are users tend to have the same taste of
others. In addition users tends to act following patters. For this reason to
fill the utility matrix ratings for the users the idea is to first split the
users in three categories: 

* 60 percent of the users rates queries that returns similar rows in the same
  way. If two queries q_1 and q_2 returns the majority of the rows in
  common, the user who rates query q_1 with rating r will rate query q_2
  with a rating that differs by r by a tiny factor gamma, for instance
  gamma = 5.

* 30 percent of users rate the queries proportionately with the amount of rows
  returned by the query. A user may be satisfied when the query returns some
  rows, but in another sense unsatisfied if the query does not return any row. 

* the remaining 10 percent of the users assign random ratings to the queries.


Even though users can evaluate queries on a scale from 1 to 100, it is
possible that two users will score a query differently in the real world. For
example, one user may rate a query positively with a rating of 50, while another
may rate the query positively with a rating of 100. Because of this, the user
ratings produced in the previous phase are randomly divided into the following
categories: users who rate on a scale of 1 to 50, users who rate on a range
of 50 to 100, 1 to 100, etc. Scaling the ratings is a straightforward
process that may be carried out using a simple normalization.

Note: the utility matrix is generated with a sparsity of 60% of the total.



## SYNTHETIC DATASET CHARACTERISTICS

The synthetic dataset is created as previously described using a relational
table with 100 features over a total of 10000 rows, and in addition, the values
of the relational table were filled with integer numbers; this latter choice is
wise because, in general, relational tables take values from a specific domain,
making representing cities by their names or by a representative integer number
equivalent. The dataset is designed to simulate a system with 500 users who have
submitted the DBMS a total of 2000 queries. Consequently, the utility matrix is
composed of 500 rows and 2000 columns (queries). Last but not least, a script
has been developed to calculate some crucial information from the relational
database and the queries, revealing that, out of a total of 2000 search queries,
721 produced at least one row, and the average amount of rows returned by these
queries is 3311. 

Even if the numbers appear to be small, they are sufficient to evaluate the
effectiveness of the proposed algorithms. They are particularly sufficient to
demonstrate how the algorithm's performance improves in various scenarios: this
dataset in particular was produced at that scale to enable its division into
smaller portions for even more specific experiments. 



## REAL DATASET CHARACTERISTICS

Relational tables typically contain experimental observations of real data; as a
result, the solutions proposed in this work are also evaluated on a dataset in
which a real relational table replaces the synthetic one. In this instance, the
relational table is taken from the relational database of the 1994 Census
Bureau(https://archive.ics.uci.edu/ml/datasets/Census%2BIncome). This dataset is
made up of 14 columns that reflect individual attributes including age, sex,
marital status, and income. The original dataset had over 50000 individuals,
however it was scaled down to only include 10000 individuals in order to measure
performance. The other components of the entire dataset are created using the
same methodology as the synthetic one, yielding a total of 2000 queries and 500
users. In this case the dataset revealed that, out of a total of 2000 search
queries, 908 produced at least one row, and the average amount of rows returned
by these queries is 3649. 

The relational table of the 1994 Census Bureau is stored in the
`scripts/dataFolder/adult.csv` file.