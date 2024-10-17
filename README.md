# STRUCTURE

## IDEAS


## DATA

noticias.csv - list of news websites and their names

data01.parquet - companies | aliases + everything from api requests

data02.parquet - companies | aliases + removed 100% duplicates + removed text without any alias

dtree01.csv - dataset to train the decision tree to choose the news

dtree01.joblib - 

data03.parquet - companies | aliases + applied decision tree


## IPYNB

parquet.ipynb - makes reading parquet files easier

*main.ipynb* - where everything should come together

main_01.ipynb - correct script to generate data01.parquet

main_02.ipynb - correct script to generate data02.parquet

dtree.ipynb - create dataset, train and explore the decision tree which chooses the news

**main_03.ipynb** - ...