# idea

# structure: folders and files

```
project
│   README.md
│   data04.parquet
│   dtree01.joblib
│   main.py
│   noticias.csv
│
└───assets
│   │   file011.txt
│   │   file012.txt
│
└───data/
│   │   file011.txt
│   │   file012.txt
│
└───notebooks
│   │   file011.txt ```dasdad```
│   │   file012.txt
│
└───tests
│   │   file011.txt - $sadsd$
│   │   file012.txt
│
└───zextra
│   │   file011.txt: bla bla
│   │   file012.txt - bla bla
```


## File Descriptions

| File Name          | Description                                       |
|--------------------|---------------------------------------------------|
| **README.md**      | Documentation for the project, including setup and usage instructions. |
| **data04.parquet** | Primary dataset used for analysis.                |
| **dtree01.joblib** | Serialized decision tree model for predictions.   |
| **main.py**        | Main script to run the analysis and model.       |
| **noticias.csv**   | CSV file containing news articles or data.       |
| **assets/**        | Folder containing additional files related to the project. |
| **assets/file011.txt** | Text file used for supplementary data.         |
| **assets/file012.txt** | Another text file for additional information.  |
| **data/**          | Directory containing raw data files.              |
| **data/file011.txt** | Raw data file for processing.                   |
| **data/file012.txt** | Another raw data file.                           |
| **notebooks/**     | Directory containing Jupyter notebooks for analysis. |
| **notebooks/file011.txt** | Jupyter notebook for exploratory data analysis. |
| **notebooks/file012.txt** | Jupyter notebook for modeling.                |
| **tests/**         | Directory containing test files.                  |
| **tests/file011.txt** | Test case for validating data processing.        |
| **tests/file012.txt** | Test case for validating model predictions.      |
| **zextra/**        | Directory for extra files and datasets.           |
| **zextra/file011.txt** | Additional dataset or supplementary file.       |
| **zextra/file012.txt** | Another supplementary file.                     |


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