# STRUCTURE

## DATA

noticias.txt - list of news websites

data01.parquet - companies | aliases

data02.parquet - companies | aliases + everything from api request

data03.parquet - companies | aliases + NEED TO FILTER FROM API REQUEST (repeated content + content without the aliases)


## IPYNB

parquet.ipynb - makes reading parquet files easier

*main.ipynb* - where everything should come together

main_02.ipynb - correct script to generate both data01/02.parquet

**main_03.ipynb** - needs to fix the filtering to data03.parquet