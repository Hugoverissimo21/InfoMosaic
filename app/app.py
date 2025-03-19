# Flask
from flask import Flask, render_template, request
from flask_socketio import SocketIO

# Spark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Others
import time
import os
import hashlib
import json

# Local
from graph import create_keyword_graph
from info import pie_newsSources, timeseries_news

# testing
from flask import redirect, url_for


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("News App") \
    .config("spark.ui.enabled", "false") \
    .getOrCreate()

# Define the data schema
schema = StructType([
    StructField("timestamp", IntegerType(), True),
    StructField("source", StringType(), True),
    StructField("archive", StringType(), True),
    StructField("id", IntegerType(), True),
    StructField("probability", FloatType(), True),
    StructField("keywords", MapType(StringType(), IntegerType()), True),
    StructField("sentiment", FloatType(), True)
])
df = spark.read.format("json").schema(schema).load("../data/news/status=success")

globalVar = {}

globalVar["graph_html"] = r'''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>HUGO</title><style>body{margin:0;height:100vh;background-color:pink;display:flex;justify-content:center;align-items:center;font-family:Arial,sans-serif;font-size:5rem;color:white;}</style></head><body><div>HUGO</div></body></html>'''

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def home():
    return render_template('index.html', globalVar=globalVar)

@app.route('/info')
def info():
    return render_template('info.html', globalVar=globalVar)

@app.route('/grafo')
def grafo():
    return render_template('graph.html', globalVar=globalVar)

@app.route('/search', methods=['GET'])
def search():
    global globalVar

    # query requested
    query = request.args.get('query', '')
    globalVar['query'] = query
    #socketio.emit('status', {'message': 'Estou à procura de notícias com a palavra-chave: ' + query})

    # data filtering
    query_col_counts = F.col("keywords").getItem(query)
    df_with_q = df.filter(query_col_counts.isNotNull() & (query_col_counts > 4)).cache()
    #socketio.emit('status', {'message': f'A ler {3034030493094039} notícias...'})
    globalVar['query_amountofnews'] = df_with_q.count()

    # more then 0 news with the query?
    if globalVar['query_amountofnews'] == 0:
        #socketio.emit('status', {'message': f'Não encontrei notícias com a palavra-chave. Tente outra.'})
        globalVar['keywords'] = {}
        globalVar["graph_html"] = create_keyword_graph(globalVar['keywords'], 150, query)

        # render the index page
        return render_template('graph.html', globalVar=globalVar)
    
    else:
        #socketio.emit('status', {'message': f'Encontrei {globalVar["query_amountofnews"]} notícias com a palavra-chave.'})
        
        # query already processed?
        hashed_query = hashlib.sha256(query.encode()).hexdigest()[:10]
        if os.path.exists(f"cache/{hashed_query}.json"):
            #socketio.emit('status', {'message': f'Encontrei um arquivo com o resultado da busca. Carregando...'})
            
            with open(f"cache/{hashed_query}.json", 'r') as json_file:
                globalVar['keywords'] = json.load(json_file)
        
        else:
            # process the news if not processed yet
            #socketio.emit('status', {'message': f'Processando notícias...'})
            result = (
                df_with_q.rdd
                .flatMap(lambda row: [
                    (key, (value,
                        {row["timestamp"]: value},
                        row["sentiment"]*value,
                        {row["source"]: 1},
                        [row["archive"]])) for key, value in row["keywords"].items()
                ])
                .reduceByKey(lambda a, b: (
                    a[0] + b[0],  # Sum count values
                    {ts: a[1].get(ts, 0) + b[1].get(ts, 0) for ts in set(a[1]) | set(b[1])},  # Merge timestamp counts
                    a[2] + b[2],  # Sum sentiment values
                    {source: a[3].get(source, 0) + b[3].get(source, 0) for source in set(a[3]) | set(b[3])},  # Merge source counts
                    a[4] + b[4]  # Merge archive lists
                ))
                .collect()
            )
            # change data schema
            globalVar['keywords'] = {key: {"count": value[0],
                            "date": value[1],
                            "sentiment": value[2]/value[0],
                            "source": value[3],
                            "news": value[4]} for key, value in result}
            # save in cache
            with open(f"cache/{hashed_query}.json", 'w') as json_file:
                json.dump(globalVar['keywords'], json_file)
        
        #socketio.emit('status', {'message': f'Processamento concluído. Encontrei {len(globalVar["keywords"])} palavras relacionadas.'})

        # create graph src code
        globalVar["graph_html"] = create_keyword_graph(globalVar['keywords'], 150, query)

        # create pie plot from news sources
        globalVar["pie_sources"] = pie_newsSources(df_with_q) 

        # create ts plot from news
        globalVar["ts_news"] = timeseries_news(df_with_q, query)


        # render the graph page
        return render_template('info.html', globalVar=globalVar)




if __name__ == '__main__':
    socketio.run(app, debug=True)
