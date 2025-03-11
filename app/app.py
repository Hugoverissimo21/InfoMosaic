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

# testing
from flask import redirect, url_for, request


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("News App") \
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

globalVAR = {}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def home():
    return render_template('index.html')  # Render the search page

@app.route('/search', methods=['GET'])
def search():
    global globalVAR

    # query requested
    query = request.args.get('query', '')
    globalVAR['query'] = query
    socketio.emit('status', {'message': 'Estou à procura de notícias com a palavra-chave: ' + query})

    # data filtering
    df_with_q = df.filter(F.col("keywords").getItem(query).isNotNull() & (F.col("keywords").getItem(query) > 4))
    socketio.emit('status', {'message': f'A ler {3034030493094039} notícias...'})
    globalVAR['query_amountofnews'] = df_with_q.count()

    # more then 0 news with the query?
    if globalVAR['query_amountofnews'] == 0:
        socketio.emit('status', {'message': f'Não encontrei notícias com a palavra-chave. Tente outra.'})
    
    else:
        socketio.emit('status', {'message': f'Encontrei {globalVAR["query_amountofnews"]} notícias com a palavra-chave.'})
        
        # query already processed?
        hashed_query = hashlib.sha256(query.encode()).hexdigest()[:10]
        if os.path.exists(f"cache/{hashed_query}.json"):
            socketio.emit('status', {'message': f'Encontrei um arquivo com o resultado da busca. Carregando...'})
            
            with open(f"cache/{hashed_query}.json", 'r') as json_file:
                globalVAR['keywords'] = json.load(json_file)
        
        else:
            # process the news if not processed yet
            socketio.emit('status', {'message': f'Processando notícias...'})
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
            globalVAR['keywords'] = {key: {"count": value[0],
                            "date": value[1],
                            "sentiment": value[2]/value[0],
                            "source": value[3],
                            "news": value[4]} for key, value in result}
            # save in cache
            with open(f"cache/{hashed_query}.json", 'w') as json_file:
                json.dump(globalVAR['keywords'], json_file)
        
        socketio.emit('status', {'message': f'Processamento concluído. Encontrei {len(globalVAR["keywords"])} palavras relacionadas.'})



    # Example: If your results are stored in a variable `search_results`
    search_results = ["result 1", "result 2", "result 3", query]  # Example results

    # After processing the search, render the template with results
    return render_template('index.html', hugo="Search completed!", results=search_results)






if __name__ == '__main__':
    socketio.run(app, debug=True)
