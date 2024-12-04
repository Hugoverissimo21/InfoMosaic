from flask import Flask, render_template, request, Response, jsonify, g
import random
import matplotlib.pyplot as plt
import io
import requests
import base64
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objs as go
import plotly.io as pio

##################################################################
##################################################################

# SEARCH | SEARCH | SEARCH | SEARCH | SEARCH
search_query = None
avaible_queries = ["galp",
                   "bcp",
                   "edp",
                   "sonae",
                   "motgil"]

# HILO | HILO | HILO | HILO | HILO | HILO
keywords_data = None
keywords = None
word1 = None
word2 = None
score = None
scores_historic = {}   

# NEWS READER | NEWS READER | NEWS READER
news_data = None
news_url = None
setences = []
ratings = None
vectorizer = TfidfVectorizer()
tfidf_matrix = None
def update_ratings(index, user_rating):
    global ratings
    global number_news_read
    number_news_read += 1
    similarity_scores = cosine_similarity(tfidf_matrix[index], tfidf_matrix).flatten()
    learning_rate = 0.999**number_news_read
    ratings += (user_rating - ratings) * (similarity_scores)**0.5 * learning_rate
    ratings[index] = -1000
number_news_read = 0
current_new_index = 0
read_news_index = []
read_news_index_curr = -1

##################################################################
##################################################################

# Intialize the Flask app
app = Flask(__name__)

@app.before_request
def before_request():
    global avaible_queries
    g.avaible_queries = avaible_queries

##################################################################
##################################################################
######################### start page #############################
##################################################################
##################################################################

@app.route('/')
def home():
    global avaible_queries
    return render_template('app.html', search_not_done=True)

##################################################################
##################################################################
############################ search ##############################
##################################################################
##################################################################

@app.route('/search', methods=['POST'])
def search():
    global search_query    
    global avaible_queries

    search_query = request.form['query']
    action = request.form.get('action')
    if action == 'search':
        if search_query not in avaible_queries:
            return render_template('app.html', search_query="404Hugo",
                                search_not_done=True)
    elif action == 'lucky':
        search_query = random.choice(avaible_queries)
    
    # Load the keywords from the JSON file
    global keywords_data
    global keywords
    with open(f'data/kwrd_{search_query}.json', 'r', encoding='utf-8') as json_file:
        keywords_data = json.load(json_file)
        keywords_data = {k: v for k, v in keywords_data.items() if int(v["count"]) > 5 and v["filter"] >= 0.015}
    keywords = list(keywords_data.keys())

    # Load the news from the JSON file
    global news_data
    global news_url
    global setences
    global ratings
    global tfidf_matrix
    with open(f'data/news_{search_query}.json', 'r', encoding='utf-8') as json_file:
        news_data = json.load(json_file)
    news_url = list(news_data.keys())
    setences = []
    for new in news_url:
        setence = " ".join(news_data[new]["keywords"]) # set
        setences.append(setence)
    ratings = np.array([3.0] * len(setences))
    tfidf_matrix = vectorizer.fit_transform(setences)
      
    return render_template('app.html', search_query=search_query,
                           search_finised=True)


##################################################################
##################################################################
############################# graph ##############################
##################################################################
##################################################################

@app.route('/graph', methods=['GET'])
def graph():
    global search_query
    graph_site = f"https://hugoverissimo21.github.io/FCD-project/assets/graph_{search_query}.html"

    response = requests.get(graph_site)
    buffer = io.StringIO()
    buffer.write(response.text)
    buffer.seek(0)
    
    return Response(buffer.getvalue(), mimetype='text/html')


##################################################################
##################################################################
############################ hilo ################################
##################################################################
##################################################################

@app.route('/hiloH_start', methods=['POST'])
def hiloH_start():
    global keywords
    global word1
    global word2
    global score
    global scores_historic
    word1 = random.choice(keywords)
    word2 = random.choice(keywords)
    while word2 == word1:
        word2 = random.choice(keywords)
    word1_mentions = int(keywords_data[word1]["count"])
    score = 0
    scores_historic = {}
    scores_historic[1] = {"score": score,
                          "text": "Current game! GL"}
    return render_template('app.html', word1=word1, word2=word2,
                           word1_mentions=word1_mentions, score=score)

@app.route('/hiloH', methods=['POST'])
def hiloH():
    global keywords
    global word1
    global word2
    global score
    global scores_historic
    word1_mentions = int(keywords_data[word1]["count"])
    word2_mentions = int(keywords_data[word2]["count"])
    positive_feedback = [
        "Great job!",
        "Well done!",
        "Fantastic!",
        "Awesome!",
        "Keep it up!",
        "Excellent!",
        "Brilliant!",
        "Nice work!",
        "Superb!",
        "Impressive!",
        "Amazing!",
        "Outstanding!",
        "Terrific!",
        "You're doing great!",
        "Good job!",
        "Perfect!",
        "Fabulous!",
        "You're on the right track!",
        "Incredible!",
        "Marvelous!",
        "Good effort!",
        "Keep going!",
        "You're crushing it!",
        "Spectacular!",
        "Top-notch!"
        ]


    if request.method == 'POST':
        user_choice = request.form.get('choice')  # Get the user's choice
        scenarios = {
            "more": word2_mentions >= word1_mentions,
            "less": word2_mentions <= word1_mentions
        }
        if scenarios[user_choice]:
            if score == 0:
                winner = "Correct!"
            else:
                winner = random.choice(positive_feedback)
            loser = False
            score += 1
            scores_historic[max(scores_historic.keys())]["score"] = score
            # reload the keywords
            word1 = word2
            word1_mentions = word2_mentions
            word2 = random.choice(keywords)
            while word2 == word1:
                word2 = random.choice(keywords)
        else:
            winner = False
            loser = f"Oops! That didn't go as planned. Final score: {score}."
            scores_historic[max(scores_historic.keys())]["score"] = score
            scores_historic[max(scores_historic.keys())]["text"] = f"""
            Score: {score}
            <br>Word 1: {word1} ({word1_mentions})
            <br>Word 2: {word2} ({word2_mentions})"""
            score = 0
            scores_historic[max(scores_historic.keys())+1] = {"score": score,
                                                             "text": "Current game! GL"}
            word1 = word2
            word1_mentions = word2_mentions
            word2 = random.choice(keywords)
            while word2 == word1:
                word2 = random.choice(keywords)
    
    return render_template('app.html', word1=word1, word2=word2,
                           word1_mentions=word1_mentions, score=score,
                           winner=winner, loser=loser)

@app.route('/hilo_plot', methods=['GET'])
def hilo_plot():
    global scores_historic
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(scores_historic.keys()),
                             y=[x["score"] for x in scores_historic.values()],
                             mode='lines+markers',
                             hovertext=[x["text"] for x in scores_historic.values()],
                             marker=dict(color=["blue"]*(len(scores_historic)-1) + ["red"]),
                             hoverinfo='text'))
    fig.update_layout(xaxis_title='Game Number',
                      yaxis_title='Score Achieved',
                      margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(
                          gridcolor='rgba(208, 216, 226, 0.4)',
                          gridwidth=1,
                          linecolor='rgba(208, 216, 226, 1)',
                          linewidth=2,
                          type='category'),
                        yaxis=dict(
                            gridcolor='rgba(208, 216, 226, 0.35)',
                            gridwidth=1,
                            linecolor='rgba(208, 216, 226, 1)',
                            linewidth=2,
                            ))

    # Use io.StringIO to save the HTML content in memory
    buffer = io.StringIO()
    pio.write_html(fig, file=buffer,
                   full_html=True,
                   config={"displayModeBar": False})
    buffer.seek(0)  # Move to the start of the buffer

    return Response(buffer.getvalue(), mimetype='text/html')


    
##################################################################
##################################################################
#$####################### readexplore ############################
##################################################################
##################################################################

@app.route('/get_first_news', methods=['POST'])
def get_first_news():
    global ratings
    global news_url
    global current_new_index
    global read_news_index
    read_news_index = []
    # weight for the random choice
    weights = np.exp(np.array(ratings))
    weights = weights / weights.sum()
    # random news index
    current_new_index = np.random.choice(len(ratings), p=weights)
    new_to_read = news_url[current_new_index]
    read_news_index.append(current_new_index)

    return render_template('app.html', new_to_read=new_to_read,
                           not_first_new=True, current=True,
                           no_more_left=True, no_more_right=True)

@app.route('/rate_news', methods=['POST'])
def rate_news():
    global current_new_index
    global ratings
    global news_url
    global read_news_index
    # process rate
    rate = int(request.form.get('rating4new'))
    if rate != -1:
        update_ratings(current_new_index, rate)
    else:
        pass
    # give a new news
    weights = np.exp(np.array(ratings))
    weights = weights / weights.sum()
    # random news index
    current_new_index = np.random.choice(len(ratings), p=weights)
    new_to_read = news_url[current_new_index]
    read_news_index.append(current_new_index)

    return render_template('app.html', new_to_read=new_to_read,
                           not_first_new=True, current=True,
                           no_more_left=False, no_more_right=True)

@app.route('/news_history', methods=['POST'])
def news_history():
    global read_news_index
    global news_url
    global current_new_index
    global read_news_index_curr
    no_more_left = False
    no_more_right = False

    move = request.form.get('news_hist')
    if move == "next":
        read_news_index_curr += 1
    elif move == "prev":
        read_news_index_curr -= 1
    
    new_to_read = news_url[read_news_index[read_news_index_curr]]
    
    if read_news_index_curr == -1:
        current = True
    else:
        current = False
    
    if read_news_index_curr == -len(read_news_index):
        no_more_left=True
    elif read_news_index_curr == -1:
        no_more_right=True

    return render_template('app.html', new_to_read=new_to_read,
                           not_first_new=True, current=current,
                           a=read_news_index_curr,
                           no_more_left=no_more_left,
                           no_more_right=no_more_right)


##################################################################
##################################################################

if __name__ == '__main__':
    app.run(debug=True)
