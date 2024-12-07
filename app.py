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
from collections import Counter
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like

##################################################################
##################################################################

# SEARCH | SEARCH | SEARCH | SEARCH | SEARCH
search_query = None
avaible_queries = sorted(["galp",
                   "bcp",
                   "edp",
                   "sonae",
                   "motgil"], key=lambda x: x.lower())

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
read_news_keys = []
read_news_dates = []
read_news_index_curr = -1

# WORDCLOUD | WORDCLOUD | WORDCLOUD | WORDCLOUD
keywords_data = None
wordcloud_pallete = None
wordcloud_data = None
def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    global wordcloud_pallete
    return random.choice(wordcloud_pallete)

##################################################################
##################################################################

# Intialize the Flask app
app = Flask(__name__)

@app.before_request
def before_request():
    global avaible_queries
    global search_query
    g.avaible_queries = avaible_queries
    g.search_query = search_query

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
            return render_template('app.html',
                                   search_query="Oops! Invalid topic. Try again or type * to explore valid options.",
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
    
    g.search_query = search_query
    return render_template('app.html', search_query=search_query,
                           search_finised=True)


@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    # Get the query from the URL (user's input)
    query = request.args.get('query', '').lower()
    if query == "*":
        return jsonify(g.avaible_queries)
    
    # Filter the recommendations based on user input
    filtered_recommendations = [item for item in g.avaible_queries if query in item.lower()]
    
    # Return the filtered suggestions as JSON
    return jsonify(filtered_recommendations)




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
    global read_news_keys
    global read_news_dates
    global news_data
    read_news_index = []
    read_news_keys = []
    read_news_dates = []
    # weight for the random choice
    weights = np.exp(np.array(ratings))
    weights = weights / weights.sum()
    # random news index
    current_new_index = np.random.choice(len(ratings), p=weights)
    new_to_read = news_url[current_new_index]
    read_news_index.append(current_new_index)
    top5words = Counter(news_data[new_to_read]["keywords"]).most_common(7)
    top5words = [f"{x[0]}" for x in top5words]
    top5words = str(top5words).strip("[]").replace("'", "")
    read_news_keys.append(top5words)
    newsdate = news_data[new_to_read]["tstamp"][:4]
    read_news_dates.append(newsdate)

    return render_template('app.html',
                           new_to_read=new_to_read,
                           not_first_new=True, current=True,
                           no_more_left=True, no_more_right=True,
                           top5words=top5words,
                           newsdate=newsdate)

@app.route('/rate_news', methods=['POST'])
def rate_news():
    global current_new_index
    global ratings
    global news_url
    global read_news_index
    global read_news_index_curr
    global read_news_keys
    global read_news_dates
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
    top5words = Counter(news_data[new_to_read]["keywords"]).most_common(7)
    top5words = [f"{x[0]}" for x in top5words]
    top5words = str(top5words).strip("[]").replace("'", "")
    read_news_keys.append(top5words)
    newsdate = news_data[new_to_read]["tstamp"][:4]
    read_news_dates.append(newsdate)

    read_news_index_curr = -1
    return render_template('app.html', new_to_read=new_to_read,
                           not_first_new=True, current=True,
                           no_more_left=False, no_more_right=True,
                           top5words=top5words,
                           newsdate=newsdate)

@app.route('/news_history', methods=['POST'])
def news_history():
    global read_news_index
    global news_url
    global current_new_index
    global read_news_index_curr
    global read_news_keys
    global read_news_dates
    no_more_left = False
    no_more_right = False

    move = request.form.get('news_hist')
    if move == "next":
        read_news_index_curr += 1
    elif move == "prev":
        read_news_index_curr -= 1
    
    new_to_read = news_url[read_news_index[read_news_index_curr]]
    top5words = read_news_keys[read_news_index_curr]
    newsdate = read_news_dates[read_news_index_curr]
    
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
                           no_more_right=no_more_right,
                           top5words=top5words,
                           newsdate=newsdate)



##################################################################
##################################################################
#$######################## wordcloud ############################
##################################################################
##################################################################

@app.route('/wordcloudgenerate', methods=['POST'])
def wordcloudgenerate():
    global keywords_data
    global wordcloud_pallete
    global wordcloud_data
    wordcloud_data = {word: data["weight"] for word, data in keywords_data.items()}
    wordcloud_pallete = ["black", "black", "black", "black"]

    wordcloud = WordCloud(width=800, height=400, background_color=None, mode='RGBA').generate_from_frequencies(wordcloud_data)

    plt.figure(figsize=(10, 5))
    random.seed(0)
    plt.imshow(wordcloud.recolor(color_func=color_func), interpolation='nearest')
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, dpi=300)
    buf.seek(0)
    plt.close()
    encoded_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return render_template('app.html', plot_data=encoded_plot,
                           wordcloud=True,
                           widthIN=800, heightIN=400,
                           col1="black", col2="black", col3="black", col4="black",
                           valid_colors=True)

@app.route('/wordcloudcolors', methods=['POST'])
def wordcloudcolors():
    global keywords_data
    global wordcloud_pallete
    global wordcloud_data

    input1 = request.form.get('input1')
    input2 = request.form.get('input2')
    input3 = request.form.get('input3')
    input4 = request.form.get('input4')

    widthIN = int(request.form.get('widthWC', 800))  
    heightIN = int(request.form.get('heightWC', 400))


    col1 = input1 if is_color_like(input1) else "black"
    col2 = input2 if is_color_like(input2) else "black"
    col3 = input3 if is_color_like(input3) else "black"
    col4 = input4 if is_color_like(input4) else "black"
    wordcloud_pallete = [col1, col2, col3, col4]

    if [input1, input2, input3, input4] == wordcloud_pallete:
        valid_colors = True
    else:
        valid_colors = False
    wordcloud = WordCloud(width=widthIN, height=heightIN, background_color=None, mode='RGBA').generate_from_frequencies(wordcloud_data)

    plt.figure(figsize=(10, 5))
    random.seed(0)
    plt.imshow(wordcloud.recolor(color_func=color_func), interpolation='nearest')
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, dpi=300)
    buf.seek(0)
    plt.close()
    encoded_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return render_template('app.html', plot_data=encoded_plot,
                           valid_colors=valid_colors,
                           wordcloud=True,
                           widthIN=widthIN,
                           heightIN=heightIN,
                           col1=col1, col2=col2,
                           col3=col3, col4=col4)



##################################################################
##################################################################

if __name__ == '__main__':
    app.run(debug=True)
