from flask import Flask, render_template, request, Response
import random
import matplotlib.pyplot as plt
import io
import base64
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objs as go
import plotly.io as pio

##################################################################
##################################################################

# HILO | HILO | HILO | HILO | HILO | HILO
# Load the keywords from the JSON file for the HiLo game
with open('data/kwrd_bcp.json', 'r', encoding='utf-8') as json_file:
    keywords_data = json.load(json_file)
keywords = list(keywords_data.keys())
word1 = None
word2 = None
score = None
scores_historic = {}

# NEWS READER | NEWS READER | NEWS READER
# Load the news from the JSON file for the News Reader
with open('data/news_bcp.json', 'r', encoding='utf-8') as json_file:
    news_data = json.load(json_file)
news_url = list(news_data.keys())
setences = []
for new in news_url:
    setence = " ".join(news_data[new]["keywords"]) # set
    setences.append(setence)
ratings = np.array([3.0] * len(setences))
# Vectorize the setences
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(setences)
# Compute the cosine similarity matrix to update ratings
def update_ratings(index, user_rating):
    global ratings
    global number_news_read
    number_news_read += 1
    # Compute similarity to all other texts
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

##################################################################
##################################################################
########################### XXXXX ################################
##################################################################
##################################################################

@app.route('/')
def home():
    return render_template('app.html')

# Route for graph tab
@app.route('/graph', methods=['POST'])
def graph():
    data = request.form.get('data', '0')
    try:
        # Here you can use any graphing library, e.g., matplotlib
        x = [i for i in range(10)]
        y = [int(data) * i for i in x]
        
        # Create a plot
        plt.figure(figsize=(5, 4))
        plt.plot(x, y)
        plt.title('Graph based on input data')
        
        # Save it to a byte stream and encode to base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_data = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()
        
        return render_template('app.html', graph_data=plot_data)
    except ValueError:
        return render_template('app.html', error="Invalid input for graph!")

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
    playable = True
    try:
        scores_historic[max(scores_historic.keys()) + 1] = {"score": 0,
                                                            "text": "Current game! GL"}
    except:
        scores_historic[1] = {"score": 0,
                              "text": "Current game! GL"}
    return render_template('app.html', word1=word1, word2=word2,
                           word1_mentions=word1_mentions, score=score,
                           playable=playable)

@app.route('/hiloH', methods=['POST'])
def hiloH():
    global keywords
    global word1
    global word2
    global score
    global scores_historic
    playable = True
    word1_mentions = int(keywords_data[word1]["count"])
    word2_mentions = int(keywords_data[word2]["count"])

    if request.method == 'POST':
        user_choice = request.form.get('choice')  # Get the user's choice
        scenarios = {
            "more": word2_mentions >= word1_mentions,
            "less": word2_mentions <= word1_mentions
        }
        if scenarios[user_choice]:
            winner = "Correct!"
            loser = None
            word1 = word2
            word1_mentions = word2_mentions
            score += 1
            word2 = random.choice(keywords)
            while word2 == word1:
                word2 = random.choice(keywords)
            word2_mentions_ans = None
            score_out = score
        else:
            scores_historic[max(scores_historic.keys())]["score"] = score
            scores_historic[max(scores_historic.keys())]["text"] = f"""
            Score: {score}
            <br>Word 1: {word1} ({word1_mentions})
            <br>Word 2: {word2} ({word2_mentions})"""
            loser = f"Incorrect! Final score: {score}"
            winner = None
            playable = None
            score = 0
            score_out = None
            word2_mentions_ans = int(keywords_data[word2]["count"])
    
    return render_template('app.html', word1=word1, word2=word2,
                           word1_mentions=word1_mentions, score=score_out,
                           winner=winner, loser=loser,
                           word2_mentions_ans=word2_mentions_ans,
                           playable = playable)

@app.route('/hiloH_restart', methods=['POST'])
def hiloH_restart():
    return render_template('app.html', word1=None, word2=None,
                           word1_mentions=None, score=0,
                           winner=None, loser=None,
                           word2_mentions_ans=None,
                           playable = None)

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
                      yaxis_title='Scores Historic')

    # Use io.StringIO to save the HTML content in memory
    buffer = io.StringIO()
    pio.write_html(fig, file=buffer, full_html=True)
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
