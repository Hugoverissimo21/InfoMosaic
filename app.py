from flask import Flask, render_template, request
import random
import matplotlib.pyplot as plt
import io
import base64
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
def update_ratings(index, user_rating, learning_rate):
    global ratings
    # Compute similarity to all other texts
    similarity_scores = cosine_similarity(tfidf_matrix[index], tfidf_matrix).flatten()
    learning_rate = 0.999**news_read
    ratings += (user_rating - ratings) * (similarity_scores)**0.5 * learning_rate
    ratings[index] = -1000
news_read = 0
current_new_index = 0

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
########################### HI LO ################################
##################################################################
##################################################################

@app.route('/hiloH_start', methods=['POST'])
def hiloH_start():
    global keywords
    global word1
    global word2
    global score
    word1 = random.choice(keywords)
    word2 = random.choice(keywords)
    while word2 == word1:
        word2 = random.choice(keywords)
    word1_mentions = int(keywords_data[word1]["count"])
    score = 0
    playable = True
    return render_template('app.html', word1=word1, word2=word2,
                           word1_mentions=word1_mentions, score=score,
                           playable=playable)

@app.route('/hiloH', methods=['POST'])
def hiloH():
    global keywords
    global word1
    global word2
    global score
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


    
##################################################################
##################################################################
########################### read ################################
##################################################################
##################################################################


@app.route('/get_news', methods=['POST'])
def get_news():
    global ratings
    global news_url
    global current_new_index
    # weight for the random choice
    weights = np.exp(np.array(ratings))
    weights = weights / weights.sum()
    # random news index
    current_new_index = np.random.choice(len(ratings), p=weights)
    new_to_read = news_url[current_new_index]

    return render_template('app.html', new_to_read=new_to_read,
                           not_first_new=True)

@app.route('/rate_news', methods=['POST'])
def rate_news():
    global current_new_index
    global ratings
    global news_read
    global news_url
    # process rate
    rate = int(request.form.get('rating4new'))
    update_ratings(current_new_index, rate, news_read)
    # give a new news
    weights = np.exp(np.array(ratings))
    weights = weights / weights.sum()
    # random news index
    current_new_index = np.random.choice(len(ratings), p=weights)
    new_to_read = news_url[current_new_index]

    return render_template('app.html', new_to_read=new_to_read,
                           not_first_new=True)


"""
weights = np.exp(np.array(ratings))  # Exponential gives higher prob to larger values
weights = weights / weights.sum()  # Normalize to prob

# Select an index based on the weights
index = np.random.choice(len(ratings), p=weights)
#print(news[index])
#rating = int(input(f"Rating of {index}: "))
rating = random.randint(1, 5)
rated_news.append(rating)
learning_rate = 0.999**_
update_ratings(index, rating, learning_rate)
"""

##################################################################
##################################################################

if __name__ == '__main__':
    app.run(debug=True)
