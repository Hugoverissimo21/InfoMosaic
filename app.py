from flask import Flask, render_template, request
import random
import matplotlib.pyplot as plt
import io
import base64
import json

# Load the keywords from the JSON file for the HiLo game
with open('data/kwrd_bcp.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
keywords = list(data.keys())
word1 = None
word2 = None
score = None

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
    word1_mentions = int(data[word1]["count"])
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
    word1_mentions = int(data[word1]["count"])
    word2_mentions = int(data[word2]["count"])

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
            word2_mentions_ans = int(data[word2]["count"])
    
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
########################### XXXXX ################################
##################################################################
##################################################################


@app.route('/read', methods=['POST'])
def read():
    input_data = request.form.get('input_text')
    return render_template('app.html', output=input_data)

if __name__ == '__main__':
    app.run(debug=True)
