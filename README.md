# Media Analysis of Five PSI-20 Companies !!!!

The primary goal of this project is to analyze media coverage of companies listed on the PSI-20 index. By leveraging various data analysis, machine learning, and visualization techniques, the project aims to extract valuable insights regarding the public perception and sentiment around these companies.

The specific objectives and features of the project are:

1. **Sentiment Analysis**  !!!!
   Perform sentiment analysis on news articles, social media mentions, and other media sources to assess the public sentiment surrounding PSI-20 companies. This will help determine whether the sentiment is positive, negative, or neutral.

2. **Named Entity Recognition (NER)**   !!!!
   Extract and identify key entities (such as company names, individuals, locations, etc.) from media coverage to understand the context in which PSI-20 companies are mentioned. This will aid in understanding the nature of discussions and their relevance.

3. **Media Coverage Analysis**  
   Track the frequency and volume of media mentions for each PSI-20 company, highlighting trends over time and uncovering patterns that can indicate shifts in public perception or market impact.

4. **Data Collection and Analysis**  
   Extract data from [arquivo.pt](https://arquivo.pt/) focusing on PSI-20 companies and perform data exploration and analysis using Jupyter notebooks.

5. **Data Visualization**  
   Create static and interactive visualizations to present insights such as sentiment trends, media mentions over time, and relationships between companies and key entities. These visualizations will help to make complex data more accessible and understandable.

6. **Web Application**  
   Build a web app using `Flask` to showcase key insights and visualizations interactively. The app will allow users to explore the data, view sentiment trends, and gain insights into media coverage in an intuitive manner.

By integrating data collection, analysis, machine learning, and visualization techniques, this project seeks to transform media coverage into actionable insights that can inform decision-making for investors, analysts, and stakeholders.

## **Deployed Version**

Visit the live app: [InfoMosaic Sandbox](https://hugover.pythonanywhere.com).

- [Slides 01](https://hugoverissimo21.github.io/InfoMosaic-sandbox/slides01)

- [Slides 02](https://hugoverissimo21.github.io/InfoMosaic-sandbox/slides02)

- [Slides 03](https://hugoverissimo21.github.io/InfoMosaic-sandbox/slides03)

## **Project Structure**

- **`assets/`**: Visualizations and images generated throughout the project.

- **`data/`**: Extracted data from [arquivo.pt](https://arquivo.pt/) and processed datasets for analysis.

- **`notebooks/`**: Jupyter notebooks for data analysis, model training, and experiments.

- **`templates/` & `static/`**: Front-end assets for the Flask application.

- **`tests/`**: Jupyter notebooks for validating models, testing methods, and generating visualizations.

## **Technologies**

- **Jupyter Notebooks**: Data extraction, curation, and analysis.

- **Python**: Backend logic and data processing.

- **Flask**: Web framework for deployment.

- **HTML**: Front-end visualizations.

## **Getting Started**

1. Clone the repository:

```bash
git clone https://github.com/Hugoverissimo21/InfoMosaic-sandbox.git
```

2. Install dependencies:

```bash
pip install -r zextra/requirements03.txt
```

3. Run the Flask app:

```bash
python app.py
```
