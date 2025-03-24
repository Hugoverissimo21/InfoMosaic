from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import collect_list
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO

#%%

def pie_newsSources(df_with_query):
    # Group by the column and count the values
    value_counts_df = df_with_query.groupBy('source').count().toPandas()

    # Extract labels and values directly
    labels = value_counts_df['source']
    values = value_counts_df['count']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hoverinfo='label+value+percent',
        hovertemplate="<b>%{label}</b><br>Notícias: %{value}<br>Percentagem: %{percent:.2%}<extra></extra>"
    )])

    fig.update_traces(
        textposition='inside',
        textinfo='label',
        textfont_size=12
    )

    fig.update_layout(
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=25, b=25, l=0, r=0)
    )

    return pio.to_html(fig, full_html=False, config={'displayModeBar': False})

#%%

def timeseries_news(df_with_query, query):
    traducao_meses = {
        "January": "Janeiro", "February": "Fevereiro", "March": "Março",
        "April": "Abril", "May": "Maio", "June": "Junho",
        "July": "Julho", "August": "Agosto", "September": "Setembro",
        "October": "Outubro", "November": "Novembro", "December": "Dezembro"
    }

    news_by_month = (
        df_with_query
        .groupBy('timestamp')
        .agg(F.count('archive').alias('count_of_news'))
        .toPandas()
    )

    keywords_by_month = (
        df_with_query
        .select('*', F.explode('keywords'))
        .groupBy("timestamp", "key")
        .agg(F.sum("value").alias("key_mentions"))
        .filter(F.col("key") != query)
        .withColumn("rank", F.row_number().over(Window.partitionBy("timestamp").orderBy(F.desc("key_mentions"))))
        .filter(F.col("rank") <= 5)
        .groupBy("timestamp")
        .agg(collect_list("key").alias("top5_keywords"))
        .toPandas()
    )

    news_history = news_by_month.merge(keywords_by_month, on="timestamp", how="inner")
    news_history["timestamp"] = pd.to_datetime(news_history["timestamp"].astype(str), format='%Y%m')

    min_date = news_history["timestamp"].min()
    max_date = news_history["timestamp"].max()
    full_range = pd.date_range(start=min_date, end=max_date, freq='MS')

    news_history = news_history.set_index("timestamp").reindex(full_range).fillna(0).reset_index()
    news_history = news_history.rename(columns={"index": "timestamp"})
    news_history = news_history.sort_values(by="timestamp")

    news_history["data_formatada"] = news_history["timestamp"].dt.strftime("%B de %Y").replace(traducao_meses, regex=True)
    news_history["top5_keywords"] = news_history["top5_keywords"].apply(
        lambda words: "-" if words == 0 else "<br>".join([f"{i+1}. {word}" for i, word in enumerate(words)])
    )

    fig = px.line(
        news_history,
        x="timestamp",
        y="count_of_news",
        line_shape="linear",
        custom_data=news_history[["data_formatada", "top5_keywords"]],
    )

    fig.update_traces(
        hovertemplate="<b>Data:</b> %{customdata[0]}<br>"
                    "<b>Notícias:</b> %{y}<br>"
                    "<b>Top 5 Tópicos:</b><br>%{customdata[1]}"
    )

    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Quantidade de Notícias",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="pink",
        margin=dict(t=0, b=0, l=0, r=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True)
    )

    fig.update_traces(line=dict(color='rgb(255, 255, 0)'),
                    hoverlabel=dict(bgcolor='rgb(0, 255, 0)',
                                    font=dict(color='black')))


    fig.update_xaxes(tickformat="%m/%Y")

    return pio.to_html(fig, full_html=False, config={'displayModeBar': False})

#%%

def topic_wordcloud(word_counts, query, FONT_PATH):
    # Constants
    MAX_SIZE = 500  # Maximum font size
    IMAGE_SIZE = (1920, 1080)  # Image dimensions

    # Load font
    font = ImageFont.truetype(FONT_PATH, size=MAX_SIZE)

    def get_optimal_font_size(text, font, max_size, image_size):
        """Determine the best font size to fit within the image."""
        temp_img = Image.new("RGBA", image_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)

        # Start at max size and reduce iteratively (Binary search can also be used)
        for size in range(max_size, 5, -5):  # Step down in increments for efficiency
            font = ImageFont.truetype(FONT_PATH, size=size)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

            if text_width <= image_size[0] and text_height <= image_size[1]:
                return font, bbox  # Return the optimal font and bounding box

        return font, bbox  # Return the smallest size if no fit

    # Get optimal font size
    font, bbox = get_optimal_font_size(query, font, MAX_SIZE, IMAGE_SIZE)

    # Create text image
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    text_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_img)
    draw.text((-bbox[0], -bbox[1]), query, fill=(255, 105, 180), font=font)  # Pink text

    # Generate mask for WordCloud (grayscale to use as mask)
    mask = np.array(text_img.convert("L"))

    wc = WordCloud(
        width=text_width, height=text_height,
        background_color=None, min_font_size=5, mode="RGBA",
        colormap="winter", mask=~mask, contour_color="black"
    ).generate_from_frequencies(word_counts)

    wc_image = wc.to_image()
    final_img = Image.alpha_composite(text_img, wc_image)

    def pil_to_base64(img):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    base64_img = pil_to_base64(final_img)

    return base64_img

# %%

if __name__ == '__main__':
    print("abracadabra")