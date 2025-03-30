import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# %%

def ts_topicrelation(news_by_month, keywords, search_topic, query):

    #traducao_meses = {
    #    "January": "Janeiro", "February": "Fevereiro", "March": "Março",
    #    "April": "Abril", "May": "Maio", "June": "Junho",
    #    "July": "Julho", "August": "Agosto", "September": "Setembro",
    #    "October": "Outubro", "November": "Novembro", "December": "Dezembro"
    #}

    # number of news per month
    #news_by_month = (
    #    df_with_query
    #    .groupBy('timestamp')
    #    .agg(F.count('archive').alias('count_of_news'))
    #    .toPandas()
    #)
    
    news_by_monthc = news_by_month.copy()
    news_by_monthc["timestamp"] = pd.to_datetime(news_by_monthc["timestamp"].astype(str), format='%Y%m')

    # number of mentions of the specific keyword
    specific_keyword = pd.DataFrame(list(keywords[search_topic]["date"].items()), columns=["date", "count_specific_keyword"])
    specific_keyword["date"] = pd.to_datetime(specific_keyword["date"], format="%Y%m")

    # merge the two dataframes
    news_history = news_by_monthc.merge(specific_keyword, left_on="timestamp", right_on="date", how="left")

    # create full data range
    min_date = news_history["timestamp"].min()
    max_date = news_history["timestamp"].max()
    full_range = pd.date_range(start=min_date, end=max_date, freq='MS')
    news_history = news_history.set_index("timestamp").reindex(full_range).fillna(0).reset_index()
    news_history = news_history.rename(columns={"index": "timestamp"})
    news_history = news_history.sort_values(by="timestamp")

    #news_history["data_formatada"] = news_history["timestamp"].dt.strftime("%B de %Y").replace(traducao_meses, regex=True)

    # create the plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=news_history["timestamp"],
        y=news_history["count_of_news"],
        mode="lines",
        name=f"Notícias sobre {query}",
        hovertemplate="%{y}"
    ))

    fig.add_trace(go.Scatter(
        x=news_history["timestamp"],
        y=news_history["count_specific_keyword"],
        mode="lines",
        name=f"Menções de {search_topic} em notícias sobre {query}",
        hovertemplate="%{y}"
    ))

    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Contagem",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=0, b=0, l=0, r=0),
        xaxis=dict(
            showgrid=False,
            zeroline=True,
            zerolinecolor="black",
            linecolor="black",
            linewidth=2
        ),
        yaxis=dict(
            range=[0, max(news_history["count_of_news"].max(), news_history["count_specific_keyword"].max()) * 1.1],
            showgrid=True,  
            gridcolor="lightgray",  
            zeroline=True,
            zerolinecolor="black",
            linecolor="black",
            linewidth=2
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,1)",
            bordercolor="black",
            borderwidth=1
        ),
    )

    fig.data[0].update(line=dict(color='rgba(101, 110, 242, 0.3)'))
    fig.data[1].update(line=dict(color='rgb(101, 110, 242)'))

    fig.update_xaxes(tickformat="%m/%Y")

    #fig.show(config={'displayModeBar': False})
    return pio.to_html(fig, full_html=False, config={'displayModeBar': False})

# %%

def sources_topicrelation(keywords, search_topic):

    sources = keywords[search_topic]['source']

    labels = list(sources.keys())
    values = list(sources.values())

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
        margin=dict(t=0, b=0, l=0, r=0)
    )

    return pio.to_html(fig, full_html=False, config={'displayModeBar': False})

# %%

if __name__ == "__main__":
    print("abracadabra")