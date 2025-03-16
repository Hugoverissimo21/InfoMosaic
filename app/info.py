from pyspark.sql import functions as F
import plotly.graph_objects as go
import plotly.io as pio

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


if __name__ == '__main__':
    print("abracadabra")