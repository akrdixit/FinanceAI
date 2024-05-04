from stockstats import StockDataFrame
import plotly.graph_objects as go


def generate_candlestick_chart(df):
    candlestick = go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Candlesticks')

    layout = go.Layout(title='Candlestick Chart',
                       xaxis=dict(title='Date'),
                       yaxis=dict(title='Price'))

    return go.Figure(data=[candlestick], layout=layout)


def generate_rsi_chart(df):
    stock = StockDataFrame.retype(df)
    rsi = stock['rsi_14']
    rsi_trace = go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='blue'))

    layout = go.Layout(title='Relative Strength Index (RSI)',
                       xaxis=dict(title='Date'),
                       yaxis=dict(title='RSI'))

    return go.Figure(data=[rsi_trace], layout=layout)