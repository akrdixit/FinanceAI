import pandas as pd
import numpy as np
import yfinance as yf
import plotly.subplots as sp
import plotly.graph_objects as go
import pandas_ta as ta

def stocks_with_positive_and_negative_sentiment():
    stock_list = ['ADANIENT', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO',
                  'BAJFINANCE', 'BAJAJFINSV', 'BPCL', 'BHARTIARTL', 'BRITANNIA',
                  'CIPLA', 'COALINDIA', 'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK',
                  'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK',
                  'ITC', 'INDUSINDBK', 'INFY', 'JSWSTEEL', 'KOTAKBANK', 'LTIM',
                  'MARUTI', 'NTPC', 'NESTLEIND', 'POWERGRID', 'RELIANCE', 'SBILIFE',
                  'SHRIRAMFIN', 'SBIN', 'SUNPHARMA', 'TCS', 'TATACONSUM',
                  'TATAMOTORS', 'TATASTEEL']
    list_dfs = []
    try:
        for symbol in stock_list:
            symbol = symbol + '.NS'
            start_date = '2022-05-01'
            end_date = '2024-04-26'
            stock_data = yf.download(symbol, start=start_date, end=end_date)
            stock_data.ta.supertrend(append=True, atr_length=7, multiplier=3)
            stock_data['SUPERT_7_3.0'].fillna(method='ffill', inplace=True)
            stock_data['symbol'] = symbol.replace(".NS", "")
            stock_data['super_trend'] = stock_data.apply(lambda x: "buy" if x['Close'] > x['SUPERT_7_3.0'] else "sell",
                                                         axis=1)
            stock_data['rsi'] = stock_data['super_trend']
            list_dfs.append(stock_data[-1:].copy())
    except:
        pass
    signals_tech = pd.concat(list_dfs)
    signals_tech['Date'] = signals_tech.index
    signals_tech['Date'] = pd.to_datetime(signals_tech['Date']).dt.date
    df_stocks = pd.read_csv('data/sentiment_1week_nifty_50_v2.csv')
    df_stocks['start_date'] = pd.to_datetime(df_stocks['start_date']).dt.date
    df_stocks.rename(columns={"start_date": "Date"}, inplace=True)
    df_stocks['sentiment_score'] = df_stocks.apply(
        lambda x: x['sentiment_score'] if x['sentiment_type'] == "POSITIVE" else -x['sentiment_score'], axis=1)
    df_results = df_stocks[df_stocks['Date'] == signals_tech['Date'].max()]
    stocks_positive = df_results.loc[df_results.groupby('Symbol')['sentiment_score'].idxmax()]
    stocks_positive = stocks_positive.sort_values('sentiment_score', ascending=False)[0:5]

    stocks_negative = df_results.loc[df_results.groupby('Symbol')['sentiment_score'].idxmin()]
    stocks_negative = stocks_negative.sort_values('sentiment_score')[0:4]

    return (stocks_positive, stocks_negative)



def generate(stock_data):
    # Create subplots in a separate plane with different row heights
    fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True,
                           subplot_titles=[f'Nifty 50 Index Prices', 'Technical and Sentiment Indicators'],
                           vertical_spacing=0.1,
                           row_heights=[0.7, 0.15, 0.15])

    # Add Stock Prices subplot
    fig.add_trace(go.Candlestick(x=stock_data.index,
                                 open=stock_data['Open'],
                                 high=stock_data['High'],
                                 low=stock_data['Low'],
                                 close=stock_data['Close'],
                                 name='Stock Prices'), row=1, col=1)

    # Add Supertrend subplot with dynamic colors and smaller marker dots
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SUPERT_7_3.0'],
                             mode='lines+markers',
                             marker=dict(
                                 color=np.where(stock_data['Close'] > stock_data['SUPERT_7_3.0'], 'green', 'red'),
                                 size=3),  # Set the size of the marker dots
                             line=dict(color='rgba(0,0,0,0)'),  # This line is added to hide the line connecting markers
                             name='Supertrend'), row=1, col=1)

    # Add ATR subplot
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Sentiment'],
                             mode='lines',
                             line=dict(color='blue'),  # Set the color to blue
                             name='Sentiment_score'), row=2, col=1)

    # Add RSI subplot
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI_14'],
                             mode='lines',
                             name='RSI'), row=3, col=1)

    # Update layout to set figure size
    fig.update_layout(xaxis_rangeslider_visible=False,
                      template='plotly_dark',
                      title_text='Nifty Index Stock Analysis',
                      xaxis=dict(type='category'),  # Set x-axis type to category which removes the sat and sun gap
                      yaxis_title_text='Price',
                      yaxis2_title_text='Sentiment_score',  # Add y-axis title for ATR subplot
                      yaxis3_title_text='RSI',  # Add y-axis title for RSI subplot
                      height=600,  # Set the height of the figure
                      width=1000)  # Set the width of the figure
    return fig
