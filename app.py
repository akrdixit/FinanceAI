import numpy as np

from datetime import datetime
import time
import streamlit as st
import pandas as pd
import pandas_ta as ta

import os
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import OpenAI
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
    
OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

model = OpenAI()
df_read = pd.read_csv('data/sentiment_1week_nifty_50_v2.csv')
df_read.rename(columns={'start_date': "date"}, inplace=True)
#del df_read['end_date']
agent = create_pandas_dataframe_agent(model, df_read, verbose=False)

sentiment = pd.read_csv("data/sentiment_1week_nifty_50_v2.csv")
sentiment["RSA"] = np.random.randint(0, 101, size=(len(sentiment), 1))
sentiment['adjusted_score'] = sentiment[['sentiment_type', 'sentiment_score']].apply(
    lambda x: -1 * x[1] if x[0] == 'NEGATIVE' else x[1], axis=1)
sentiment['normalised_sentiment_score'] = (sentiment.adjusted_score - sentiment.adjusted_score.min()) / (
            sentiment.adjusted_score.max() - sentiment.adjusted_score.min())
positive_sentiment = sentiment[sentiment["sentiment_type"] == "POSITIVE"]
positive_sentiment["super_trend"] = ""
negative_sentiment = sentiment[sentiment["sentiment_type"] == "NEGATIVE"]
negative_sentiment["super_trend"] = ""

st.sidebar.title('DataHack App')

mapping = {'dashboard': '📈 Dashboard',
           'chat': '💬 Chat'}

col = st.columns((6, 6), gap='large')

selected_tab = None
selected_tab = st.sidebar.radio(label='Go to', options=("dashboard", "chat"), format_func=lambda x: mapping[x],
                                    label_visibility='hidden')

st.markdown("""
    <style>
    [role=radiogroup]{
        gap: 1rem;
    }
    </style>
    """,unsafe_allow_html=True)

if selected_tab == 'dashboard':
    with st.container():
        with col[0]:
            start_date = st.date_input('Start date', datetime(2024, 2, 1))
        with col[1]:
            end_date = st.date_input('End date', datetime(2024, 2, 29))

        # stock = st.sidebar.text_input('Enter a stock name', 'AAPL')

        start_timestamp = str(int(time.mktime(start_date.timetuple())))
        end_timestamp = str(int(time.mktime(end_date.timetuple())))

        original_url = "https://query1.finance.yahoo.com/v7/finance/download/%5ENSEI?period1=" + start_timestamp + "&period2=" + end_timestamp + "&interval=1d&events=history&includeAdjustedClose=true"
        df_nifty_data = pd.read_csv(original_url)
        df_nifty_data.columns = [cl for cl in df_nifty_data.columns]
        stock_data = df_nifty_data.copy()
        df_agg_sentiment = pd.read_csv('data/agrregated_sentiment.csv')
        df_agg_sentiment.rename(columns={"start_date": "Date", "mean": "Sentiment"}, inplace=True)
        df_agg_sentiment = df_agg_sentiment[['Date', 'Sentiment']]

        # Calculate technical indicators
        # stock_data.ta.atr(length=14, append=True)
        stock_data.ta.rsi(length=14, append=True)
        stock_data.ta.supertrend(append=True, atr_length=7, multiplier=3)
        stock_data['SUPERT_7_3.0'].fillna(method='ffill', inplace=True)
        stock_data = stock_data[20:]
        stock_data['signal_st'] = stock_data.apply(lambda x: "buy" if x['Close'] > x['SUPERT_7_3.0'] else "sell",
                                                   axis=1)
        stock_data = pd.merge(stock_data, df_agg_sentiment, on='Date', how='left')
        stock_data = stock_data[stock_data['Date'] <= stock_data[~stock_data['Sentiment'].isnull()]['Date'].max()]
        st.plotly_chart(generate(stock_data))

    with st.container():
        # positive sentiment table
        st.subheader('Positive Sentiment Table')

        positive_sentiment_data = \
        positive_sentiment.sort_values(by=["normalised_sentiment_score"], ascending=False).drop_duplicates(
            subset=["company", "Symbol", "sentiment_type"], keep="first")[
            ["company", "Symbol", "sentiment_type", "normalised_sentiment_score", "RSA", "super_trend"]].head(5)

        st.table(positive_sentiment_data)

        # negative sentiment table
        st.subheader('Negative Sentiment Table')
        negative_sentiment_data = \
        negative_sentiment.sort_values(by=["normalised_sentiment_score"], ascending=True).drop_duplicates(
            subset=["company", "Symbol", "sentiment_type"], keep="first")[
            ["company", "Symbol", "sentiment_type", "normalised_sentiment_score", "RSA", "super_trend"]].head(5)

        st.table(negative_sentiment_data)

if selected_tab == 'chat':
    st.markdown(
        """
        <style>
        body {
            background-color: #2b2b2b; /* Dark gray or any other dark color you prefer */
            color: white; /* Text color */
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    def get_response(prompt):
        # import requests

        # url = "http://127.0.0.1:8000/memory_buffer/"
        # data = {"message": prompt}
        # response = requests.post(url, json=data)
        # #print(response.json()['message']['response'])
        # return response.json()['message']['response']

        try:
            result = (agent.invoke(prompt))

            return result['output']
        except:
            return "Could you ask pharase the questions differently ?"


    def main():
        print("starting")
        # st.title("WealthWizard Bot:Wbot")
        st.title("🚀 Welcome to WealthWizard Bot! 📈")

        st.write("Unlock the Power of AI for Stock Success!")
        # Instructions
        st.markdown("## How to Use:")
        st.markdown("Ask a question about Indian stocks, such as:")
        st.code("- 'What is the current sentiment on Reliance Industries?'")
        st.code("- 'Show me the technical analysis for Tata Consultancy Services.'")
        st.code("- 'What are the fundamental metrics for HDFC Bank?'")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("What is up?"):
            # '''
            # code to write the response logic
            # '''
            # print(prompt)
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                response = f"w_bot: {get_response(prompt)}"
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})



    if __name__ == "__main__":
        main()
