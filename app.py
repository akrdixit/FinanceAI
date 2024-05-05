import numpy as np

from datetime import datetime
import time
from utils import stocks_with_positive_and_negative_sentiment,generate
import streamlit as st
import pandas as pd
import pandas_ta as ta

import os
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import OpenAI

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
