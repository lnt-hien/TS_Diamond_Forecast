import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

from darts import TimeSeries
from darts.models import NBEATSModel
from darts.models import NHiTSModel
from darts.models import NLinearModel
from darts.models import DLinearModel

from darts.utils.model_selection import train_test_split
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.metrics import mape, r2_score
from darts.metrics import mae
from darts.datasets import EnergyDataset

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Diamond Price Forecaster", page_icon="💎"
)

def main():
    st.sidebar.header('Diamond Price Forecaster')
    st.sidebar.info("Make sure your data is a time series data \
                     containing two columns date & price.")
    option = st.sidebar.selectbox('How do you want to get the data?', ['File', 'URL'])
    if option == 'URL':
        url = st.sidebar.text_input('Enter a URL')
        if url:
            dataframe(url)
    else:
        file = st.sidebar.file_uploader('Choose a file', type=['csv', 'txt'])
        if file:
            dataframe(file)



def dataframe(file):
    st.header('Diamond Price Forecaster')

    series, df = process_data(file)

    chart = plot_chart(df)
    st.altair_chart((chart).interactive(), use_container_width=True)
    
    period = st.number_input('Enter the next period(s) you want to forecast', value=7, help="Keep in mind that forecasts become less accurate with larger forecast horizons.")
    button = st.button('Forecast')
    if button:
        # pred_df = model_forecast(series, period)
        pred_df = model_forecast(series, period, weights="./model/dlinear.pt")

        df['date'] = pd.to_datetime(df['date'])
        df.set_index(df.columns[0], inplace=True)

        merged_df = pd.merge(df, pred_df, on='date', how='left')
        merged_df = merged_df[['price', 'pred_price']].reset_index()
        merged_df.rename({'price': 'actual', 'pred_price': 'pred'}, axis=1, inplace=True)

        st.text('The line chart below shows how well the model fits into your data.')
        merged_chart = plot_chart(merged_df.melt('date', var_name='y', value_name='price').dropna(), color='y')
        st.altair_chart(merged_chart.interactive(), use_container_width=True)


# We use @st.cache_data to keep the dataset in cache
@st.cache_data
def process_data(file):
    df = pd.read_csv(file, index_col=[0])[::-1]
    df.reset_index(inplace=True, drop=True)
    df['date'] = pd.to_datetime(df['date'])
    df.rename({'diamond price': 'price'}, axis=1, inplace=True)
    df.set_index(df.columns[0], inplace=True)

    df_day_avg = df.groupby(df.index.astype(str).str.split(" ").str[0]).mean().reset_index()
    
    # Fill missing value
    filler = MissingValuesFiller()
    series = filler.transform(
        TimeSeries.from_dataframe(
            df_day_avg, df_day_avg.columns[0], df_day_avg.columns[1]
        )
    ).astype(np.float32)

    return series, df_day_avg

def plot_chart(data, color=None):
    hover = alt.selection_single(
        fields=["date"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lower = min(data.price)
    upper = max(data.price)
    if color:
        line = (
            alt.Chart(data, title="Diamond prices")
            .mark_line()
            .encode(
                x=alt.X("yearmonthdate(date):T", title="Date"),
                y=alt.Y("price:Q", title="Price", scale=alt.Scale(domain=[lower, upper])),
                color=alt.Color(
                    color,
                    # scale=alt.Scale(
                    #     domain=['actual', 'predicted'],
                    #     range=["#F20505", "#63CAF2"]
                    # )
                ),
            )
        )
    else:
        line = (
            alt.Chart(data, title="Diamond prices")
            .mark_line()
            .encode(
                x=alt.X("yearmonthdate(date):T", title="Date"),
                y=alt.Y("price:Q", title="Price", scale=alt.Scale(domain=[lower, upper])),
            )
        )

    # Draw points on the line, and highlight based on selection
    points = line.transform_filter(hover).mark_circle(size=65)

    # Draw a rule  at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="yearmonthdate(date)",
            y="price",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("yearmonthdate(date)", title="Date"),
                alt.Tooltip("price", title="Price (USD)"),
            ],
        )
        .add_selection(hover)
    )

    return (line + points + tooltips).interactive()


def model_forecast(series, period, weights=None):
    # Split train, val
    train, val = train_test_split(series, test_size=0.2)

    if weights:
        print('Loading weights...')
        model = DLinearModel.load(weights)
    else:
        print('Loading model...')
        model = DLinearModel(
            input_chunk_length=30, 
            output_chunk_length=period,
            n_epochs=100,
            nr_epochs_val_period=1,
            batch_size=500,
        )
        
        model.to_cpu()
        
        model.fit(series=train, val_series=val, verbose=False)
        model.save('./model/dlinear.pt')

    dlinear_pred_series = model.historical_forecasts(
        series,
        start=series.time_index[0],
        forecast_horizon=period,
        stride=5,
        retrain=False,
        verbose=False,
    )
    
    # display_forecast(dlinear_pred_series, series, period, start_date=val[0].time_index[0])
    # display_forecast(dlinear_pred_series, series, period)
    pred_df = dlinear_pred_series.pd_dataframe()
    pred_df.reset_index(inplace=True)
    pred_df['time'] = pd.to_datetime(pred_df['time'])
    pred_df.rename({'time': 'date', 'price': 'pred_price'}, axis=1, inplace=True)
    pred_df.set_index(pred_df.columns[0], inplace=True)
    
    # predict
    st.text(f"Prediction for {period} period(s) onward:")
    future_pred_df = model.predict(n=period).pd_dataframe()
    for i, j in enumerate(future_pred_df.to_numpy()):
        st.text(f"Period {i+1}: {j[0]:.2f}")
    
    return pred_df


def display_forecast(pred_series, series, periods, start_date=None):
    fig = plt.figure(figsize=(8, 5))

    if start_date:
        series = series.drop_before(start_date)

    series.univariate_component(0).plot(label="actual")
    pred_series.plot(label=("historic " + str(periods) + " days forecasts"))

    plt.title(f"R2: {r2_score(series.univariate_component(0), pred_series)}")
    plt.legend()
    st.pyplot(fig)

if __name__ == '__main__':
    main()