import xgboost as xgb
import streamlit as st
import pandas as pd
import base64

st.set_page_config(
    page_title="Diamond Price Forecaster", 
    page_icon="💎"
)

#Caching the model for faster loading
@st.cache_resource


# Define the prediction function
def predict(carat, cut, color, clarity, depth, table, x, y, z):
    #Predicting the price of the carat
    if cut == 'Fair':
        cut = 0
    elif cut == 'Good':
        cut = 1
    elif cut == 'Very Good':
        cut = 2
    elif cut == 'Premium':
        cut = 3
    elif cut == 'Ideal':
        cut = 4

    if color == 'J':
        color = 0
    elif color == 'I':
        color = 1
    elif color == 'H':
        color = 2
    elif color == 'G':
        color = 3
    elif color == 'F':
        color = 4
    elif color == 'E':
        color = 5
    elif color == 'D':
        color = 6
    
    if clarity == 'I1':
        clarity = 0
    elif clarity == 'SI2':
        clarity = 1
    elif clarity == 'SI1':
        clarity = 2
    elif clarity == 'VS2':
        clarity = 3
    elif clarity == 'VS1':
        clarity = 4
    elif clarity == 'VVS2':
        clarity = 5
    elif clarity == 'VVS1':
        clarity = 6
    elif clarity == 'IF':
        clarity = 7
    

    prediction = model.predict(pd.DataFrame([[carat, cut, color, clarity, depth, table, x, y, z]], columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']))
    return prediction

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )


tooltip_txt = {
'carat': "Carat is the standard unit of measurement used to indicate the weight of diamonds and precious gemstones. A small unit of measurement equals to 200 milligrams",
'cut': """
    Diamond Cut, more than any other quality aspect, gives a diamond its sparkle. A diamond gets its brilliance by cutting and polishing the diamond facets to allow the maximum amount of light to be dispersed back.

    Ideal - Premium - Very Good - Good - Fair

    Ideal is the highest cut grade.
    """,
'color': """
    Diamond Color is one of the most important factors to consider, as it is noticeable to the "naked" eye.

    Diamond color has a significant impact on its value. Although many diamonds appear to be colorless, many of them have at least a hint of body color.

    D - E - F - G - H - I - J
    
    D is the highest color grade. Diamonds are usually in the color range from D - J.
    """,
'clarity': """
    Most diamonds have unique clarity characteristics, much like a fingerprint. These distinguishing characteristics can be classified as inclusions and blemishes.

    If there is anything disrupting the flow of this in the diamonds, such as an inclusion, a proportion of the light reflected may be lost and can detract from the pure beauty of the diamond.

    IF - VVS1 - VVS2 - VS1 - VS2 - SI1 - SI2 - I1

    IF (Internally Flawless) is the highest clarity grade.
    """,
'depth': """
    Depth of the diamond is its height (in millimetres) measured from the culet (bottom tip) to the table (flat, top surface). 

    Depth = z / mean(x, y) = 2 * z / (x + y).
    """,
'table': """
    A diamond's table refers to the flat facet of the diamond seen when the stone is face up. The main purpose of a diamond table is to refract entering light rays and allow reflected light rays from within the diamond to meet the observer\'s eye. The ideal table cut diamond will give the diamond stunning fire and brilliance.
    """
}

def main():
    autoplay_audio("./media/audio/FKJ - Ylang Ylang_EfgAd6iHApE.mp3")
    st.snow()
    st.title('💎DIAMOND FORTUNE TELLER💎')
<<<<<<< HEAD
    st.image('./media/images/Thantai.jpeg', width=100)
=======
    st.image('./media/images/Thantai.jpeg', use_column_width=True)
>>>>>>> 1e25149868f8d9549dc6c3b128ea1f7f28c04b55
    st.header('Input the diamond\'s information in the fields below.')
    st.image('./media/images/dia_dim.png')
    carat = st.number_input('Carat Weight:', min_value=0.1, max_value=10.0, value=1.0, help=tooltip_txt['carat'])
    cut = st.select_slider('Cut Rating:', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], help=tooltip_txt['cut'])
    color = st.select_slider('Color Rating:', ['J', 'I', 'H', 'G', 'F', 'E', 'D'], help=tooltip_txt['color'])
    clarity = st.select_slider('Clarity Rating:', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], help=tooltip_txt['clarity'])
    depth = st.number_input('Diamond Depth Percentage:', min_value=0.1, max_value=100.0, value=1.0, help=tooltip_txt['depth'])
    table = st.number_input('Diamond Table Percentage:', min_value=0.1, max_value=100.0, value=1.0, help=tooltip_txt['table'])
    x = st.number_input('Diamond Length (X) in mm:', min_value=0.1, max_value=100.0, value=1.0)
    y = st.number_input('Diamond Width (Y) in mm:', min_value=0.1, max_value=100.0, value=1.0)
    z = st.number_input('Diamond Height (Z) in mm:', min_value=0.1, max_value=100.0, value=1.0)

    if st.button('Predict Price'):
        #Loading up the Regression model we created
        model = xgb.XGBRegressor()
        model.load_model("./model/non_timeseries/xgb_model.bin")

        price = predict(carat, cut, color, clarity, depth, table, x, y, z)
        st.success(f'Predicted diamond price: ${price[0]:.2f} USD')

if __name__ == '__main__':
    main()