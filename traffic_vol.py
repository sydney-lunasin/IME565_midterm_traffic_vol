# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
import datetime
warnings.filterwarnings('ignore')

st.markdown(
    """
    <h1 style="
        text-align: center;
        background: linear-gradient(to right, red, yellow, green);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    ">
        Traffic Volume Predictor
    </h1>
    """,
    unsafe_allow_html=True
)
# Used chat to get syntax
st.markdown("<h3 style='text-align: center;'>Utilize our advanced Machine Learning application to predict traffic volume.</h3>", unsafe_allow_html=True)

# Display the image
st.image('traffic_image.gif', width = 700)

# Load the pre-trained model from the pickle file
traffic_pickle = open('traffic_vol.pickle', 'rb') 
mapie = pickle.load(traffic_pickle) 
traffic_pickle.close()

# Create a sidebar for input collection
st.sidebar.image('traffic_sidebar.jpg', width = 300)
st.sidebar.header('Input Features')
st.sidebar.write('You can either upload your data file or manually enter input features.')

if 'manual_form_submitted' not in st.session_state:
    st.session_state.manual_form_submitted = False

# Default Data for automation
cols_to_drop = ['traffic_volume', 'date_time']

default_data = (
    pd.read_csv('Traffic_Volume (1).csv')
      .assign(date_time=lambda x: pd.to_datetime(x['date_time'], format='%m/%d/%y %H:%M'))
      .assign(
          month=lambda x: x['date_time'].dt.strftime('%B'),
          weekday=lambda x: x['date_time'].dt.strftime('%A'),
          hour=lambda x: x['date_time'].dt.hour,
          holiday=lambda x: x['holiday'].fillna('None')
      )
      .drop(columns=cols_to_drop)
      .reset_index(drop=True)
)

#used chat to simplify and make more efficient


# Option 1: Asking users to input their data as a file
option1_expander = st.sidebar.expander('Option 1: Upload CSV file')
option1_expander.write('Upload a CSV file')
user_data = option1_expander.file_uploader('Choose a CSV file')
option1_expander.subheader('Sample Data Format for Upload')
option1_expander.dataframe(default_data.head())
option1_expander.warning("⚠️ Ensure your uploaded file has the same column names and data types as shown above.")

# Option 2: Fill out form
option2_expander = st.sidebar.expander('Option 2: Fill out the form')
option2_expander.write('Enter the traffic details manually using the form below.')

# Create form for users
with option2_expander.form("manual_form"):
    holiday = st.selectbox('Choose whether today is designated holiday or not', options=default_data['holiday'].unique(), key="holiday_manual")
    temp = st.number_input('Average temperature in Kelvin', 
                                         min_value = float(default_data['temp'].min()), 
                                         max_value = float(default_data['temp'].max()),
                                         step = 0.01,
                                         value = float(default_data['temp'].mean()),
                                         key= "temp_manual")
    rain_1h = st.number_input('Amount in mm of rain that ocurred in the hour', 
                                             min_value = float(default_data['rain_1h'].min()), 
                                             max_value = float(default_data['rain_1h'].max()),
                                             step = 0.01,
                                             value = float(default_data['rain_1h'].mean()),
                                             key = "rain_manual")
    snow_1h = st.number_input('Amount in mm of snow that occurred in the hour', 
                                             min_value = float(default_data['snow_1h'].min()), 
                                             #max_value = float(default_data['snow_1h'].max()),
                                             step = 0.01,
                                             value = float(default_data['snow_1h'].mean()),
                                             key="snow_manual")
    clouds_all = st.number_input('Percentage of cloud cover',
                                               min_value = float(default_data['clouds_all'].min()), 
                                               max_value = float(default_data['clouds_all'].max()),
                                               step = 1.0,
                                               value = float(default_data['clouds_all'].mean()),
                                               key="clouds_manual")
    weather_main = st.selectbox('Choose the current weather', options=default_data['weather_main'].unique(), key="weather_manual")
    month = st.selectbox('Choose month', options=['January', 'February', 'March', 'April', 'May', 'June', 'August', 'September', 'October', 'Novemeber', 'December'], key="month_manual")
    weekday = st.selectbox('Choose day of the week', options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], key="day_manual")
    hour = st.selectbox('Choose hour', options=[0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], key="hour_manual")

    # Submit button
    submitted = st.form_submit_button("Submit Form Data")

default_data = default_data.dropna()

# persist the click into session_state
if submitted:
    st.session_state.manual_form_submitted = True

# Create placeholders for status messages
status_placeholder = st.empty()

# Determine the current app state
if user_data is not None:
    status_placeholder.success("✅ CSV file uploaded successfully.")
elif st.session_state.manual_form_submitted:
    status_placeholder.success("✅ Form data submitted successfully.")
else:
    status_placeholder.info("ℹ️ Please choose a data input method to proceed.")
#used chat to create status messages

# Add slider to set alpha value for prediction confidence interval
alpha = st.slider('Select Alpha for Prediction Interval (Confidence Level)', min_value = 0.01, max_value = 0.90, value = 0.32, step = 0.01)

# If no file is provided, allow user to provide inputs using form
if user_data is None:
    encode_df = default_data.copy()
    #encode_df = encode_df.drop(columns = ['traffic_volume'])

    encode_df.loc[len(encode_df)] = [holiday, temp, rain_1h, snow_1h, clouds_all, weather_main,  month, weekday, hour]

    # One-Hot Encoding for Categorical Features
    encode_dummy_df = pd.get_dummies(encode_df)

    user_encoded_df = encode_dummy_df.tail(1)

    y_pred, y_pis = mapie.predict(user_encoded_df, alpha=alpha)

    y_pred = y_pred.round(2)
    y_pis = y_pis.round(2)

    st.subheader("Predicting Traffic Volume...")
    st.metric(label="Predicted Traffic Volume", value=f"{y_pred[0]:,.2f}") # used chat for syntax to round
    st.write(f"Prediction Interval ({(1-alpha):.0%} confidence level): [{y_pis[0][0].round(2)}, {y_pis[0][1].round(2)}]")

else:
    user_df = pd.read_csv(user_data)
    original_df = default_data.copy()
   # Dropping null values
    user_data = user_df.fillna("None").replace({None: "None"})
    original_df = original_df.dropna().reset_index(drop = True)
   
   # Ensure the order of columns in user data is in the same order as that of original data
    user_df = user_df[original_df.columns]

   # Concatenate two dataframes together along rows (axis = 0)
    combined_df = pd.concat([original_df, user_df], axis = 0)

   # Number of rows in original dataframe
    original_rows = original_df.shape[0]

   # Create dummies for the combined dataframe
    combined_df_encoded = pd.get_dummies(combined_df)

   # Split data into original and user dataframes using row index
    original_df_encoded = combined_df_encoded[:original_rows]
    user_df_encoded = combined_df_encoded[original_rows:]
  
    y_pred, y_pis = mapie.predict(user_df_encoded, alpha=alpha)
    # Round predictions and intervals to 1 decimal place
    y_pred = y_pred.round(1)
    y_pis = y_pis.round(1)

    user_df["Predicted Volume"] = y_pred.ravel()
    user_df[f"Lower {(1-alpha):.0%} PI"] = y_pis[:, 0].ravel()
    user_df[f"Upper {(1-alpha):.0%} PI"] = y_pis[:, 1].ravel()

    st.subheader(f"Predicting Results with {((1-alpha)*100):.0f}% Interval")
    st.dataframe(user_df)
# used chat to get interval values


# Showing additional items in tabs
st.subheader("Model Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Histogram of Residuals", "Predicted Vs. Actual", "Coverage Plot"])

# Tab 1: Feauture Importance Visualization
with tab1:
    st.write("### Feature Importance")
    st.image('feature_importance_traffic_vol.svg')
    st.caption("Relative importance of features in prediction.")

# Tab 2: Histogram of Residuals
with tab2:
    st.write("### Histogram of Residuals")
    st.image('residuals_traffic_vol.svg')
    st.caption("Histogram showing the distribution of residuals from model predictions.")

# Tab 3: Predicted Vs. Actual Prices
with tab3:
    st.write("### Predicted Vs. Actual Prices")
    st.image('pred_vs_actual_traffic_vol.svg')
    st.caption("Visual comparison of predicted and actual values.")

# Tab 4: Coverage Plot
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage_traffic_vol.svg')
    st.caption("Range of prediction with confidence interval.")