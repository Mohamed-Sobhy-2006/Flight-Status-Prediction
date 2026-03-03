import streamlit as st
import base64
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# =========================
# Load Saved Objects
# =========================
with open('one_hot_encoder.pkl', 'rb') as f:
    ohe = pickle.load(f)

with open('target_encoder.pkl', 'rb') as f:
    tg = pickle.load(f)

with open('standard_scaler.pkl', 'rb') as f:
    standard_scaler = pickle.load(f)

with open('robust_scaler.pkl', 'rb') as f:
    robust_scaler = pickle.load(f)

with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# =========================
# Load Distances File
# =========================
distances_df = pd.read_csv(r"D:\NTI - ML\Flight_Status\distances_symmetric.csv")
distances_df['City1'] = distances_df['City1'].str.strip()
distances_df['City2'] = distances_df['City2'].str.strip()

airport_distance = {
    frozenset([row['City1'], row['City2']]): row['Distance']
    for _, row in distances_df.iterrows()
}

# =========================
# UI
# =========================
st.title("Flight Status Prediction ✈️")

origin_city_options = sorted([
    'Atlanta, GA','Fort Lauderdale, FL','Jackson/Vicksburg, MS','Richmond, VA',
    'Minneapolis, MN','Raleigh/Durham, NC','Nashville, TN','Indianapolis, IN',
    'New York, NY','Savannah, GA','Fayetteville, AR','San Antonio, TX','Tampa, FL',
    'Salt Lake City, UT','Hartford, CT','Jacksonville, FL','Boston, MA',
    'Fort Myers, FL','Seattle, WA','Harrisburg, PA','Miami, FL','Pittsburgh, PA',
    'Orlando, FL','Charlotte, NC','Columbia, SC','Newark, NJ','Buffalo, NY',
    'Philadelphia, PA','Cincinnati, OH','Las Vegas, NV','Portland, OR',
    'Washington, DC','Detroit, MI','Norfolk, VA','Sioux Falls, SD','Los Angeles, CA',
    'Dallas/Fort Worth, TX','Memphis, TN','San Juan, PR','Rochester, NY','Chicago, IL',
    'Louisville, KY','Columbus, OH','Pensacola, FL','Phoenix, AZ','Charleston, SC',
    'Dayton, OH','Portland, ME','Jackson, WY','Santa Ana, CA','Kansas City, MO',
    'Long Beach, CA','Boise, ID','New Orleans, LA','Tulsa, OK','Charlotte Amalie, VI',
    'Grand Rapids, MI','Madison, WI','Austin, TX','Christiansted, VI','Oakland, CA',
    'Denver, CO','Baltimore, MD','Little Rock, AR','Houston, TX','Panama City, FL',
    'Daytona Beach, FL','Sarasota/Bradenton, FL','San Diego, CA',
    'Valparaiso, FL','Huntsville, AL','San Jose, CA','West Palm Beach/Palm Beach, FL',
    'St. Louis, MO','Myrtle Beach, SC','Cleveland, OH','Ontario, CA','Omaha, NE',
    'San Francisco, CA','Milwaukee, WI','Anchorage, AK','Fairbanks, AK','White Plains, NY',
    'Birmingham, AL','Knoxville, TN','Oklahoma City, OK','Hayden, CO','Chattanooga, TN',
    'Bozeman, MT','Des Moines, IA','Key West, FL','Wichita, KS','Palm Springs, CA',
    'Roanoke, VA','Greensboro/High Point, NC','Tallahassee, FL','Spokane, WA',
    'Appleton, WI','Gainesville, FL','Kalispell, MT','Melbourne, FL','Greer, SC',
    'Lexington, KY','Baton Rouge, LA','Asheville, NC','Reno, NV','Gulfport/Biloxi, MS',
    'Springfield, MO','Dallas, TX','Providence, RI','Sacramento, CA','Albany, NY',
    'Missoula, MT','Syracuse, NY','Eagle, CO','Honolulu, HI','Lihue, HI','Kahului, HI',
    'Kona, HI','Albuquerque, NM','Tucson, AZ','El Paso, TX','Burlington, VT','Fargo, ND'
])

dest_city_options = origin_city_options.copy()

col1, col2, col3 = st.columns(3)
with col1:
    day = st.selectbox("Day", list(range(1,32)))
with col2:
    month = st.selectbox("Month", list(range(1,13)))
with col3:
    st.write("Distance calculated automatically")

col4, col5, col6 = st.columns(3)
with col4:
    airlines = ['Delta Airlines','Frontier Airlines','Allegiant Air','Hawaiian Airlines',
                'American Airlines','Spirit Airlines','Alaska Airlines',
                'Southwest Airlines','United Airlines','JetBlue Airways']
    airline = st.selectbox("Airline", sorted(airlines))
with col5:
    origin_city = st.selectbox("Origin City", origin_city_options)
with col6:
    dest_city = st.selectbox("Destination City", dest_city_options)

# =========================
# Prediction
# =========================
if st.button("Predict"):

    year = 2022

    # ===== Date Validation =====
    try:
        date_obj = datetime(year, month, day)
    except ValueError:
        st.warning("This date does not exist. Please select a valid day for the chosen month.")
        st.stop()

    day_of_week = date_obj.isoweekday()
    is_weekend = 1 if day_of_week in [6,7] else 0

    origin_city = origin_city.strip()
    dest_city = dest_city.strip()

    if origin_city == dest_city:
        st.info("No Flight Needed 😂😂 (Same Origin and Destination)")
        st.stop()

    # ===== Distance Calculation =====
    pair = frozenset([origin_city, dest_city])
    distance = airport_distance.get(pair, None)

    if distance is None:
        related_distances = [
            d for key, d in airport_distance.items()
            if origin_city in key or dest_city in key
        ]
        if related_distances:
            distance = min(related_distances)
        else:
            distance = distances_df['Distance'].mean()

    st.write(f"🛫 Distance between {origin_city} and {dest_city} is {round(distance,2)} miles")

    # ===== Feature Engineering =====
    airline_delay_rate = {
        'Allegiant Air':0.2,'American Airlines':0.25,'Delta Airlines':0.22,
        'Frontier Airlines':0.28,'Hawaiian Airlines':0.15,'JetBlue Airways':0.18,
        'Southwest Airlines':0.2,'Spirit Airlines':0.3,
        'United Airlines':0.23,'Alaska Airlines':0.21
    }

    airline_history_delay = airline_delay_rate.get(airline,0.2)
    is_peak_month = 1 if month in [6,12] else 0

    def get_season(m):
        if m in [12,1,2]: return 'Winter'
        elif m in [3,4,5]: return 'Spring'
        elif m in [6,7,8]: return 'Summer'
        else: return 'Autumn'

    season = get_season(month)

    month_sin = np.sin(2*np.pi*month/12)
    month_cos = np.cos(2*np.pi*month/12)
    dow_sin = np.sin(2*np.pi*day_of_week/7)
    dow_cos = np.cos(2*np.pi*day_of_week/7)

    user_df = pd.DataFrame({
        'DayofMonth':[day],
        'Year':[year],
        'Distance':[distance],
        'OriginCityName':[origin_city],
        'DestCityName':[dest_city],
        'Airline_History_Delay':[airline_history_delay],
        'Is_Peak_Month':[is_peak_month],
        'IsWeekend':[is_weekend],
        'Month_sin':[month_sin],
        'Month_cos':[month_cos],
        'DayOfWeek_sin':[dow_sin],
        'DayOfWeek_cos':[dow_cos],
        'Airlines':[airline],
        'Season':[season]
    })

    # Encoding
    encoded = ohe.transform(user_df[['Airlines','Season']])
    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(['Airlines','Season'])
    )

    user_df = pd.concat(
        [user_df.drop(['Airlines','Season'],axis=1),
         encoded_df],
        axis=1
    )

    user_df[['OriginCityName','DestCityName']] = \
        tg.transform(user_df[['OriginCityName','DestCityName']])

    user_df[['DayofMonth','Year']] = \
        standard_scaler.transform(user_df[['DayofMonth','Year']])

    user_df[['Distance']] = \
        robust_scaler.transform(user_df[['Distance']])

    model_cols = model.get_booster().feature_names
    for col in model_cols:
        if col not in user_df.columns:
            user_df[col] = 0

    user_df = user_df[model_cols]

    prediction = model.predict(user_df)[0]

    if prediction == 1:
        st.error("Flight is DELAYED 🚨")
    else:
        st.success("Flight is ON TIME ✅")