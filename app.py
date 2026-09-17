import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# =========================================================
# MODEL PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fraud_detection_model.pkl"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model_data = load_model()

    model = model_data["model"]
    scaler = model_data["scaler"]
    encoder = model_data["encoder"]

    numeric_features = model_data["numeric_features"]
    categorical_features = model_data["categorical_features"]

    best_threshold = model_data.get("threshold", 0.5)

    model_loaded = True

except Exception as e:
    model_loaded = False
    model_error = str(e)


# =========================================================
# HAVERSINE DISTANCE
# =========================================================

def haversine_distance(lat1, lon1, lat2, lon2):

    R = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def prepare_transaction(transaction):

    transaction = transaction.copy()

    # Convert dates
    transaction["trans_date_trans_time"] = pd.to_datetime(
        transaction["trans_date_trans_time"]
    )

    transaction["dob"] = pd.to_datetime(
        transaction["dob"]
    )

    # Time features
    transaction["hour"] = (
        transaction["trans_date_trans_time"].dt.hour
    )

    transaction["day_of_week"] = (
        transaction["trans_date_trans_time"].dt.dayofweek
    )

    transaction["month"] = (
        transaction["trans_date_trans_time"].dt.month
    )

    transaction["day"] = (
        transaction["trans_date_trans_time"].dt.day
    )

    transaction["is_weekend"] = (
        transaction["day_of_week"].isin([5, 6]).astype(int)
    )

    # Age
    transaction["age"] = (
        (
            transaction["trans_date_trans_time"]
            - transaction["dob"]
        ).dt.days
        / 365.25
    )

    # Log amount
    transaction["log_amt"] = np.log1p(
        transaction["amt"]
    )

    # Distance
    transaction["distance_km"] = haversine_distance(
        transaction["lat"],
        transaction["long"],
        transaction["merch_lat"],
        transaction["merch_long"]
    )

    return transaction


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_transaction(transaction):

    transaction = prepare_transaction(transaction)

    # Numeric features
    numeric_data = transaction[numeric_features]

    # Categorical features
    categorical_data = transaction[categorical_features]

    # Scaling
    numeric_scaled = scaler.transform(
        numeric_data
    )

    # Encoding
    categorical_encoded = encoder.transform(
        categorical_data
    )

    # Combine
    processed_data = np.hstack(
        [
            numeric_scaled,
            categorical_encoded
        ]
    )

    # Probability
    probability = model.predict_proba(
        processed_data
    )[:, 1][0]

    # Prediction
    prediction = int(
        probability >= best_threshold
    )

    result = (
        "Fraud"
        if prediction == 1
        else "Normal"
    )

    return result, probability


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("💳 Fraud Detection")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Transaction Prediction",
        "Model Information"
    ]
)


# =========================================================
# MODEL ERROR
# =========================================================

if not model_loaded:

    st.error(
        "⚠️ Could not load the trained model."
    )

    st.write(
        "Make sure `fraud_detection_model.pkl` "
        "is in the same folder as `app.py`."
    )

    st.code(model_error)

    st.stop()


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title("💳 Credit Card Fraud Detection")

    st.markdown(
        """
        ## Welcome 👋

        This application uses a Machine Learning model
        to detect potentially fraudulent credit card
        transactions.
        """
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🎯 Problem",
            "Binary Classification"
        )

    with col2:
        st.metric(
            "🤖 Model",
            "XGBoost"
        )

    with col3:
        st.metric(
            "⚖️ Imbalance",
            "Highly Imbalanced"
        )

    with col4:
        st.metric(
            "🔍 Threshold",
            f"{best_threshold * 100:.2f}%"
        )

    st.divider()

    st.subheader("📊 Project Overview")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            ### What does the model do?

            The model analyzes transaction information such as:

            - 💰 Transaction amount
            - 🏪 Transaction category
            - 📍 Customer location
            - 📍 Merchant location
            - 👤 Customer age
            - 🕐 Transaction hour
            - 📅 Day and month
            - 🚗 Distance between customer and merchant

            The model then estimates the probability
            that the transaction is fraudulent.
            """
        )

    with col2:

        st.markdown(
            """
            ### Why is fraud detection difficult?

            Credit card fraud datasets are usually highly
            imbalanced.

            Most transactions are normal while only a very
            small percentage are fraudulent.

            Therefore, accuracy alone is not enough.

            Important evaluation metrics include:

            - Precision
            - Recall
            - F1 Score
            - ROC-AUC
            - PR-AUC
            """
        )

    st.divider()

    st.info(
        "💡 Go to 'Transaction Prediction' from the sidebar "
        "to test a transaction."
    )


# =========================================================
# TRANSACTION PREDICTION
# =========================================================

elif page == "Transaction Prediction":

    st.title("🔎 Transaction Prediction")

    st.markdown(
        """
        Enter transaction information below and the
        Machine Learning model will estimate the fraud
        probability.
        """
    )

    st.divider()

    # =====================================================
    # FRAUD DEMO
    # =====================================================

    st.subheader("🧪 Demo Transaction")

    st.write(
        "Use the button below to load a high-risk sample "
        "transaction for testing the fraud detection model."
    )

    if st.button(
        "🚨 Test Fraud Transaction",
        use_container_width=True
    ):

        st.session_state["demo_amount"] = 5000.0
        st.session_state["demo_category"] = "shopping_net"
        st.session_state["demo_gender"] = "M"
        st.session_state["demo_state"] = "NY"

        st.session_state["demo_latitude"] = 40.7128
        st.session_state["demo_longitude"] = -74.0060

        st.session_state["demo_merch_latitude"] = 34.0522
        st.session_state["demo_merch_longitude"] = -118.2437

        st.session_state["demo_population"] = 100000

        st.session_state["demo_date"] = (
            pd.Timestamp("2026-09-15").date()
        )

        st.session_state["demo_time"] = (
            pd.Timestamp("2026-09-15 03:00:00").time()
        )

        st.session_state["demo_dob"] = (
            pd.Timestamp("1980-01-01").date()
        )

        st.session_state["demo_loaded"] = True

        st.rerun()

    st.divider()

    # =====================================================
    # TRANSACTION INFORMATION
    # =====================================================

    st.subheader("💳 Transaction Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        amount = st.number_input(
            "💰 Transaction Amount ($)",
            min_value=0.0,
            value=st.session_state.get(
                "demo_amount",
                100.0
            ),
            step=1.0
        )

    with col2:

        categories = [
            "grocery_pos",
            "shopping_net",
            "misc_net",
            "shopping_pos",
            "gas_transport",
            "home",
            "kids_pets",
            "entertainment",
            "food_dining",
            "personal_care",
            "health_fitness",
            "travel"
        ]

        selected_category = st.session_state.get(
            "demo_category",
            "grocery_pos"
        )

        if selected_category not in categories:
            selected_category = "grocery_pos"

        category = st.selectbox(
            "🛒 Category",
            categories,
            index=categories.index(
                selected_category
            )
        )

    with col3:

        gender_options = ["M", "F"]

        selected_gender = st.session_state.get(
            "demo_gender",
            "M"
        )

        gender = st.selectbox(
            "👤 Gender",
            gender_options,
            index=gender_options.index(
                selected_gender
            )
        )

    with col4:

        state = st.text_input(
            "🇺🇸 State",
            value=st.session_state.get(
                "demo_state",
                "NY"
            )
        )

    st.divider()

    # =====================================================
    # LOCATION INFORMATION
    # =====================================================

    st.subheader("📍 Location Information")

    # FIXED:
    # 4 columns = 4 variables

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        latitude = st.number_input(
            "Customer Latitude",
            value=st.session_state.get(
                "demo_latitude",
                40.7128
            ),
            format="%.6f"
        )

    with col2:

        longitude = st.number_input(
            "Customer Longitude",
            value=st.session_state.get(
                "demo_longitude",
                -74.0060
            ),
            format="%.6f"
        )

    with col3:

        merchant_latitude = st.number_input(
            "Merchant Latitude",
            value=st.session_state.get(
                "demo_merch_latitude",
                40.7138
            ),
            format="%.6f"
        )

    with col4:

        merchant_longitude = st.number_input(
            "Merchant Longitude",
            value=st.session_state.get(
                "demo_merch_longitude",
                -74.0050
            ),
            format="%.6f"
        )

    st.divider()

    # =====================================================
    # CUSTOMER INFORMATION
    # =====================================================

    st.subheader("👤 Customer Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        city_population = st.number_input(
            "🏙️ City Population",
            min_value=0,
            value=st.session_state.get(
                "demo_population",
                100000
            ),
            step=1000
        )

    with col2:

        date_value = st.date_input(
            "📅 Transaction Date",
            value=st.session_state.get(
                "demo_date",
                pd.Timestamp.today().date()
            )
        )

    with col3:

        transaction_time = st.time_input(
            "🕐 Transaction Time",
            value=st.session_state.get(
                "demo_time",
                pd.Timestamp(
                    "2026-09-15 12:00:00"
                ).time()
            )
        )

    with col4:

        date_of_birth = st.date_input(
            "🎂 Date of Birth",
            value=st.session_state.get(
                "demo_dob",
                pd.Timestamp(
                    "1980-01-01"
                ).date()
            )
        )

    st.divider()

    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    predict_button = st.button(
        "🔍 Predict Transaction",
        use_container_width=True
    )

    if predict_button:

        # Combine date and time
        transaction_datetime = pd.Timestamp.combine(
            date_value,
            transaction_time
        )

        # Input DataFrame
        transaction = pd.DataFrame(
            {
                "amt": [amount],

                "city_pop": [
                    city_population
                ],

                "lat": [
                    latitude
                ],

                "long": [
                    longitude
                ],

                "merch_lat": [
                    merchant_latitude
                ],

                "merch_long": [
                    merchant_longitude
                ],

                "category": [
                    category
                ],

                "gender": [
                    gender
                ],

                "state": [
                    state.strip().upper()
                ],

                "trans_date_trans_time": [
                    transaction_datetime
                ],

                "dob": [
                    pd.Timestamp(
                        date_of_birth
                    )
                ]
            }
        )

        try:

            # Predict
            result, probability = predict_transaction(
                transaction
            )

            probability_percent = (
                probability * 100
            )

            st.divider()

            # =================================================
            # PREDICTION RESULT
            # =================================================

            st.subheader("📊 Prediction Result")

            if result == "Fraud":

                st.error(
                    f"""
                    ⚠️ FRAUD DETECTED

                    Fraud Probability:
                    {probability_percent:.4f}%
                    """
                )

            else:

                st.success(
                    f"""
                    ✅ TRANSACTION IS NORMAL

                    Fraud Probability:
                    {probability_percent:.4f}%
                    """
                )

            # =================================================
            # GAUGE
            # =================================================

            st.subheader("🎯 Fraud Probability")

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability_percent,
                    number={
                        "suffix": "%"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        },

                        "bar": {
                            "thickness": 0.6
                        },

                        "steps": [
                            {
                                "range": [0, 30]
                            },
                            {
                                "range": [30, 70]
                            },
                            {
                                "range": [70, 100]
                            }
                        ],

                        "threshold": {
                            "line": {
                                "width": 4
                            },
                            "value":
                                best_threshold * 100
                        }
                    }
                )
            )

            fig.update_layout(
                height=350
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # =================================================
            # RISK ASSESSMENT
            # =================================================

            st.subheader("⚠️ Risk Assessment")

            if probability < 0.30:

                st.success(
                    "🟢 LOW RISK"
                )

            elif probability < 0.70:

                st.warning(
                    "🟡 MEDIUM RISK"
                )

            else:

                st.error(
                    "🔴 HIGH RISK"
                )

            # =================================================
            # TRANSACTION SUMMARY
            # =================================================

            st.subheader("📋 Transaction Summary")

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Amount",
                    f"${amount:,.2f}"
                )

            with col2:

                st.metric(
                    "Category",
                    category
                )

            with col3:

                st.metric(
                    "State",
                    state.strip().upper()
                )

            with col4:

                distance = haversine_distance(
                    latitude,
                    longitude,
                    merchant_latitude,
                    merchant_longitude
                )

                st.metric(
                    "Distance",
                    f"{distance:.2f} km"
                )

            # =================================================
            # MODEL DECISION
            # =================================================

            st.subheader("🤖 Model Decision")

            st.info(
                f"""
                The model classified this transaction as:

                **{result}**

                Fraud probability:
                **{probability_percent:.4f}%**

                Decision threshold:
                **{best_threshold * 100:.2f}%**
                """
            )

        except Exception as e:

            st.error(
                "❌ An error occurred while predicting."
            )

            st.code(
                str(e)
            )


# =========================================================
# MODEL INFORMATION
# =========================================================

elif page == "Model Information":

    st.title("🤖 Model Information")

    st.markdown(
        """
        This page provides information about the
        Machine Learning model used by the application.
        """
    )

    st.divider()

    # =====================================================
    # MODEL DETAILS
    # =====================================================

    st.subheader("🧠 Machine Learning Model")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Model",
            "XGBoost"
        )

    with col2:

        st.metric(
            "Task",
            "Binary Classification"
        )

    with col3:

        st.metric(
            "Threshold",
            f"{best_threshold * 100:.2f}%"
        )

    st.divider()

    # =====================================================
    # FEATURES
    # =====================================================

    st.subheader("📌 Features Used")

    st.markdown(
        """
        ### Numerical Features

        - Transaction Amount
        - Log Transaction Amount
        - City Population
        - Customer Latitude
        - Customer Longitude
        - Merchant Latitude
        - Merchant Longitude
        - Distance
        - Age
        - Hour
        - Day of Week
        - Month
        - Day
        - Weekend Indicator

        ### Categorical Features

        - Category
        - Gender
        - State
        """
    )

    st.divider()

    # =====================================================
    # PREPROCESSING
    # =====================================================

    st.subheader("⚙️ Preprocessing")

    st.markdown(
        """
        The data preprocessing process includes:

        1. Date conversion
        2. Feature engineering
        3. Numerical feature scaling using StandardScaler
        4. Categorical feature encoding using OneHotEncoder
        5. Combining numerical and categorical features
        6. Handling class imbalance using model weighting
        """
    )

    st.divider()

    # =====================================================
    # CLASS IMBALANCE
    # =====================================================

    st.subheader("⚖️ Class Imbalance")

    st.markdown(
        """
        Credit card fraud detection is a highly imbalanced
        classification problem.

        The number of normal transactions is much larger
        than the number of fraudulent transactions.

        Therefore, the model uses class weighting
        to give more importance to fraudulent transactions.
        """
    )

    st.divider()

    # =====================================================
    # EVALUATION METRICS
    # =====================================================

    st.subheader("📊 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            """
            **Precision**

            Measures how many transactions predicted
            as fraud were actually fraud.
            """
        )

    with col2:

        st.info(
            """
            **Recall**

            Measures how many actual fraudulent
            transactions were detected.
            """
        )

    with col3:

        st.info(
            """
            **F1 Score**

            Provides a balance between Precision
            and Recall.
            """
        )

    st.divider()

    st.success(
        "✅ Model loaded successfully and ready for predictions."
    )