# ============================================================
# LATE DELIVERY RISK PREDICTOR
# ============================================================

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Late Delivery Risk Predictor",
    page_icon="🚚",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 30px;
    }

    .risk-low {
        padding: 20px;
        border-radius: 10px;
        background-color: #123d2a;
        color: #55e69b;
        font-size: 20px;
        font-weight: 600;
    }

    .risk-medium {
        padding: 20px;
        border-radius: 10px;
        background-color: #4a3f0b;
        color: #ffd84d;
        font-size: 20px;
        font-weight: 600;
    }

    .risk-high {
        padding: 20px;
        border-radius: 10px;
        background-color: #472126;
        color: #ff6b6b;
        font-size: 20px;
        font-weight: 600;
    }

    .probability {
        font-size: 38px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "Data" / "APL_Logistics.csv"

MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"

PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"

FEATURE_INFO_PATH = BASE_DIR / "models" / "feature_info.pkl"


# ============================================================
# MODEL FEATURES
# ============================================================

DEFAULT_FEATURES = [
    "Type",
    "Days for shipment (scheduled)",
    "Benefit per order",
    "Sales per customer",
    "Category Name",
    "Customer Segment",
    "Customer Country",
    "Customer State",
    "Department Name",
    "Market",
    "Order Country",
    "Order Region",
    "Product Price",
    "Shipping Mode"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    feature_info = joblib.load(FEATURE_INFO_PATH)

    return model, preprocessor, feature_info


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    # Try different encodings.
    # This fixes the UTF-8 error you previously got.

    encodings = [
        "utf-8",
        "latin1",
        "cp1252"
    ]

    for encoding in encodings:

        try:

            df = pd.read_csv(
                DATA_PATH,
                encoding=encoding
            )

            return df

        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        "Could not read CSV using UTF-8, latin1 or cp1252."
    )


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    model, preprocessor, feature_info = load_model()

    df = load_data()

except Exception as e:

    st.error("Unable to load the model or dataset.")

    st.exception(e)

    st.stop()


# ============================================================
# GET FEATURES
# ============================================================

if isinstance(feature_info, dict):

    FEATURES = feature_info.get(
        "features",
        DEFAULT_FEATURES
    )

else:

    FEATURES = DEFAULT_FEATURES


# Make sure our expected features exist.

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    st.error(
        f"These model features are missing from the dataset: "
        f"{missing_features}"
    )

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🚚 Late Delivery Risk Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Predict whether an order is likely to experience late delivery
    using a Machine Learning model.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# ORDER INFORMATION
# ============================================================

st.header("📦 Order Information")


# ============================================================
# COLUMNS
# ============================================================

left_column, right_column = st.columns(2)


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_options(column):

    values = (
        df[column]
        .dropna()
        .unique()
        .tolist()
    )

    return values


# ============================================================
# LEFT COLUMN - CATEGORICAL FEATURES
# ============================================================

with left_column:

    # --------------------------------------------------------
    # TYPE
    # --------------------------------------------------------

    type_options = get_options("Type")

    type_value = st.selectbox(
        "Type",
        type_options
    )


    # --------------------------------------------------------
    # CATEGORY NAME
    # --------------------------------------------------------

    category_options = get_options("Category Name")

    category_value = st.selectbox(
        "Category Name",
        category_options
    )


    # --------------------------------------------------------
    # CUSTOMER SEGMENT
    # --------------------------------------------------------

    segment_options = get_options("Customer Segment")

    segment_value = st.selectbox(
        "Customer Segment",
        segment_options
    )


    # --------------------------------------------------------
    # CUSTOMER COUNTRY
    # --------------------------------------------------------

    customer_country_options = get_options(
        "Customer Country"
    )

    customer_country_value = st.selectbox(
        "Customer Country",
        customer_country_options
    )


    # --------------------------------------------------------
    # CUSTOMER STATE
    # --------------------------------------------------------

    customer_state_options = get_options(
        "Customer State"
    )

    customer_state_value = st.selectbox(
        "Customer State",
        customer_state_options
    )


    # --------------------------------------------------------
    # DEPARTMENT
    # --------------------------------------------------------

    department_options = get_options(
        "Department Name"
    )

    department_value = st.selectbox(
        "Department Name",
        department_options
    )


    # --------------------------------------------------------
    # MARKET
    # --------------------------------------------------------

    market_options = get_options("Market")

    market_value = st.selectbox(
        "Market",
        market_options
    )


    # --------------------------------------------------------
    # ORDER COUNTRY
    # --------------------------------------------------------

    order_country_options = get_options(
        "Order Country"
    )

    order_country_value = st.selectbox(
        "Order Country",
        order_country_options
    )


    # --------------------------------------------------------
    # ORDER REGION
    # --------------------------------------------------------

    order_region_options = get_options(
        "Order Region"
    )

    order_region_value = st.selectbox(
        "Order Region",
        order_region_options
    )


    # --------------------------------------------------------
    # SHIPPING MODE
    # --------------------------------------------------------

    shipping_options = get_options(
        "Shipping Mode"
    )

    shipping_value = st.selectbox(
        "Shipping Mode",
        shipping_options
    )


# ============================================================
# RIGHT COLUMN - NUMERICAL FEATURES
# ============================================================

with right_column:

    # --------------------------------------------------------
    # DAYS FOR SHIPMENT
    # --------------------------------------------------------

    days_column = "Days for shipment (scheduled)"

    days_data = pd.to_numeric(
        df[days_column],
        errors="coerce"
    ).dropna()

    days_value = st.number_input(
        "Days for shipment (scheduled)",
        min_value=float(days_data.min()),
        max_value=float(days_data.max()),
        value=float(days_data.median()),
        step=1.0
    )


    # --------------------------------------------------------
    # BENEFIT PER ORDER
    # --------------------------------------------------------

    benefit_column = "Benefit per order"

    benefit_data = pd.to_numeric(
        df[benefit_column],
        errors="coerce"
    ).dropna()

    benefit_value = st.number_input(
        "Benefit per order",
        min_value=float(benefit_data.min()),
        max_value=float(benefit_data.max()),
        value=float(benefit_data.median()),
        step=0.01
    )


    # --------------------------------------------------------
    # SALES PER CUSTOMER
    # --------------------------------------------------------

    sales_column = "Sales per customer"

    sales_data = pd.to_numeric(
        df[sales_column],
        errors="coerce"
    ).dropna()

    sales_value = st.number_input(
        "Sales per customer",
        min_value=float(sales_data.min()),
        max_value=float(sales_data.max()),
        value=float(sales_data.median()),
        step=0.01
    )


    # --------------------------------------------------------
    # PRODUCT PRICE
    # --------------------------------------------------------

    price_column = "Product Price"

    price_data = pd.to_numeric(
        df[price_column],
        errors="coerce"
    ).dropna()

    price_value = st.number_input(
        "Product Price",
        min_value=float(price_data.min()),
        max_value=float(price_data.max()),
        value=float(price_data.median()),
        step=0.01
    )


# ============================================================
# SEPARATOR
# ============================================================

st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔮 Predict Delivery Risk",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------------------------

        input_data = {

            "Type": type_value,

            "Days for shipment (scheduled)": days_value,

            "Benefit per order": benefit_value,

            "Sales per customer": sales_value,

            "Category Name": category_value,

            "Customer Segment": segment_value,

            "Customer Country": customer_country_value,

            "Customer State": customer_state_value,

            "Department Name": department_value,

            "Market": market_value,

            "Order Country": order_country_value,

            "Order Region": order_region_value,

            "Product Price": price_value,

            "Shipping Mode": shipping_value
        }


        # ----------------------------------------------------
        # CONVERT TO DATAFRAME
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [input_data]
        )


        # Make sure columns are in exactly the same
        # order used during model training.

        input_df = input_df[FEATURES]


        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        X_input = preprocessor.transform(
            input_df
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            X_input
        )[0]


        # ----------------------------------------------------
        # PREDICTION PROBABILITY
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            X_input
        )[0]


        # Find probability corresponding to class 1.

        if hasattr(model, "classes_"):

            class_list = list(model.classes_)

            if 1 in class_list:

                positive_class_index = class_list.index(1)

            else:

                positive_class_index = 1

        else:

            positive_class_index = 1


        probability = probabilities[
            positive_class_index
        ]


        # Convert to percentage.

        probability_percent = probability * 100


        # ====================================================
        # THREE-LEVEL RISK CLASSIFICATION
        # ====================================================

        if probability < 0.40:

            risk_status = (
                "LOW RISK - Likely On-Time Delivery"
            )

            risk_type = "low"


        elif probability <= 0.60:

            risk_status = (
                "MEDIUM RISK - Borderline Delivery"
            )

            risk_type = "medium"


        else:

            risk_status = (
                "HIGH RISK - Likely Late Delivery"
            )

            risk_type = "high"


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.divider()

        st.header("📊 Prediction Result")


        # ----------------------------------------------------
        # LOW RISK
        # ----------------------------------------------------

        if risk_type == "low":

            st.markdown(
                f"""
                <div class="risk-low">
                🟢 {risk_status}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            st.write(
                "Late Delivery Probability"
            )

            st.markdown(
                f"""
                <div class="probability">
                {probability_percent:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                "This order has been classified as low risk. "
                "The model predicts that the order is more "
                "likely to be delivered on time."
            )


        # ----------------------------------------------------
        # MEDIUM RISK
        # ----------------------------------------------------

        elif risk_type == "medium":

            st.markdown(
                f"""
                <div class="risk-medium">
                🟡 {risk_status}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            st.write(
                "Late Delivery Probability"
            )

            st.markdown(
                f"""
                <div class="probability">
                {probability_percent:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

            st.warning(
                "This order is in the borderline risk range. "
                "The model is not strongly confident that the "
                "order will be on time or late. Consider "
                "reviewing the shipping mode, scheduled shipment "
                "time and other order characteristics."
            )


        # ----------------------------------------------------
        # HIGH RISK
        # ----------------------------------------------------

        else:

            st.markdown(
                f"""
                <div class="risk-high">
                🔴 {risk_status}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            st.write(
                "Late Delivery Probability"
            )

            st.markdown(
                f"""
                <div class="probability">
                {probability_percent:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

            st.error(
                "This order has been classified as high risk. "
                "Consider reviewing the shipping mode, scheduled "
                "shipment time and other order characteristics."
            )


        # ====================================================
        # ABOUT MODEL
        # ====================================================

        with st.expander("ℹ️ About this Model"):

            st.write(
                """
                **Model:** Random Forest Classifier

                **Purpose:** Predict the risk of late delivery.

                **Model Output:**
                - 🟢 Low Risk → Probability below 40%
                - 🟡 Medium Risk → Probability between 40% and 60%
                - 🔴 High Risk → Probability above 60%

                The model uses order, customer, product and
                shipping-related features to make the prediction.
                """
            )

            st.write(
                f"**Model Prediction Class:** {prediction}"
            )

    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Late Delivery Risk Prediction • Machine Learning Project"
)