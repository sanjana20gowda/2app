import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
# -----------------------------
# Login Authentication
# -----------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

st.set_page_config(
    page_title="Late Delivery Risk Prediction",
    page_icon="🚚",
    layout="wide"
)
st.markdown("""
<style>

/* Main Background */
.stApp{
    background-color:#f5f7fa;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background-color:#003366;
}

section[data-testid="stSidebar"] *{
    color:white;
}

/* Main Title */
h1{
    color:#003366;
    text-align:center;
}

/* Headers */
h2,h3{
    color:#0a4d8c;
}

/* Metric Cards */
div[data-testid="metric-container"]{
    background:white;
    border-radius:12px;
    padding:15px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.15);
    border-left:6px solid #0066cc;
}

/* DataFrame */
[data-testid="stDataFrame"]{
    border-radius:10px;
}

/* Buttons */
.stButton>button{
    background-color:#0066cc;
    color:white;
    border-radius:10px;
    height:3em;
    width:100%;
    font-size:16px;
}

.stButton>button:hover{
    background-color:#004b99;
    color:white;
}

/* Download Button */
.stDownloadButton>button{
    background-color:#28a745;
    color:white;
    border-radius:10px;
}

.stDownloadButton>button:hover{
    background-color:#1f8a37;
    color:white;
}

</style>
""", unsafe_allow_html=True)
# -----------------------------
# Login Page
# -----------------------------

if not st.session_state.logged_in:

    st.title("🔐 Login")

    st.markdown("### Late Delivery Risk Prediction System")

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        if username == "admin" and password == "admin123":

            st.session_state.logged_in = True

            st.success("Login Successful")

            st.rerun()

        else:

            st.error("Invalid Username or Password")

    st.stop()
st.title("🚚 Machine Learning–based Late Delivery Risk Prediction")
st.markdown("### APL Logistics (KWE Group)")
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv(
    "data/APL_Logistics.csv",
    encoding="latin1"
)
feature_df = pd.read_csv("model_features.csv")
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

if st.sidebar.button("🚪 Logout"):

    st.session_state.logged_in = False

    st.rerun()
st.sidebar.markdown("---")

st.sidebar.success("Machine Learning Dashboard")

st.sidebar.write("Version : 1.0")

st.sidebar.write("Developer : Sanjana D")

st.sidebar.write("University : Mysore University")

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Choose Module",
    [
         "🏠 Dashboard",
        "🔍 Predict Delivery Risk",
        "🌍 Region Analysis",
        "📈 Model Performance",
        "ℹ About Project",
    ]
)
if page == "🏠 Dashboard":

    st.header("📊 Dashboard")

    current_time = datetime.now().strftime("%d-%m-%Y %I:%M %p")
    st.info(f"📅 Dashboard Updated : {current_time}")

    st.markdown("""
Welcome to the **Late Delivery Risk Prediction Dashboard**

This dashboard helps logistics managers monitor delayed deliveries,
analyze shipping performance and predict delivery risk using Machine Learning.
""")

    st.markdown("---")

    # -----------------------------
    # Filters
    # -----------------------------

    st.subheader("🔎 Dashboard Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_region = st.selectbox(
            "🌍 Order Region",
            ["All"] + sorted(df["Order Region"].dropna().unique())
        )

    with col2:
        selected_shipping = st.selectbox(
            "🚚 Shipping Mode",
            ["All"] + sorted(df["Shipping Mode"].dropna().unique())
        )

    with col3:
        selected_status = st.selectbox(
            "📦 Order Status",
            ["All"] + sorted(df["Order Status"].dropna().unique())
        )

    filtered_df = df.copy()

    if selected_region != "All":
        filtered_df = filtered_df[
            filtered_df["Order Region"] == selected_region
        ]

    if selected_shipping != "All":
        filtered_df = filtered_df[
            filtered_df["Shipping Mode"] == selected_shipping
        ]

    if selected_status != "All":
        filtered_df = filtered_df[
            filtered_df["Order Status"] == selected_status
        ]

    st.markdown("---")

    # -----------------------------
    # KPI Cards
    # -----------------------------

    total_orders = len(filtered_df)
    delayed_orders = filtered_df["Late_delivery_risk"].sum()
    ontime_orders = total_orders - delayed_orders

    delay_rate = 0

    if total_orders > 0:
        delay_rate = delayed_orders / total_orders * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📦 Total Orders", f"{total_orders:,}")
    c2.metric("🚚 Delayed Orders", f"{int(delayed_orders):,}")
    c3.metric("✅ On-Time Orders", f"{int(ontime_orders):,}")
    c4.metric("⚠ Delay Rate", f"{delay_rate:.2f}%")

    st.markdown("---")

    # -----------------------------
    # Charts
    # -----------------------------

    left, right = st.columns(2)

    with left:

        pie = px.pie(
            filtered_df,
            names="Late_delivery_risk",
            hole=0.45,
            title="Delivery Status"
        )

        pie.update_traces(textinfo="percent+label")

        st.plotly_chart(
            pie,
            use_container_width=True
        )

    with right:

        region = (
            filtered_df.groupby("Order Region")["Late_delivery_risk"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            region,
            x="Order Region",
            y="Late_delivery_risk",
            color="Late_delivery_risk",
            title="Top 10 Regions with Late Deliveries"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # -----------------------------
    # Recent Orders
    # -----------------------------

    st.subheader("📄 Recent Orders")

    st.dataframe(
        filtered_df.head(20),
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------
    # Operations Action Panel
    # -----------------------------

    st.subheader("🚨 Operations Action Panel")

    X_filtered = feature_df.loc[filtered_df.index]

    X_scaled = scaler.transform(X_filtered)

    filtered_df = filtered_df.copy()

    filtered_df["Risk Probability"] = model.predict_proba(X_scaled)[:, 1]

    high_risk = filtered_df[
        filtered_df["Risk Probability"] >= 0.70
    ].copy()

    if len(high_risk) > 0:

        high_risk["Recommended Action"] = "Prioritize Shipment"

        st.warning(
            f"⚠ {len(high_risk)} High-Risk Orders Need Immediate Attention"
        )

        st.dataframe(
            high_risk[
                [
                    "Order Region",
                    "Shipping Mode",
                    "Order Status",
                    "Risk Probability",
                    "Recommended Action"
                ]
            ],
            use_container_width=True
        )

    else:

        st.success("🎉 No High-Risk Orders Found")

    st.markdown("---")

    # -----------------------------
    # Download Report
    # -----------------------------

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Dashboard Report",
        csv,
        "Dashboard_Report.csv",
        "text/csv"
    )

    st.markdown("---")

    # -----------------------------
    # Statistics
    # -----------------------------

    left, right = st.columns(2)

    with left:

        st.subheader("💰 Sales Statistics")

        st.write(f"Average Sales : ${filtered_df['Sales'].mean():,.2f}")
        st.write(f"Maximum Sales : ${filtered_df['Sales'].max():,.2f}")
        st.write(f"Minimum Sales : ${filtered_df['Sales'].min():,.2f}")

    with right:

        st.subheader("💵 Profit Statistics")

        st.write(f"Average Profit : ${filtered_df['Order Profit Per Order'].mean():,.2f}")
        st.write(f"Maximum Profit : ${filtered_df['Order Profit Per Order'].max():,.2f}")
        st.write(f"Minimum Profit : ${filtered_df['Order Profit Per Order'].min():,.2f}")

    st.markdown("---")

    # -----------------------------
    # Top Products
    # -----------------------------

    st.subheader("🏆 Top 10 Products by Sales")

    top_products = (
        filtered_df.groupby("Product Name")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        top_products,
        x="Product Name",
        y="Sales",
        color="Sales",
        title="Top 10 Selling Products"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

elif page == "🔍 Predict Delivery Risk":

    st.header("🔍 Predict Delivery Risk")

    st.markdown(
        "Select an order from the dataset to predict the delivery risk."
    )

    st.markdown("---")

    order_index = st.slider(
        "Select Order Index",
        0,
        len(df) - 1,
        0
    )

    sample = feature_df.iloc[[order_index]]

    st.subheader("📄 Selected Order")

    st.dataframe(
        df.iloc[[order_index]],
        use_container_width=True
    )

    st.markdown("---")

    if st.button("🚚 Predict Delivery Risk"):

        sample_scaled = scaler.transform(sample)

        prediction = model.predict(sample_scaled)[0]

        probability = model.predict_proba(sample_scaled)[0][1]

        st.subheader("Prediction Result")

        # Risk Category
        if probability < 0.30:
            risk = "🟢 LOW RISK"
            st.success(risk)

        elif probability < 0.70:
            risk = "🟡 MEDIUM RISK"
            st.warning(risk)

        else:
            risk = "🔴 HIGH RISK"
            st.error(risk)

        # Gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": "Delay Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "red"}
                ]
            }
        ))

        st.plotly_chart(gauge, use_container_width=True)

        st.markdown("---")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "📦 Order Number",
                order_index
            )

        with c2:
            st.metric(
                "Delay Probability",
                f"{probability*100:.2f}%"
            )

        st.progress(float(probability))

        st.markdown("---")

        st.subheader("💡 Recommendation")

        if probability >= 0.70:

            st.error("""
• Prioritize Shipment

• Assign Express Shipping

• Notify Customer

• Monitor Warehouse Processing
""")

        elif probability >= 0.30:

            st.warning("""
• Monitor Shipment

• Keep Customer Updated

• Check Warehouse Status
""")

        else:

            st.success("""
• Normal Processing

• Delivery Expected On Time
""")

        st.markdown("---")

        st.subheader("📊 Feature Values")

        feature_table = pd.DataFrame({
            "Feature": sample.columns,
            "Value": sample.iloc[0].values
        })

        st.dataframe(
            feature_table,
            use_container_width=True
        )

        st.markdown("---")

        csv = feature_table.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Prediction Report",
            csv,
            "Prediction_Report.csv",
            "text/csv"
        )
elif page == "🌍 Region Analysis":

    st.header("🌍 Region Risk Analysis Dashboard")

    st.markdown("""
Analyze delivery risk across regions, markets, customer segments and shipping modes.
""")

    st.markdown("---")

    # ============================
    # Filters
    # ============================

    col1, col2 = st.columns(2)

    with col1:

        selected_market = st.selectbox(

            "Select Market",

            ["All"] + sorted(df["Market"].unique())

        )

    with col2:

        selected_segment = st.selectbox(

            "Customer Segment",

            ["All"] + sorted(df["Customer Segment"].unique())

        )

    region_df = df.copy()

    if selected_market != "All":

        region_df = region_df[
            region_df["Market"] == selected_market
        ]

    if selected_segment != "All":

        region_df = region_df[
            region_df["Customer Segment"] == selected_segment
        ]

    st.markdown("---")

    # ============================
    # KPI Cards
    # ============================

    total_orders = len(region_df)

    delayed_orders = region_df["Late_delivery_risk"].sum()

    delay_rate = delayed_orders / total_orders * 100

    c1, c2, c3 = st.columns(3)

    c1.metric("Orders", total_orders)

    c2.metric("Delayed Orders", int(delayed_orders))

    c3.metric("Delay Rate", f"{delay_rate:.2f}%")

    st.markdown("---")

    # ============================
    # Region Delay
    # ============================

    region_delay = (

        region_df.groupby("Order Region")["Late_delivery_risk"]

        .mean()

        .reset_index()

    )

    region_delay["Late_delivery_risk"] *= 100

    fig1 = px.bar(

        region_delay,

        x="Order Region",

        y="Late_delivery_risk",

        color="Late_delivery_risk",

        title="Delay Percentage by Region"

    )

    st.plotly_chart(

        fig1,

        use_container_width=True

    )

    st.markdown("---")

    left, right = st.columns(2)

    # ============================
    # Shipping Mode
    # ============================

    with left:

        ship = (

            region_df.groupby("Shipping Mode")["Late_delivery_risk"]

            .mean()

            .reset_index()

        )

        ship["Late_delivery_risk"] *= 100

        fig2 = px.pie(

            ship,

            names="Shipping Mode",

            values="Late_delivery_risk",

            hole=.45,

            title="Shipping Mode Risk"

        )

        st.plotly_chart(

            fig2,

            use_container_width=True

        )

    # ============================
    # Customer Segment
    # ============================

    with right:

        segment = (

            region_df.groupby("Customer Segment")["Late_delivery_risk"]

            .mean()

            .reset_index()

        )

        segment["Late_delivery_risk"] *= 100

        fig3 = px.bar(

            segment,

            x="Customer Segment",

            y="Late_delivery_risk",

            color="Late_delivery_risk",

            title="Risk by Customer Segment"

        )

        st.plotly_chart(

            fig3,

            use_container_width=True

        )

    st.markdown("---")

    # ============================
    # Market Analysis
    # ============================

    market = (

        region_df.groupby("Market")["Late_delivery_risk"]

        .mean()

        .reset_index()

    )

    market["Late_delivery_risk"] *= 100

    fig4 = px.bar(

        market,

        x="Market",

        y="Late_delivery_risk",

        color="Late_delivery_risk",

        title="Market-wise Delay Percentage"

    )

    st.plotly_chart(

        fig4,

        use_container_width=True

    )

    st.markdown("---")

    # ============================
    # Region Table
    # ============================

    st.subheader("📋 Region Statistics")

    st.dataframe(

        region_delay,

        use_container_width=True

    )

    st.markdown("---")

    # ============================
    # Operations Insight
    # ============================

    highest = region_delay.sort_values(

        by="Late_delivery_risk",

        ascending=False

    ).iloc[0]

    st.warning(

        f"""
Highest Risk Region : **{highest['Order Region']}**

Delay Percentage : **{highest['Late_delivery_risk']:.2f}%**

Recommended Action :

• Increase monitoring

• Allocate additional logistics resources

• Prefer Express Shipping
"""
    )
elif page == "📈 Model Performance":

    st.header("📈 Model Performance Dashboard")

    st.markdown("""
This section evaluates the Machine Learning model used to predict
late delivery risk.
""")

    st.markdown("---")

    # -----------------------------
    # Predictions
    # -----------------------------

    X = feature_df

    X_scaled = scaler.transform(X)

    y_true = df["Late_delivery_risk"]

    y_pred = model.predict(X_scaled)

    y_prob = model.predict_proba(X_scaled)[:, 1]

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score
    )

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob)

    # -----------------------------
    # KPI Cards
    # -----------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Accuracy", f"{accuracy*100:.2f}%")
    c2.metric("Precision", f"{precision*100:.2f}%")
    c3.metric("Recall", f"{recall*100:.2f}%")
    c4.metric("F1 Score", f"{f1*100:.2f}%")
    c5.metric("ROC-AUC", f"{roc*100:.2f}%")

    st.markdown("---")

    # -----------------------------
    # Confusion Matrix
    # -----------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    st.markdown("---")

    # -----------------------------
    # Feature Importance
    # -----------------------------

    st.subheader("Top 10 Important Features")

    importance = pd.DataFrame({

        "Feature": feature_df.columns,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    ).head(10)

    fig2 = px.bar(

        importance,

        x="Importance",

        y="Feature",

        orientation="h",

        color="Importance",

        title="Feature Importance"

    )

    st.plotly_chart(

        fig2,

        use_container_width=True

    )

    st.markdown("---")

    # -----------------------------
    # Classification Report
    # -----------------------------

    from sklearn.metrics import classification_report

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.subheader("Classification Report")

    st.dataframe(
        report_df,
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------
    # Model Summary
    # -----------------------------

    st.success("""
### Model Summary

✔ Random Forest Classifier

✔ Trained on APL Logistics Dataset

✔ Predicts Late Delivery Risk

✔ Provides Probability Scores

✔ Supports Operational Decision Making
""")   
elif page == "ℹ About Project":

    st.header("ℹ About the Project")

    st.markdown("---")

    st.subheader("📌 Project Title")

    st.success("""
Machine Learning–based Late Delivery Risk Prediction in Global Supply Chain Operations
""")

    st.markdown("---")

    st.subheader("📖 Background")

    st.write("""
Late deliveries are one of the major challenges in global logistics.
This project predicts whether an order is likely to be delivered late
before shipment, allowing logistics managers to take preventive action.
""")

    st.markdown("---")

    st.subheader("🎯 Objectives")

    st.markdown("""
- Predict late delivery risk using Machine Learning.
- Identify high-risk shipments.
- Improve operational planning.
- Reduce delivery delays.
- Support logistics decision making.
""")

    st.markdown("---")

    st.subheader("🛠 Technologies Used")

    tech = pd.DataFrame({

        "Technology":[
            "Python",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "Streamlit",
            "Plotly",
            "Joblib"
        ],

        "Purpose":[
            "Programming Language",
            "Data Processing",
            "Numerical Computing",
            "Machine Learning",
            "Web Application",
            "Visualization",
            "Model Saving"
        ]

    })

    st.dataframe(
        tech,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("🤖 Machine Learning Model")

    st.info("""
Algorithm Used :

✔ Random Forest Classifier

The model predicts whether an order will be delivered late by learning
historical shipment patterns.
""")

    st.markdown("---")

    st.subheader("📊 Key Performance Indicators")

    kpi = pd.DataFrame({

        "KPI":[
            "Late Delivery Probability",
            "Risk Category",
            "High-Risk Orders",
            "Feature Importance",
            "Region Risk",
            "Shipping Mode Risk"
        ],

        "Description":[
            "Probability of delivery delay",
            "Low / Medium / High",
            "Orders requiring attention",
            "Most influential variables",
            "Region-wise delay analysis",
            "Shipping performance comparison"
        ]

    })

    st.dataframe(
        kpi,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("✨ Features of the System")

    st.markdown("""
✅ Interactive Dashboard

✅ Delay Prediction

✅ Region Analysis

✅ Model Performance Evaluation

✅ Operations Action Panel

✅ Download Reports

✅ Feature Importance Analysis
""")

    st.markdown("---")

    st.subheader("👩‍💻 Developed By")

    st.success("""
Sanjana D

Bachelor of Computer Applications (BCA)

Machine Learning Project

APL Logistics (KWE Group)
""")
    
