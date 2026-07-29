import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Load Dataset
df = pd.read_csv("data/APL_Logistics.csv", encoding="latin1")

print(df.shape)
print(df.head())
drop_columns = [
    "Customer Fname",
    "Customer Lname",
    "Customer Street",
    "Customer Zipcode",
    "Latitude",
    "Longitude",
    "Customer Id",
    "Order Customer Id",
    "Category Id",
    "Department Id"
]

df.drop(columns=drop_columns, inplace=True, errors="ignore")
label_encoders = {}

categorical_columns = df.select_dtypes(include="object").columns

for col in categorical_columns:

    if col != "Late_delivery_risk":

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col].astype(str))

        label_encoders[col] = le
        # ==========================
# Feature Engineering
# ==========================

df["Shipping_Delay"] = (
    df["Days for shipping (real)"] -
    df["Days for shipment (scheduled)"]
)

df["Shipping_Pressure_Index"] = (
    df["Order Item Quantity"] /
    df["Days for shipment (scheduled)"]
)

df["Order_Complexity_Score"] = (
    df["Order Item Quantity"] *
    df["Product Price"]
)

df["Profit_Margin"] = (
    df["Order Profit Per Order"] /
    df["Sales"].replace(0, np.nan)
)

# Replace infinity values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaN values
df.fillna(0, inplace=True)

df["Express_Shipping"] = df["Shipping Mode"].isin([0, 1]).astype(int)
# ==========================
# Remove Leakage Features
# ==========================

leakage_columns = [
    "Delivery Status",
    "Days for shipping (real)",
    "Shipping_Delay"
]

df.drop(columns=leakage_columns, inplace=True, errors="ignore")
# ==========================
# Features & Target
# ==========================

# ==========================
# Features & Target
# ==========================

X = df.drop("Late_delivery_risk", axis=1)
y = df["Late_delivery_risk"]

# Convert all columns to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Remove Infinity
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaN
X.fillna(0, inplace=True)

print(X.shape)
print("Infinity:", np.isinf(X).sum().sum())
print("NaN:", X.isna().sum().sum())

# ==========================
# Train Test Split
# ==========================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# Scaling
# ==========================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=15,
    class_weight="balanced",
    n_jobs=-1
)
print("Starting model training...")

model.fit(X_train_scaled, y_train)
print("Model training completed!")

pred = model.predict(X_test_scaled)
# ==========================
# Accuracy
# ==========================

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy)
# ==========================
# Save Files
# ==========================

import joblib

joblib.dump(model, "models/best_model.pkl", compress=3)
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

X.to_csv("data/model_features.csv", index=False)

print("✅ Everything Saved Successfully!")