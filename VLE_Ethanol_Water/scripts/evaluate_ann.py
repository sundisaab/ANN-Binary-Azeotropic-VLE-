import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import losses, metrics  # type: ignore

 
# 1. Load Model and Scalers
 
model = tf.keras.models.load_model(
    "VLE_Ethanol_Water/results/best_ann_model.h5",
    custom_objects={
        "mse": losses.MeanSquaredError(),
        "mae": metrics.MeanAbsoluteError(),
    }
)

scaler_X = joblib.load("VLE_Ethanol_Water/results/scaler_X.gz")
scaler_y = joblib.load("VLE_Ethanol_Water/results/scaler_y.gz")

 
# 2. Load Dataset
 
df = pd.read_csv("VLE_Ethanol_Water/data/vle_data.csv")
X = df[["x_ethanol", "Tbub"]].values  # inputs
y = df["y_ethanol"].values.reshape(-1, 1)  # target reshaped for scaler

# Normalize with same scalers
X_scaled = scaler_X.transform(X)
y_true = scaler_y.inverse_transform(scaler_y.transform(y))  # keep same scale as pred

 
# 3. Predictions
 
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
 
 
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("🔹 ANN Results:")
print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")

 
# 5. Azeotrope Detection (ANN)
 
diff = np.abs(df["x_ethanol"].values - y_pred.flatten())
aze_index = np.argmin(diff)
aze_x = df["x_ethanol"].iloc[aze_index]
aze_y = y_pred[aze_index][0]
aze_T = df["Tbub"].iloc[aze_index]

print("\n🔹 Azeotrope (from ANN):")
print(f"x_ethanol ≈ {aze_x:.4f}, y_ethanol ≈ {aze_y:.4f}, T ≈ {aze_T:.2f} K")

 
# 6. Raoult's Law Baseline
 
P_total = 101.325  # kPa (1 atm)

def Psat_ethanol(T):
    A, B, C = 8.20417, 1642.89, 230.3
    T_C = T - 273.15
    return 10**(A - B / (T_C + C))

def Psat_water(T):
    A, B, C = 8.07131, 1730.63, 233.426
    T_C = T - 273.15
    return 10**(A - B / (T_C + C))

def raoult_y(x, T):
    P_eth = Psat_ethanol(T) / 760 * 101.325
    P_wat = Psat_water(T) / 760 * 101.325
    return (x * P_eth) / (x * P_eth + (1 - x) * P_wat)

y_raoult = [raoult_y(x, T) for x, T in zip(df["x_ethanol"], df["Tbub"])]

# Detect azeotrope (Raoult)
diff_r = np.abs(df["x_ethanol"].values - np.array(y_raoult))
aze_index_r = np.argmin(diff_r)
aze_x_r = df["x_ethanol"].iloc[aze_index_r]
aze_y_r = y_raoult[aze_index_r]
aze_T_r = df["Tbub"].iloc[aze_index_r]

print("\n🔹 Azeotrope (Raoult’s Law, ideal):")
print(f"x_ethanol ≈ {aze_x_r:.4f}, y_ethanol ≈ {aze_y_r:.4f}, T ≈ {aze_T_r:.2f} K")

 
# 7. Plots
 

# Parity plot
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.6, label="ANN Predictions")
plt.plot([0,1],[0,1],"k--",label="Ideal")
plt.xlabel("y_exp")
plt.ylabel("y_pred")
plt.title("Parity Plot: ANN")
plt.legend()
plt.grid()
plt.savefig("VLE_Ethanol_Water/results/parity_plot.png", dpi=300, bbox_inches="tight")
plt.close()

# VLE curve plot 
plt.figure(figsize=(6,6))
plt.plot(df["x_ethanol"], y_true, "o", alpha=0.4, label="Experimental/Simulated")
plt.plot(df["x_ethanol"], y_pred, "-", label="ANN")
plt.plot(df["x_ethanol"], y_raoult, "--", label="Raoult’s Law (ideal)")
plt.axvline(aze_x, color="r", linestyle=":", label="ANN Azeotrope")
plt.axvline(aze_x_r, color="b", linestyle=":", label="Raoult Azeotrope")
plt.xlabel("x_ethanol (liquid)")
plt.ylabel("y_ethanol (vapor)")
plt.title("VLE Curve: Ethanol-Water")
plt.legend()
plt.grid()
plt.savefig("VLE_Ethanol_Water/results/vle_curve.png", dpi=300, bbox_inches="tight")
plt.close()

print("\n Plots saved in: VLE_Ethanol_Water/results/")
