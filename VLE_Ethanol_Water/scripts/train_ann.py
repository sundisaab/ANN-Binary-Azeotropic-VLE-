import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks # type: ignore
import joblib
import os

# 1. Load Dataset
df = pd.read_csv("VLE_Ethanol_Water/data/vle_data.csv")

# Use x_ethanol and Tbub as input features
X = df[["x_ethanol", "Tbub"]].values   
y = df["y_ethanol"].values            

# 2. Normalize Inputs/Outputs
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# 3. Build ANN Model
def build_model():
    model = models.Sequential([
        layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")  # output between 0–1
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss="mse",
                  metrics=["mae"])
    return model

model = build_model()

# 4. Training
early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
lr_scheduler = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=1e-6)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=1000,
    batch_size=32,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# 5. Evaluation
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_true = scaler_y.inverse_transform(y_test)

mae = mean_absolute_error(y_test_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_true, y_pred))
r2 = r2_score(y_test_true, y_pred)

print(f"MAE: {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R2: {r2:.6f}")

# 6. Parity Plot
plt.figure(figsize=(6,6))
plt.scatter(y_test_true, y_pred, alpha=0.7, label="Predictions")
plt.plot([0,1], [0,1], "k--", label="Ideal Fit")
plt.xlabel("y_exp (Experimental)")
plt.ylabel("y_pred (ANN)")
plt.title("Parity Plot: y_exp vs y_pred")
plt.legend()
plt.grid(True)

results_dir = "VLE_Ethanol_Water/results"
os.makedirs(results_dir, exist_ok=True)   

plt.savefig(os.path.join(results_dir, "parity_plot.png"), dpi=300)
plt.show()

#  7. Loss Curve Plot (NEW)
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training & Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, "loss_curve.png"), dpi=300)  # save image
plt.show()

# 8. Save Model & Scalers
model.save(os.path.join(results_dir, "best_ann_model.h5"))
joblib.dump(scaler_X, os.path.join(results_dir, "scaler_X.gz"))
joblib.dump(scaler_y, os.path.join(results_dir, "scaler_y.gz"))

print("Model and scalers saved in:", results_dir)
print("Plots saved: parity_plot.png, loss_curve.png")
