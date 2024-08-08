import time

import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

btc = yf.download("BTC-USD", start="2015-01-01", end="2024-07-30")
# eth = yf.download("ETH-USD", start="2015-01-01", end="2024-07-30")
# gold = yf.download("GOLD", start="2015-01-01", end="2024-07-30")

# Feature Extraction
btc.dropna(inplace=True)

btc["Benefit"] = btc["Close"] - btc["Open"]
btc["Tomorrow"] = btc["Close"].shift(-1)
# btc["SMA14"] = btc["Close"].rolling(14).mean()
# btc["SMA21"] = btc["Close"].rolling(21).mean()

# PreProcessing
btc.dropna(inplace=True)

# X = btc[["Open", "Close", "Low", "High", "SMA-14", "SMA-21"]]
X = btc[["Open", "Close", "Low", "High"]]
y = btc["Tomorrow"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# model = LinearRegression(n_jobs=-1)
model = Lasso()
# model = Ridge()

result_list= []

for model in [LinearRegression(), Lasso(), Ridge()]:
    start = time.time()
    # print(model)
    # Train
    model.fit(x_train, y_train)

    # Predict
    all_pred = model.predict(X)  # y_hat , h
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # print(model.predict([[66201.27, 64619.25, 64532.05, 66810.21, 31292785994]]))    # 64571  65500
    # print(model.predict([[66201.27, 64619.25, 64532.05, 66810.21]]))                 # 64571  64499
    # print(model.predict([[66201.27, 64619.25, 64532.05, 66810.21]]))                 # 64571  64486

    # Evaluate
    all_rmse = root_mean_squared_error(y, all_pred)
    # print("All RMSE :", all_rmse)

    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    # print("Train RMSE :", train_rmse)

    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    # print("Test RMSE :", test_rmse)
    end = time.time()

    # print(end - start, "Second")

    result = {
        "model": f"{str(model).replace("(", "").replace(")","")}",
        "all_rmse": all_rmse,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "time": end - start
    }
    result_list.append(result)

    # Visualize
    # plt.plot(X.index, y, label="Close Price")
    # plt.plot(X.index, all_pred, label="Tomorrow Close Price Predict")
    #
    # plt.show()

result_df = pd.DataFrame(result_list)
print(result_df)

# All RMSE : 1156.3583763177855
# Train RMSE : 1008.0580862368935
# Test RMSE : 1618.9808681671232

# All RMSE : 1156.3583762829535
# Train RMSE : 1008.0580862368938
# Test RMSE : 1618.9808680427284

# All RMSE : 1155.8067785618161
# Train RMSE : 1008.5490430099893
# Test RMSE : 1615.7852141363821

# All RMSE : 877.8626779412676
# Train RMSE : 817.2626356706966
# Test RMSE : 1086.7848431912846

# All RMSE : 897.9505399036326
# Train RMSE : 835.0217333297204
# Test RMSE : 1114.4789393594463

plt.subplot(2,1,1)
plt.plot(result_df["model"], result_df["all_rmse"], label="All")
plt.plot(result_df["model"],result_df["train_rmse"], label="Train")
plt.plot(result_df["model"],result_df["test_rmse"], label="Test")
plt.legend()


plt.subplot(2,1,2)
plt.plot(result_df["model"],result_df["time"], label="Time")

plt.legend()
plt.show()