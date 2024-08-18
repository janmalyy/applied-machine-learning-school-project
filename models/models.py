import pandas as pd
import numpy as np
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense


def load_and_split_data(filepath, train_pct, val_pct):
    # load the dataset
    stocks_df = pd.read_csv(filepath)

    # train-validation-test split
    feature_names = ['5d_close_pct', 'ma14', 'rsi14', 'ma3', 'rsi3']
    features = stocks_df[feature_names]
    targets = stocks_df['5d_future_close_pct']

    train_size = int(train_pct * targets.shape[0])
    val_size = int(val_pct * targets.shape[0])
    train_features = features[:train_size]
    train_targets = targets[:train_size]
    val_features = features[train_size:train_size+val_size]
    val_targets = targets[train_size:train_size+val_size]
    test_features = features[train_size+val_size:]
    test_targets = targets[train_size+val_size:]

    return train_features, train_targets, val_features, val_targets, test_features, test_targets


def reshape_to_3D(array_2D):
    """
    3D format (samples, timesteps, features) is expected by RNN, LSTM and Transformer models
    2D format (samples, features) is expected by dummy and Linear models
    """
    return np.array(array_2D).reshape(array_2D.shape[0], 1, array_2D.shape[1])

#####################################################
################# RUN MODELS ########################
#####################################################

def run_dummy_model(train_features, train_targets, test_features):
    # Define the dummy model
    dummy_model = DummyRegressor(strategy='mean')

    # Fit the dummy model
    dummy_model.fit(train_features, train_targets)

    # Make predictions
    dummy_train_predict = dummy_model.predict(train_features)
    dummy_test_predict = dummy_model.predict(test_features)

    return dummy_train_predict, dummy_test_predict

def run_linear_model(train_features, train_targets, test_features):
    # Define the linear model
    model = LinearRegression()

    # Fit the model
    model.fit(train_features, train_targets)

    # Make predictions
    train_predict = model.predict(train_features)
    test_predict = model.predict(test_features)

    return model, train_predict, test_predict

def run_rnn_model(train_features, train_targets, test_features, units=128, epochs=512):
    # Reshape
    train_features = reshape_to_3D(train_features)
    test_features = reshape_to_3D(test_features)

    # Define the RNN model
    model = Sequential()
    model.add(SimpleRNN(units, activation='relu', input_shape=(train_features.shape[1], train_features.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Fit the model
    history = model.fit(train_features, train_targets, epochs=epochs, verbose=1)

    # Make predictions
    train_predict = model.predict(train_features)
    test_predict = model.predict(test_features)

    return model, history, train_predict, test_predict

def run_lstm_model(train_features, train_targets, test_features, units=256, epochs=256):
    # Reshape
    train_features = reshape_to_3D(train_features)
    test_features = reshape_to_3D(test_features)

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(units, activation='relu', input_shape=(train_features.shape[1], train_features.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Fit the model
    history = model.fit(train_features, train_targets, epochs=epochs, verbose=1)

    # Make predictions
    train_predict = model.predict(train_features)
    test_predict = model.predict(test_features)

    return model, history, train_predict, test_predict


def run_complex_lstm_model(train_features, train_targets, test_features, epochs=256):
    # Reshape
    train_features = reshape_to_3D(train_features)
    test_features = reshape_to_3D(test_features)

    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(train_features.shape[1], train_features.shape[2]),
                   return_sequences=True))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss='mse')

    # Fit the model
    history = model.fit(train_features, train_targets, epochs=epochs, verbose=1)

    # Make predictions
    train_predict = model.predict(train_features)
    test_predict = model.predict(test_features)

    return model, history, train_predict, test_predict


# imports for transformer model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def run_transformer_model(train_features, train_targets, test_features, embed_dim=32, num_heads=2, ff_dim=32, epochs=512):
    # Reshape
    train_features = reshape_to_3D(train_features)
    test_features = reshape_to_3D(test_features)

    # Define the Transformer model
    inputs = Input(shape=(train_features.shape[1], train_features.shape[2]))
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attention_output = Dense(embed_dim)(attention_output)  # Add this line
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    ffn_output = Dense(ff_dim, activation='relu')(attention_output)
    ffn_output = Dense(embed_dim)(ffn_output)
    ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)
    outputs = Dense(1)(ffn_output)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='mse')

    # Fit the model
    history = model.fit(train_features, train_targets, epochs=epochs, verbose=1)

    # Make predictions
    train_predict = model.predict(train_features).flatten()
    test_predict = model.predict(test_features).flatten()

    return model, history, train_predict, test_predict

#####################################################
############ EVALUTATIONS AND GRAPHS ################
#####################################################

def calculate_error(targets, predictions):
    # Calculate MSE; the most basic metrics
    mse = mean_squared_error(targets, predictions)
    return mse


def plot_histories(histories, labels):
    """
    Plot the training loss history of multiple models on a single graph.
    @param histories: A list of training history objects for each model.
    @param labels: A list of labels corresponding to each model in 'histories'.
    @returns None: This function does not return anything, it simply displays the plot.
    """
    plt.figure(figsize=(14,5))
    for i, history in enumerate(histories):
        plt.plot(history.history['loss'], label=labels[i])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.yscale('log')
    plt.legend()
    plt.show()


def plot_predictions(train_targets, test_targets, test_features, predictions, labels):
    # Plot the real vs predicted values
    plt.figure(figsize=(14,5))

    # # Plot the input features
    # features = ['5d_close_pct', 'ma14', 'rsi14', 'ma3', 'rsi3']
    # for i, feature in enumerate(features):
    #     plt.plot(np.arange(len(train_targets), len(train_targets) + len(test_targets)), test_features[test_features.columns[i]], label=feature, alpha=0.3)

    # Plot the real values
    plt.plot(np.arange(len(train_targets), len(train_targets) + len(test_targets)), test_targets, 'b', label="true", alpha=0.8)

    # Plot each prediction
    for prediction, label in zip(predictions, labels):
        plt.plot(np.arange(len(train_targets), len(train_targets) + len(test_targets)), prediction, label=label, alpha=0.8)

    plt.ylabel('Value')
    plt.xlabel('Time Step')
    plt.legend()
    plt.show()


def moving_average_crossover_strategy(predictions, short_window=40, long_window=100):
    """
    @author of moving_average_crossover_strategy and plots below:
        https://zodiactrading.medium.com/top-10-quantitative-trading-strategies-with-python-82b1eff67650
    @returns: signals when to buy and sell
    """
    signals = pd.DataFrame(index=predictions.index)
    signals['signal'] = 0.0

    # Create short simple moving average
    signals['short_mavg'] = predictions.rolling(window=short_window, min_periods=1, center=False).mean()

    # Create long simple moving average
    signals['long_mavg'] = predictions.rolling(window=long_window, min_periods=1, center=False).mean()

    # Create signals
    signals['signal'][short_window:] = np.where(
        signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    return signals


def buy_sell(positions, opening_prices, n=10):
    """
    algo, which buys and sells stocks given positions signals
    returns the resulting difference of number of stocks in possession and how much money was gained or lost
    it sells/buys on the opening price of stock at the day
    @param positions: when to buy/sell
    @param opening_prices: the price of stocks at the corresponding time
    @param n: how many stocks are bought/sold in one signal
    """
    stocks_hold = 0
    money = 0
    buy_counter = 0
    sell_counter = 0
    for i in range(len(positions)):
        if positions[i] == 1.0:  # buy
            stocks_hold += n
            money -= n * opening_prices[i]
            buy_counter += 1
        elif positions[i] == -1.0:  # sell
            stocks_hold -= n
            money += n * opening_prices[i]
            sell_counter += 1
    return stocks_hold, round(money, 2), buy_counter, sell_counter


def get_corresponding_prices_and_dates(filepath, train_pct, val_pct, part_of_data):
    """
    chtěl jsem to předělat do funkcí, ale nevím, jestli tohle je nejlepší řešení...
    """
    stocks_df = pd.read_csv(filepath)
    train_size = int(train_pct * stocks_df["Open"].shape[0])
    val_size = int(val_pct * stocks_df["Open"].shape[0])

    if part_of_data == "train":
        opening_prices = stocks_df["Open"][:train_size]
        dates = stocks_df["Date"][:train_size]
    elif part_of_data == "val":
        opening_prices = stocks_df["Open"][train_size:train_size + val_size]
        dates = stocks_df["Date"][train_size:train_size + val_size]
    elif part_of_data == "test":
        opening_prices = stocks_df["Open"][train_size + val_size:]
        dates = stocks_df["Date"][train_size + val_size:]
    else:
        raise ValueError(f'Not valid argument: "{part_of_data}", try train, val or test instead.')

    return opening_prices, dates


def eval_the_strategy(signals, opening_prices):
    """
    @param signals: dataframe which determines when to buy and sell the stocks
    @param opening_prices: array with corresponding opening prices for dates in signals dataframe
    @returns: None, prints the results of buy_sell function
    """
    positions = signals["positions"].tolist()  # using a list is faster and easier than the column in dataframe

    stocks_hold, money, buy_counter, sell_counter = buy_sell(positions, opening_prices.tolist())
    print(f"stock hold: {stocks_hold}\n" +
          f"money: {money}\n" +
          f"buy counter: {buy_counter}\n" +
          f"sell counter: {sell_counter}")


def plot_the_signals(signals, targets, short_window=40, long_window=100):
    """
    @param signals: dataframe which determines when to buy and sell the stocks
    @param targets: the real values, expected to be: train_targets, val_targets or test_targets
    @returns: None, plots the buy and sell signals in graph with target, short and long Mavg curves on y-axis and date on x-axis
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(signals.index, targets, label='5d_future_close_pct')
    ax.plot(signals.index, signals['short_mavg'], label=f'Short {short_window} days Mavg')
    ax.plot(signals.index, signals['long_mavg'], label=f'Long {long_window} days Mavg')
    plt.xticks(np.arange(0, len(signals.index), 80), rotation=35)  # show only some x-labels

    # Plotting buy signals
    ax.plot(signals.loc[signals.positions == 1.0].index,
            signals.short_mavg[signals.positions == 1.0],
            '^', markersize=10, color='g', label='Buy Signal')

    # Plotting sell signals
    ax.plot(signals.loc[signals.positions == -1.0].index,
            signals.short_mavg[signals.positions == -1.0],
            'v', markersize=10, color='r', label='Sell Signal')

    plt.title('Moving Average Crossover Strategy')
    plt.xlabel('Date')
    plt.ylabel('5d_future_close_pct')
    plt.legend()
    plt.show()

#####################################################
################## CODE EXECUTION ###################
#####################################################
filepath = r'C:\Users\haemk\applied-machine-learning-project-2024\models\nee_stocks_with_features.csv'
train_pct = 0.6
val_pct = 0.2

train_features, train_targets, val_features, val_targets, test_features, test_targets = load_and_split_data(filepath, train_pct, val_pct)

# !!! IMPORTANT NOTE !!!: test dataset is used for running and evaluating the model for now (even though sometimes it is called val_something)

# # Run the dummy model
# train_predict_dummy, val_predict_dummy = run_dummy_model(train_features, train_targets, test_features)

# Run the Linear model
# model_linear, train_predict_linear, val_predict_linear = run_linear_model(train_features, train_targets, test_features)

# Run the RNN model
# model_rnn, history_rnn, train_predict_rnn, val_predict_rnn = run_rnn_model(train_features, train_targets, test_features)

# Run the LSTM model
# model_lstm, history_lstm, train_predict_lstm, val_predict_lstm = run_lstm_model(train_features, train_targets, test_features)

# Run the complex LSTM model
# model_complex, history_complex, train_predict_complex, val_predict_complex = run_complex_lstm_model(train_features, train_targets, test_features)

# Run the Transformer model
model_transformer, history_transformer, train_predict_transformer, val_predict_transformer = run_transformer_model(train_features, train_targets, test_features)

# Plot histories of LSTM, Complex LSTM, RNN and Transformer
# plot_histories([history_lstm, history_complex, history_rnn, history_transformer], ['LSTM', "Complex LSTM", 'RNN', 'Transformer'])
# ------------------
# nefunkční pokus o uložení predictions do txt
# model_names = {"Linear": val_predict_linear,
#                # 'LSTM': val_predict_lstm,
#                "Complex LSTM": val_predict_complex}
#                # 'RNN':val_predict_rnn, 'Transformer':val_predict_transformer}
#
# for model_name in model_names.keys():
#     with open(f"{model_name}_predictions_on_test_set.txt", "w", encoding="utf-8") as f:
#         f.writelines(str(model_names[model_name]))

# ------------------
# "aby v tom výslednom grafe bolo lepšie vidieť rozdiely;
# teoreticky to tam vôbec byť nemusí a možno existuje aj dáke krajšie riešenie - možno iba zmeniť škálu v grafe."
# Define the scaler
# scaler = MinMaxScaler(feature_range=(-1, 1))
#
# # Multiply the targets by 10
# train_targets = train_targets * 10
# val_targets = val_targets * 10
# val_predict_dummy = val_predict_dummy * 10
# val_predict_rnn = val_predict_rnn * 10
# val_predict_ = val_predict_lstm * 10
# val_predict_complex = val_predict_complex * 10
# val_predict_transformer = val_predict_transformer * 10
#
# # Fit the scaler to the training data and transform both the training and val data
# train_features = scaler.fit_transform(train_features)
# val_features = scaler.transform(val_features)

# Calculate and print MSE
# train_mse_dummy = calculate_error(train_targets, train_predict_dummy)
# print(f"Dummy Model Train MSE: {train_mse_dummy}")
# dummy_val_mse = calculate_error(test_targets, val_predict_dummy)
# print(f"Dummy Model val MSE: {dummy_val_mse}")

# train_mse_linear = calculate_error(train_targets, train_predict_linear)
# print(f"Linear Model Train MSE: {train_mse_linear}")
# val_mse_linear = calculate_error(test_targets, val_predict_linear)
# print(f"Linear Model val MSE: {val_mse_linear}")
#
# train_mse_rnn = calculate_error(train_targets, train_predict_rnn)
# print(f"RNN Model Train MSE: {train_mse_rnn}")
# val_mse_rnn = calculate_error(test_targets, val_predict_rnn)
# print(f"RNN Model val MSE: {val_mse_rnn}")
#
# train_mse_lstm = calculate_error(train_targets, train_predict_lstm)
# print(f"LSTM Model Train MSE: {train_mse_lstm}")
# val_mse_lstm = calculate_error(test_targets, val_predict_lstm)
# print(f"LSTM Model val MSE: {val_mse_lstm}")

# train_mse_complex = calculate_error(train_targets, train_predict_complex)
# print(f"Complex Model Train MSE: {train_mse_complex}")
# val_mse_complex = calculate_error(test_targets, val_predict_complex)
# print(f"Complex Model val MSE: {val_mse_complex}")

# train_mse_transformer = calculate_error(train_targets, train_predict_transformer)
# print(f"Transformer Model Train MSE: {train_mse_transformer}")
# val_mse_transformer = calculate_error(test_targets, val_predict_transformer)
# print(f"Transformer Model val MSE: {val_mse_transformer}")

# # Call the function with multiple predictions
# plot_predictions(train_targets, test_targets, test_features,
#                  # [val_predict_complex],
#                  [val_predict_complex, val_predict_lstm, val_predict_rnn, val_predict_linear, val_predict_transformer],
#                  # ["Complex prediction"])
#                  ["Complex prediction", "LSTM prediction", "RNN prediction", "Linear prediction", "Transformer prediction"])

# -------------------------------------------------------------
# evaluation using moving average crossover strategy
part_of_data = "test"
predictions = val_predict_transformer
targets = test_targets

opening_prices, dates = get_corresponding_prices_and_dates(filepath, train_pct, val_pct, part_of_data)
predicted_series = pd.Series([float(prediction) for prediction in predictions], index=dates)        # the conversion to float is because of the fact that the model predictions (for lstm, rnn and transformer) are stored as list or something like that
signals = moving_average_crossover_strategy(predicted_series)

eval_the_strategy(signals, opening_prices)
plot_the_signals(signals, targets)
