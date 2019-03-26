from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt 
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
 
# load dataset
# period1-trp for period 1
# the whole code is predicting the pressure, the dataset is [Time, Rate Pressure]
dataframe = read_csv('path')
dataset = dataframe.values
dataset = dataset.astype('float32')
print(len(dataset))

#uncomment this section and just run again, if you want to predict the rate
'''
#creating 3 separate arrays for each value type
time = dataset[:,0]
rate = dataset[:,1]
press = dataset[:,2]

#reshaping to [None, 1] for concatenation
time = time.reshape((-1,1))
rate = rate.reshape((-1,1))
press = press.reshape((-1,1))

#concatenate to create the new [Time, Pressure, Rate] dataset
dataset = concatenate((time, press, rate), axis=1)
'''
#create scaler and scale all values to the range (0,1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(dataset)

# create train and test sets
split_point = 2022 #2022 for period1, 2638 for period2
train = scaled[:split_point, :]
test = scaled[split_point:, :]
# separate the inputs X and outputs y
train_X, train_Y = train[:, :-1], train[:, -1]
test_X, test_Y = test[:, :-1], test[:, -1]
# reshape input to be 3D tensor [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print("Train_X shape:", train_X.shape) 
print("Train_Y shape:", train_Y.shape)
print("Test_X shape:", test_X.shape)
print("Test_Y shape:", test_Y.shape)

# build LSTM network
model = Sequential()
model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# model.fit will match the training inputs X to the training outputs (labels) y
history = model.fit(train_X, train_Y, nb_epoch=10, batch_size=10, validation_data=(test_X, test_Y), verbose=2, shuffle=False)
# plot the learning
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Testing loss')
plt.grid(True)
plt.legend()
plt.xlabel('Numer of iterations')
plt.ylabel('Mean Absolute Error')
plt.show()

# make predictions
ytrain = model.predict(train_X)
y_pred = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))

#compute RMSE
rmse = sqrt(mean_squared_error(test_Y, y_pred))
print('Test RMSE: %.3f' % rmse)

#invert scaling for train predictions
inv_ytrain = concatenate((ytrain, train_X), axis=1)
inv_ytrain = scaler.inverse_transform(inv_ytrain)
inv_ytrain = inv_ytrain[:,0]
# invert scaling for test predictions
inv_y_pred = concatenate((y_pred, test_X), axis=1)
inv_y_pred = scaler.inverse_transform(inv_y_pred)
inv_y_pred = inv_y_pred[:,0]
# invert scaling for actual
test_Y = test_Y.reshape((len(test_Y), 1))
inv_y = concatenate((test_Y, test_X), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

#period1-trp for period 1
time_test = read_csv('path', usecols = [0])
time_test = time_test.values
time_test  = time_test[2022:]

pressure_test = read_csv('path', usecols = [1])
pressure_test = pressure_test.values
pressure_test = pressure_test[2022:]

rate_test = read_csv('path', usecols = [2])
rate_test = rate_test.values
rate_test = rate_test[2022:]

time_train = read_csv('path', usecols = [0])
time_train = time_train.values
time_train  = time_train[:2022]

pressure_train = read_csv('path', usecols = [1])
pressure_train = pressure_train.values
pressure_train = pressure_train[:2022]

rate_train = read_csv('path', usecols = [2])
rate_train = rate_train.values
rate_train = rate_train[:2022]

plt.plot(time_train, pressure_train, 'g')
plt.plot(time_train, rate_train, 'b')
plt.plot(time_test, rate_test, 'b')
plt.plot(time_test, pressure_test, 'g', label = 'Pressure')
plt.plot(time_test, inv_y_pred, 'r', label = 'Test pred')
plt.plot(time_train, inv_ytrain, 'k', label = 'Train pred')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
plt.grid(True)
plt.xlim([0,3000])
plt.title("True values vs Predicted values")
plt.xlabel('Time')
plt.ylabel('Rate and pressure')
plt.show()
