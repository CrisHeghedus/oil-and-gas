import shutil
import numpy as np
from numpy import newaxis
from numpy import concatenate
from math import sqrt
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.layers import LSTM, Activation, Dropout
import keras.layers
 
#shift down to create lagged data
def create_lag(data, n_in=1, n_out=1, dropnan=True):
	n_var = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_var)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_var)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_var)]
	merged = concat(cols, axis=1)
	merged.columns = names
	# drop rows with NaN values
	if dropnan:
		merged.dropna(inplace=True)
	return merged
  
  
dataset = read_csv('path', header=1)
values = dataset.values
values = values.astype('float32')


#create lagged data (in other word add the previous timestep to the curent input)
reframed = create_lag(values, 1, 1)
reframed.drop(reframed.columns[[1,2,4,5,6,9,10,12,13,14]], axis=1, inplace=True)
print(reframed.head())

#arrange for pressure prediction
reframed = reframed[[0,1,3,4,2,5]]
print(reframed.head())

#arrange for rate prediction
reframed = reframed[[0,2,3,5,1,4]]
print(reframed.head())

#create scaler and scale all values to the range (0,1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(reframed)
scaled = DataFrame(data=scaled)

# create train and test sets
values = scaled.values
split_point = int(len(values)*.75) 
train = values[2:split_point, :]
test = values[split_point+4:, :]

# separate the inputs X and outputs y
train_X, train_Y = train[:, :-1], train[:, -1]
test_X, test_Y = test[:, :-1], test[:, -1]

# reshape input to be 3D tensor [samples, timesteps, features]
train_X = train_X.reshape((202, 10, train_X.shape[1]))
test_X = test_X.reshape((67, 10, test_X.shape[1]))
train_Y = train_Y.reshape((202,10))
test_Y = test_Y.reshape((67,10))

print("Train_X shape:", train_X.shape) 
print("Train_Y shape:", train_Y.shape)
print("Test_X shape:", test_X.shape)
print("Test_Y shape:", test_Y.shape)

# build LSTM network
model = Sequential()
model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(10))
model.compile(loss='mae', optimizer='adam')
#model.summary()
# model.fit will match the training inputs X to the training outputs (labels) y
history = model.fit(train_X, train_Y, nb_epoch=10, batch_size=1, validation_split = 0.2, verbose=2, shuffle=False)

# plot the learning
plt.plot(history.history['loss'], label='Training loss')
plt.grid(True)
plt.legend()
plt.xlabel('Numer of iterations')
plt.ylabel('Mean Absolute Error')
plt.show()

#make predictions
preds = model.predict(test_X)
preds_train_10 = model.predict(train_X)

#reshape for plotting
test_X = test_X.reshape((670, 5))
test_Y = test_Y.reshape((670,1))
preds = preds.reshape((670,1))

#reshape for concatenation
train_X = train_X.reshape((2020,5))
train_Y = train_Y.reshape((2020,1))
preds_train_10 = preds_train_10.reshape(2020,1)

#concatenate
inv_y_train_10 = concatenate((train_X[:, 0:], preds_train_10 ), axis=1)
inv_y_train_10 = scaler.inverse_transform(inv_y_train_10)
inv_y_train_10 = inv_y_train_10[:,5]

inv_y_pred = concatenate((test_X[:, 0:], preds ), axis=1)
inv_y_pred = scaler.inverse_transform(inv_y_pred)
inv_y_pred = inv_y_pred[:,5]

############ plot here

############ predict based on predicitons 
train_X = train_X.reshape((1, 2020, 5))
test_X = test_X.reshape((1, 670, 5))

train_Y = train_Y.reshape((202,10))
test_Y = test_Y.reshape((67,10))

list_of_pred = []
for i in range(659):
    k = i+10
    moving_window = [test_X[:,i:k,:].tolist()]
    moving_window = np.array(moving_window).reshape((1,10,5))   
    pred_one_row = model.predict(moving_window)
    
    curr_pred = pred_one_row[:,9]
    list_of_pred.append(curr_pred)
    curr_pred = curr_pred.reshape((1,1))
    last_row = test_X[:,k+1,:4] # has shape (1,4)
    next_row = np.concatenate((last_row, curr_pred), axis = 1) #shape (1,5)
    next_row = next_row.reshape((1,1,5))
    moving_window = np.concatenate((moving_window[:,1:,:], next_row), axis = 1)
    #print(moving_window)
#print(list_of_pred)

list_pred_train = []
for i in range(2009):
    k = i+10
    moving_window_train = [train_X[:,i:k,:].tolist()]
    moving_window_train = np.array(moving_window_train).reshape((1,10,5))   
    pred_one_row_train = model.predict(moving_window_train)
    
    curr_pred_train = pred_one_row_train[:,9]
    list_pred_train.append(curr_pred_train)
    curr_pred_train = curr_pred.reshape((1,1))
    last_row_train = train_X[:,k+1,:4] # has shape (1,4)
    next_row_train = np.concatenate((last_row_train, curr_pred_train), axis = 1) #shape (1,5)
    next_row_train = next_row_train.reshape((1,1,5))
    moving_window_train = np.concatenate((moving_window_train[:,1:,:], next_row_train), axis = 1)

#predict based on predictions
list_pred = []
test_X = test_X.reshape((1,670,5))
window = [test_X[:,:10,:].tolist()]
window = np.array(window).reshape((1,10,5))

for i in range(0,659):
    pred_one_step = model.predict(window)
    list_pred.append(pred_one_step[:,9])
    curr_pred = list_pred[-1]
    curr_pred = curr_pred.reshape((1,1))
    last_line = np.concatenate((test_X[:,i+10,:4], curr_pred), axis = 1)
    last_line = last_line.reshape((1,1,5))
    window = np.concatenate((window[:,1:,:], last_line), axis = 1)
    #print(window)
    
    
list_train = []
train_X = train_X.reshape((1,2020,5))
window = [train_X[:,:10,:].tolist()]
window = np.array(window).reshape((1,10,5))

for i in range(0,2009):
    pred_one_step = model.predict(window)
    list_train.append(pred_one_step[:,9])
    curr_pred = list_pred[-1]
    curr_pred = curr_pred.reshape((1,1))
    last_line = np.concatenate((train_X[:,i+10,:4], curr_pred), axis = 1)
    last_line = last_line.reshape((1,1,5))
    window = np.concatenate((window[:,1:,:], last_line), axis = 1)
    #print(window)


list_of_pred = np.array(list_of_pred).reshape(659,1)
list_pred_train = np.array(list_pred_train).reshape(2009,1)

test_X = test_X.reshape((670,5))
train_X = train_X.reshape((2020,5))

#concatenate
inv_total_train = concatenate((train_X[11:, 0:], list_pred_train ), axis=1)
inv_total_train = scaler.inverse_transform(inv_total_train)
inv_total_train = inv_total_train[:,5]

inv_total_pred = concatenate((test_X[11:, 0:], list_of_pred ), axis=1)
inv_total_pred = scaler.inverse_transform(inv_total_pred)
inv_total_pred = inv_total_pred[:,5]




#plot
time_train = read_csv('path', usecols = [0])
time_train = time_train.values
time_train  = time_train[11:2020]

pressure_train = read_csv('path', usecols = [7])
pressure_train = pressure_train.values
pressure_train = pressure_train[11:2020]

rate_train = read_csv('path', usecols = [3])
rate_train = rate_train.values
rate_train = rate_train[11:2020]

time_test = read_csv('path', usecols = [0])
time_test = time_test.values
time_test  = time_test[2039:]

pressure_test = read_csv('path', usecols = [7])
pressure_test = pressure_test.values
pressure_test = pressure_test[2039:]

rate_test = read_csv('path', usecols = [3])
rate_test = rate_test.values
rate_test = rate_test[2039:]

plt.plot(time_train, rate_train, 'b')
plt.plot(time_train, pressure_train, 'g')

plt.plot(time_test, rate_test, 'b', label = 'Rate')
plt.plot(time_test, pressure_test, 'g', label = 'Pressure')
plt.plot(time_train, inv_total_train, 'k', label = 'Training')
plt.plot(time_test, inv_total_pred, 'r', label = 'Window predictions')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel('Time')
plt.ylabel('Rate and Pressure')
plt.xlim([0,3000])
plt.grid(True)
plt.show()

