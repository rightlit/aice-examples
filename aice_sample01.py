df = pd.read_csv('signal_data.csv')


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#model = DecisionTreeClassifier(random_state=42)
#model2 = RandomForestClassifier(n_estimators=10)

from sklearn.tree import DecisionTreeRegressor
#model = DecisionTreeRegressor(random_state = 120)
model = DecisionTreeRegressor(random_state = 120, min_samples_split=3)
model.fit(X_train, Y_train)

from sklearn.ensemble import RandomForestRegressor
model2 = RandomForestRegressor()
model2.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

y_predict = model.predict(X_test)
mse = mean_squared_error(Y_test, y_predict)
mse

y_predict = model2.predict(X_test)
mse2 = mean_squared_error(Y_test, y_predict)
mse2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Dense(64, activation='selu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='selu'))
model.add(Dense(16, activation='selu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

history = model.fit(X_train, Y_train, epochs=50, batch_size=128)
#loss, mse = model.evaluate(X_test, Y_test)

#early_stop = EarlyStopping(monitor='val_loss', patience=3)
early_stop = EarlyStopping(monitor='val_loss')
history = model.fit(X_train, Y_train, 
          validation_data=(X_test, Y_test), epochs=50, batch_size=128, 
          callbacks=[early_stop])
          
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.plot(history.history['mse'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train', 'val'], loc='upper right')
plt.show()
          




