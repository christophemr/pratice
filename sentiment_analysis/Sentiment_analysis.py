import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Load the dataset
max_features = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Padding sequences
max_len = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)

# Model Building
model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_len))
model.add(LSTM(64, return_sequences=True))  # Use LSTM for better performance with sequences
model.add(Dropout(0.5))  # Add dropout to prevent overfitting
model.add(LSTM(32))  # Another LSTM layer
model.add(Dropout(0.5))  # Another dropout layer
model.add(Dense(128, activation='relu')) # Added a dense layer
model.add(BatchNormalization()) # Batch Normalization
model.add(Dropout(0.5)) #Dropout
model.add(Dense(1, activation='sigmoid'))

# Optimizer
optimizer = Adam(learning_rate=0.001)

# Compile the model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
earlystopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Training the model
history = model.fit(
    X_train, y_train, epochs=10, batch_size=64,
    validation_split=0.2,
    callbacks=[earlystopping, lr_scheduler]
)

# Evaluation on the test sets
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")