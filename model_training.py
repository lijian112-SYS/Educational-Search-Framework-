
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load data
df = pd.read_csv('preprocessed_dataset.csv')
X = df['Processed_Description']
y = df['Subject']

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Simple text vectorization (basic)
vectorizer = tf.keras.layers.TextVectorization(max_tokens=1000, output_mode='int')
vectorizer.adapt(X.values)
X_vectorized = vectorizer(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized.numpy(), y_encoded, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(None,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(set(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save('educational_search_model.h5')
