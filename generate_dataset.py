import os
import numpy as np
import pandas as pd
import tensorflow as tf

def generate_csv_datasets():
    print("Loading MNIST dataset from Keras...")
    # Load raw MNIST data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Reshape images to 784 flat pixels (28x28)
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    x_train_flat = x_train.reshape(num_train, 784)
    x_test_flat = x_test.reshape(num_test, 784)
    
    # Generate headers: 'label', '1x1', '1x2', ..., '28x28'
    pixel_headers = [f"{i}x{j}" for i in range(1, 29) for j in range(1, 29)]
    headers = ["label"] + pixel_headers
    
    print("Creating training DataFrame...")
    # Combine label column with pixel features
    train_data = np.hstack((y_train.reshape(-1, 1), x_train_flat))
    train_df = pd.DataFrame(train_data, columns=headers)
    
    print("Creating testing DataFrame...")
    test_data = np.hstack((y_test.reshape(-1, 1), x_test_flat))
    test_df = pd.DataFrame(test_data, columns=headers)
    
    # Ensure dataset directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save to CSV
    print("Saving to data/mnist_train.csv...")
    train_df.to_csv("data/mnist_train.csv", index=False)
    print("Saving to data/mnist_test.csv...")
    test_df.to_csv("data/mnist_test.csv", index=False)
    
    print("Dataset generation completed successfully!")

if __name__ == "__main__":
    generate_csv_datasets()
