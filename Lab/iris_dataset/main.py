import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Function to get the euclidean distance between 2 vectors
def euc_dist(v1, v2):
    '''
    v1 = A list of floats
    v2 = A list of floats
    '''
    distance = 0.0
    # Following the formula:
    # d(p, q)^2 = (p1 - q1)^2 + (p2 - q2)^2 + (p3 - q3)^2 + (p4 - q4)^2
    # With p and q being vectors containing 4 elements each (4 features from Iris dataset)
    # -1 because we ignore the label
    for i in range(len(v1) - 1):
        distance += (v1[i] - v2[i]) ** 2
    return math.sqrt(distance)

# function to perform k-nearest neighbors
def knn(train, test_v, k):
    '''
    train = The train dataset
    test_v = A data point/vector to determine its nearest neighbors
    k = number of neighbors
    '''
    # distances is a list of tuples.
    # The tuple will consist of 2 values:
    # 1. A data point from train dataset
    # 2. Its euclidean distance with respect to test_v
    distances = []
    for t in train:
        # Get distance from each train data with respect to test_v
        d = euc_dist(test_v, t)
        distances.append([t, d])

    # Sort the distance in ascending order
    distances.sort(key=lambda tup: tup[1])

    # Get k neighbors
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    
    return neighbors

# Predict the classification of the flower by using majority voting
def predict(train, test_v, k):
    '''
    train = The train dataset
    test_v = A data point/vector to determine its nearest neighbors
    k = number of neighbors
    '''
    neighbors = knn(train, test_v, k)
    # Get only the labels
    neighbor_labels = [n[-1] for n in neighbors]
    # Determine the highest number of the same class
    prediction = max(set(neighbor_labels), key=neighbor_labels.count)
    return prediction

# Evaluate the error of the model
def evaluate(train, test, k:int):
    '''
    train = list of values from DataFrame.
    test = list of values from DataFrame. The actual label must be present.
    k = number of neighbors.
    '''
    wrong_prediction = 0
    for test_v in test:
        prediction = predict(train, test_v, k)
        actual = test_v[-1]
        # Check if the predicted label is the same as the actual label
        if prediction != actual:
            wrong_prediction += 1
    
    # Determine accuracy
    error = wrong_prediction / len(test)
    return error


def main():
    # Load the iris dataset into a pandas DataFrame
    df = pd.read_csv("./iris.csv", header=None)

    # Shuffle the dataset
    seed = 69
    df = df.sample(frac=1, random_state=seed)

    # Split DataFrame based on classification
    df_setosa = df[df[4] == 'Iris-setosa']
    df_versicolor = df[df[4] == 'Iris-versicolor']
    df_virginica = df[df[4] == 'Iris-virginica']

    # Create train and test dataset (50%)
    # This is done by taking the first 50% and last 50% respectively from each individual classifications.
    # No need to worry about randomization because it has been done beforehand
    df_train = pd.concat([df_setosa.head(25), df_versicolor.head(25), df_virginica.head(25)])
    df_test = pd.concat([df_setosa.tail(25), df_versicolor.tail(25), df_virginica.tail(25)])
    train = df_train.values
    test = df_test.values

    # Set the values of k
    k_list = [x for x in range(1, len(train), 2)]

    # Evaluate prediction according to the values of k
    errors = []
    for k in k_list:
        errors.append(evaluate(train, test, k))
    
    # Plot the errors
    plt.plot( k_list, errors)
    plt.suptitle('Classification error rate vs k', fontsize=20)
    plt.title(f'with seed of {seed}')
    plt.xlabel('k')
    plt.ylabel('Classification error rate')
    plt.show()
    

if __name__ == '__main__':
    main()