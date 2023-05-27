
# Alphabet Soup Charity Deep Learning Model

This repository contains code and files for the Alphabet Soup Charity Deep Learning Challenge. The goal of this project is to develop a binary classification model that can predict the success of organizations funded by Alphabet Soup.

# Background

Alphabet Soup is a nonprofit foundation that aims to select applicants with the highest chance of success for funding. To achieve this, a machine learning and neural network-based solution is required. The provided dataset consists of over 34,000 organizations that have received funding from Alphabet Soup. Various columns capture metadata about each organization, including application type, affiliation, classification, use case, organization type, funding amount requested, and more.

# Preprocessing the Data

In this step, the dataset is preprocessed to prepare for model compilation, training, and evaluation. The following tasks are performed:

The 'charity_data.csv' file is read into a Pandas DataFrame.
The target variable(s) and feature(s) for the model are identified.
The EIN and NAME columns are dropped.
The number of unique values for each column is determined.
Columns with more than 10 unique values are analyzed to determine the number of data points for each unique value.
"Rare" categorical variables are binned together into a new value called "Other" based on a cutoff point.
Categorical variables are encoded using pd.get_dummies().
The data is split into features array (X) and target array (y) using train_test_split.
The training and testing features datasets are scaled using StandardScaler.

# Compiling, Training, and Evaluating the Model

In this step, a neural network model is designed, compiled, trained, and evaluated using TensorFlow and Keras. The following tasks are performed:

The neural network model is created by specifying the number of input features and nodes for each layer.
The first hidden layer is created with an appropriate activation function.
If necessary, a second hidden layer is added with an appropriate activation function.
The output layer is created with an appropriate activation function.
The structure of the model is checked.
The model is compiled and trained using the training data.
A callback is created to save the model's weights every five epochs.
The model is evaluated using the test data to calculate the loss and accuracy.
The results are saved and exported to an HDF5 file named "AlphabetSoupCharity.h5".

# Optimizing the Model

In this step, the model is optimized to achieve a target predictive accuracy higher than 75%. Various methods can be used for optimization, including:

Adjusting the input data to handle outliers or confusing variables.
Modifying the dataset by dropping or creating more bins for rare occurrences.
Adding more neurons or hidden layers.
Using different activation functions.
Adjusting the number of epochs for training.
The optimized model is saved and exported to an HDF5 file named "AlphabetSoupCharity_Optimization.h5".

# Report on the Neural Network Model

A report is written to summarize the performance of the deep learning model for Alphabet Soup. The report includes an overview of the analysis, results, and a summary. The following questions are addressed:

Data Preprocessing:

Identification of target(s) and feature(s) for the model.
Removal of irrelevant columns from the input data.
Compiling, Training, and Evaluating the Model:

Selection of the number of neurons, layers, and activation functions for the neural network model.
Achievement of the target model performance.
Steps taken to increase model performance.
Summary:

Overall results of the deep learning model.
