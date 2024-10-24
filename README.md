# Airline Passenger Forecasting Project

This project aims to forecast the number of airline passengers over time using two approaches:
1. **Deep Learning** with Long Short-Term Memory (LSTM) networks.
2. **Machine Learning** with multiple regression models such as Linear Regression, Multi-Layer Perceptron (MLP) Regressor, and various Support Vector Regressor (SVR) kernels.

## Project Overview

Forecasting airline passenger numbers can help airlines and stakeholders make data-driven decisions about capacity, pricing, and scheduling. In this project, we tackle the problem by implementing two different forecasting approaches. This involves building and training models that can predict future airline passenger numbers based on past data.

- **Deep Learning Model**: Uses a sequential LSTM model to capture the long-term temporal dependencies of the data.
- **Machine Learning Models**: Implements Linear Regression, MLP (Neural Network), and Support Vector Regression (SVR) models with multiple kernels (Linear, RBF, Sigmoid, and Polynomial).

## Dataset

The dataset used in this project is the **AirPassengers** dataset, which includes:
- **144 samples** representing **monthly** data on the number of airline passengers over a span of **12 years**.
- Columns: 
  - `Month`: Timestamp of each observation.
  - `#Passengers`: Number of passengers per month.

The dataset is **non-stationary**, meaning that trends and seasonality are present in the data, making it a challenging problem for predictive modeling.

## Approaches

### 1. Deep Learning Approach (LSTM)

LSTM is a type of Recurrent Neural Network (RNN) specifically designed for capturing temporal dependencies in time series data. The key components of the LSTM approach in this project include:
- **Input Sequence Length**: A sequence of the past 25 months is used to predict the next month’s passenger count.
- **Network Architecture**:
  - An LSTM layer with 20 units, which allows the model to learn the relationships in the sequence data effectively.
  - Followed by dense layers that output the final prediction.
- **Training**: The dataset is split into training (80%) and testing (20%) sets to evaluate model performance. The model is trained for 100 epochs with a batch size of 2, optimizing for mean squared error (MSE) using the Adam optimizer.

### 2. Machine Learning Approach

The machine learning models implemented in this project include:
- **Linear Regression**: A simple yet effective model for capturing linear relationships in the data.
- **Multi-Layer Perceptron (MLP) Regressor**: A neural network approach that can learn complex patterns but may require careful tuning of hyperparameters.
- **Support Vector Regression (SVR)**: Implemented with multiple kernels:
  - Linear Kernel: Captures linear relationships.
  - RBF Kernel: Effective for non-linear data distributions.
  - Sigmoid and Polynomial Kernels: Used to capture various complex relationships.

Each of these models is trained on the same training dataset and evaluated on the testing dataset.

## Performance Evaluation

### Evaluation Metrics
The performance of each model is evaluated using the following metrics:
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
- **Mean Absolute Error (MAE)**: Represents the average absolute difference between predicted and actual values.
- **Mean Absolute Percentage Error (MAPE)**: Indicates the average absolute percentage difference between predicted and actual values.

### Results
- **Best Model**: Linear Regression with:
  - MSE: 307.24
  - MAE: 12.60
  - MAPE: 2.78%
- **Second Best Model**: SVR Polynomial with:
  - MSE: 1324.16
  - MAE: 30.52
  - MAPE: 6.69%
- **LSTM Performance**:
  - MSE: 2829.91
  - MAE: 40.99
  - MAPE: 8.71%

While the LSTM model performs decently, it is not as effective as the Linear Regression model in this particular case.

## How to Run the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AminHentati/NLP_Project.git
   cd NLP_Project
2. **Install the required libraries**:
   ```bash
   pip install -r requirements.txt
3. **Run the Jupyter notebooks**:
   Open the Jupyter notebook interface:
    ```bash
    jupyter notebook
   Run the Final_Time_Series_&_Forecasting_using_Deep_Learning.ipynb notebook to execute the LSTM model.
   Run the Final_Time_Series_&_Forecasting_using_Machine_Learning.ipynb notebook to execute the machine learning models.
Conclusion
This project demonstrates the use of both deep learning and traditional machine learning techniques for time series forecasting, specifically for airline passenger data. The evaluation metrics provide insight into model performance, guiding the selection of the best model for prediction. The findings highlight the efficacy of linear regression in this context, with LSTM showing potential for further exploration.