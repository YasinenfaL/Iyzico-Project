# İyzico / Merchant Daily Trading Volume Forecast
![Iyzico_logo svg](https://github.com/YasinenfaL/Iyzico-Project/assets/111612847/9896dbc7-30fa-489f-98eb-e7f60582efa4)

# Business Problem
Operating in the fintech sector, İyzico provides payment infrastructure for its customers. The company's data science team was asked to estimate the daily trading volume of merchants.

# About Dataset
7767 Observation

| Transaction_date | Merchant ID | Total Transactions | Total Paid |
|------------------|-------------|--------------------|------------|
| History of sales data | Numbers of member businesses | Number of transactions | Payment amount |

# About the solution to the Business Problem
At the core of the problem lies a time series. Since it is a time series problem, the stationarity of the dataset has been examined. The non-stationary nature of the dataset has been detected through graphical representation. To make the dataset stationary, the difference from the previous day has been taken. The dataset contains 4 variables, but we generated new variables such as roll means and lag day to solve this problem. Using the date variable, we derived new variables. We used the LGBM model with these newly created variables to predict trading volumes. We used the smape ratio as a measure of error, and obtained a smape ratio of 21%. This means there is a deviation of 21% in each prediction.

After installing the model, we successfully deployed our project on Kubernetes with the YAML file we created during the deployment phase. In order to be able to interact with our project with the outside world, we chose Ingress, which is a safer option. In addition, we have created a drift mechanism against unexpected deviations that may occur in the predictions. Using this mechanism, we based on the Kolmogorov-Smirnov method. In this way, we can detect deviations in our model and intervene in the model when necessary.

# Project Diagram
                      +-----------------+
                      |     Client      |
                      +--------+--------+
                               |
                               |
                      +--------v--------+
                      |   FastAPI API   |
                      +--------+--------+
                               |
                               |
                      +--------v--------+
                      |   LightGBM      |
                      |    Model        |
                      +--------+--------+
                               |
                               |
                      +--------v--------+
             +--------|   Kubernetes    |--------+
             |        |   Cluster       |        |
             |        +--------+--------+        |
             |                 |                 |
             |                 |                 |
             |        +--------v--------+        |
             |        |   Ingress       |        |
             |        |   Controller    |        |
             |        +--------+--------+        |
             |                 |                 |
             |                 |                 |
             |        +--------v--------+        |
             +--------|   LightGBM      |--------+
                      |    API Service  |
                      +--------+--------+
                               |
                      +--------v--------+
                      |   Processing    |
                      |   System        |
                      +-----------------+


           
