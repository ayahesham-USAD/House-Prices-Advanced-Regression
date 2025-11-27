## Abstract
In this project, I worked on predicting house prices using the Kaggle “House Prices” dataset. After experimenting with several supervised learning algorithms, the best performance was achieved using a Stacked Regressor model, which reached an RMSE of 0.3769 on the validation set.
The workflow included training nine different models and tuning some of them using GridSearchCV. Performance was mainly assessed using cross-validation scores and RMSE. The final stacked model combined Random Forest, SVR, KNN, and Ridge Regression, with a Random Forest used as the meta-learner. This stacking approach improved the overall predictive performance compared to the individual models.

## Introduction
Predicting housing prices is useful in many real-world applications. Housing markets are strongly connected to economic indicators like inflation and consumer demand. Because of that, being able to estimate how house prices might change can help analysts, policy makers, and financial institutions make informed decisions.
With the growth of machine learning, regression techniques have become a practical and effective way to model and predict house prices based on various property characteristics. This project focuses on building and comparing machine-learning models to find the most accurate one for predicting sale price.

## Dataset
The dataset used is the publicly available Kaggle “House Prices: Advanced Regression Techniques” dataset.
Training data: 1460 rows and 80 columns (house features + sale price).
Test data: contains 1461 rows and 79 features (without the target).
The data includes numerical and categorical attributes describing different structural and environmental characteristics of residential properties.  

## Feature Engineering
All analysis and modelling were carried out in a Jupyter Notebook using Python. I mainly relied on pandas, NumPy, and scikit-learn for data processing and modelling, while matplotlib and seaborn were used to visualise patterns and detect issues like skewness or outliers.
Key preprocessing steps included:

1. Removing irrelevant or misleading columns
The Id column was removed since it doesn’t contribute to prediction.

2. Handling outliers
Scatter plots helped identify extreme data points that could negatively affect training.
For example, houses with GrLivArea > 4000 but unusually low prices were removed to avoid distortion.

3. Dealing with missing values
Different strategies were applied depending on the meaning of the feature:
Columns with more than 90% missing entries were dropped.
Columns like Fence, FireplaceQu, GarageType, etc., had missing values replaced with “None”, since missingness usually indicates absence of that feature.
Numerical features such as GarageCars were filled with 0 when appropriate.
Columns with only a few missing entries were filled using the most common value.
LotFrontage values were imputed using the median, assuming neighbouring houses would have similar frontage lengths.

4. Creating new features

Additional engineered features were added, such as:
total area
total bathrooms
combined quality score
total porch area
These helped the models capture more meaningful patterns.

5. Encoding and alignment
Categorical variables were converted into dummy variables.
Finally, the training set was split into 70% training and 30% validation to evaluate model performance.

## Data Modelling
A total of nine different models were used for prediction.Root mean square error and k-fold cross validation were the primary metrics used for evaluating the models. 

#Model,	RMSE on validation set,  Mean CV Score

- Linear Regression:	0.42793480397157035,0.4752199075347933

- Ridge:	0.3957886167433282,0.4167270749625921

- Lasso:	0.4059493256188701, 0.4297643272515321

- K- Nearest Neighbour: 0.41351487769327555
	
- Decision Tree:	0.4583579345988703 , 0.45825272349446555

- Random Forest:	0.38616747296757176, 0.403804172243945

- Support Vector Regressor:	0.3900469727418305, 0.40963206887105647

- Gradient Boosting Regressor:	0.4118219430457788, 0.4292489907640939

- Stacked Regressor:	0.3769718491202983, 0.4093903027876036

From the cross-validation error scores, it can be observed that the random forest and stacked regressor model have the lowest error.  The root mean square error of stacked regressor is the lowest. Scatter plots were also observed between the actual values and predicted values. Overall, it is observed that the stacked regressor model showed improved performance compared to the performance of all the other models used as estimators for this model.

