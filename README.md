# Predicting your house value in SoCal

# Project Goals

 - Identify what drives home values 
 - Build a model to best predict home value

# Project Description

We are looking to come up with a machine learning model that will help us see which features gives us the best indicators of price and also be able to predict the price of the home. These homes are located in Southern California within the Orange County, Los Angeles County, and the Ventura County. After we have explored and made our models we will recommend what features are best to help predict price and give usesful insights on our data.

# Initial Questions

 1. Will more bedrooms and bathrooms drive house value up?
 2. Does lot size sqft increase home value?
 3. Does the month of the transaction of the home affect the price of the home?
 4. Are Orange County home values higher vs. Los Angeles and Ventura home values?



# The Plan

 - Create README with project goals, project description, initial hypotheses, planning of project, data dictionary, and come up with recommedations/takeaways

### Acquire Data
 - Acquire data from Sequel Ace and create a function to later import the data into a juptyer notebook. (acquire.py)

### Prepare Data
 - Clean and prepare the data creating a function that will give me data that is ready to be explored upon. Within this step we will also write a function to split our data into train, validate, and test. (prepare.py) 
 
### Explore Data
- Create Vizuals on our data 

- Create at least two hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, document any findings and takeaways that are observed.

### Feature Engineering:
 - Scale our data for our models
 
 - Create dummies to encode categorical variables for our models

### Model Data 
 - Establish a baseline(mean or median of target variable)
 
 - Create, Fit, Predict on train subset on four regression models.
 
 - Evaluate models on train and validate datasets.
 
 - Evaluate which model performs the best and on that model use the test data subset.
 
### Delivery  
 - Create a Final Report Notebook to document conclusions, takeaways, and next steps in recommadations for predicitng house values. Also, inlcude visualizations to help explain why the model that was selected is the best to better help the viewer understand. 


## Data Dictionary


| Target Variable |     Definition     |
| --------------- | ------------------ |
|      home_value      | price of the house |

| Feature  | Definition |
| ------------- | ------------- |
| year_built | The year the house was built  |
| lot_sqft | The square feet of the lot  |
| long | The longititude coordinates of the house |
| lat | The latitude coordinates of the house |
| transaction_month | The month of the transaction date |
| bathrooms | Number of bathrooms in the house | 
| bedrooms | Nummber of bedrooms in the house |
| fips | Code identifier for county |
| sqft | Square feet of the property|
| county | Name of the county the house is located |


## Steps to Reproduce

 - You will need an env.py file that contains the hostname, username and password of mySQL database that contains the telco dataset. Store that env.py file locally in the repository.

- Clone my repo including the acquire.py, prepare.py, and explore.py (make sure to create a .gitignore to hide your env.py file since it will have your credentials to access the data)

- Put the data in a file containing the cloned repo.

- Run notebook.

## Conclusionss

**Home value predicions were used by minimizing RMSE within our models. County and transaction month have proven to be the most valuable, but there is still room for improvement.**


 
**Best Model's performance:<br>
My best model reduced the root mean squared error by 82500 compared to the baseline results.**

**RMSE 82493.41 on in-sample (train), RMSE 84100.76 on out-of-sample data (validate) and RMSE of 80897.87 on the test data.**

## Recommendations
- I would recommend using counties and transaction months to build models to predict home values. 

- I would also recommend collecting more data about the homes. For example, crime rates in the county, average salary within neighborhoods, and school rankings

## Next Steps

- I would add more columns/features from the database to see if there are better relationships to help predict price
- Consider adding different hyperparameters to models for better results.
