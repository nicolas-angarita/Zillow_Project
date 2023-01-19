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
| customer_id  | unique id for each customer  |
| payment_type_id | 1, 2, 3, 4 related to payment type |
| internet_service_type_id | 1, 2, 3 related to internet service type |
| contract_type_id | 1, 2, 3 related to contract type |
| gender | male or female |
| senior_citizen | yes(1) or no(0) senior citizen |
| partner | yes or no has a partner | 
| dependents | yes or no has dependents |
| tenure | how long the customer has been with company |
| phone_service | yes or no has phone service |
| multiple_lines | yes or no has multiple_lines |
| online_security | yes or no has online security |
| online_backup | yes or no has online backup |
| device_protection | yes or no has device protection |
| tech_support | yes or no has tech support |
| streaming_tv | yes or no has streaming tv |
| streaming_movies | yes or no has streaming movies |
| paperless_billing | yes or no has paperless billing |
| monthly_charges | monthly charge to customer |
| total_charges | total charge to customer |
| contract_type |  month to month, 1 year, 2 month |
| internet_service_type | Fiber Optic, DSL, None | 
| payment_type | Electronic check, Mailed check, Bank transfer, Credit card |

## Steps to Reproduce

 - You will need an env.py file that contains the hostname, username and password of mySQL database that contains the telco dataset. Store that env.py file locally in the repository.

- Clone my repo including the acquire.py, prepare.py, and explore.py (make sure to create a .gitignore to hide your env.py file since it will have your credentials to access the data)

- Put the data in a file containing the cloned repo.

- Run notebook.

## Takeaways and Conclusions

 - TBD
  
 - TBD
 
 - TBD
 
 - TBD
 
**Best Model's performance:<br>
TBD 

## Recommendations
- TBD

- TBD

## Next Steps

- TBD
- Consider adding different hyperparameters to models for better results. 
