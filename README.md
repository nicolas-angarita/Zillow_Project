# Predicting your house value in SoCal

# Project Goals

 - Identify key drivers of churn. 
 - Predict which customers are at risk of churning

# Project Description

Anytime a business is struggling, they have to revert to the basic questions of business and answer who is the customer and what makes them stay or leave. In this case why does it seem like so many customers are churning. What are key drivers that are turning our customers away? Could it be the services we are providing, our pricing policy, our target market (age of customer, gender, etc.), or other drivers. We will analyze some of the drivers just mentioned to see if we can come up with a model to see what is causing our customers to chun. Once we have ran our models we will make recommendations on how to better retain our customers from churning in the future. 

# Initial Questions

 1. Do customers with partners or dependents churn more than those with no dependents or partner?
 2. Does the type of contract a customer have make it more likley to churn?
 3. Are customers with different internet service more or less likely to churn?
 4. Are Senior Citizens more or less likely to churn than non Senior Citizen?

# The Plan

 - Create README with project goals, project description, initial hypotheses, planning of project, data dictionary, and come up with recommedations/takeaways

### Acquire Data
 - Acquire data from Sequel Ace and create a function to later import the data into a juptyer notebook. (acquire.py)

### Prepare Data
 - Clean and prepare the data creating a function that will give me data that is ready to be explored upon. Within this step we will also write a function to split our data into train, validate, and test. (prepare.py) 
 
### Explore Data
 - Create at least two hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, document any findings and takeaways that are observed.
 
### Model Data 
 - Establish a baseline accuracy
 
 - Create and train four classification models.
 
 - Evaluate models on train and validate datasets.
 
 - Evaluate which model performs the best and on that model use the test data subset.
 
### Delivery  
 - Create CSV file with the customer_id, probability of churn, and prediction of churn for each observation in my test dataset.
 
 - Create a Final Report Notebook to document conclusions, takeaways, and next steps in recommadations for customers churning. Also, inlcude visualizations to help explain why the model that was selected is the best to better help the viewer understand. 


## Data Dictionary


| Target Variable |     Definition     |
| --------------- | ------------------ |
|      Churn      | yes(1) or no(0) customer left |

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

 - Customers who have dependents or partners churned at a much lower rate those those who did not
  
 - The longer the contract a customer had the less likely they were to churn. With month-to-month having the most churned customers.
 
 - Customers who have fiber optic internet are more likely to churn compared to other internet customers
 
 - Non Senior citizens are more likely to churn than senior citizens.
 
**Best Model's performance:<br>
Accuracy of 85.72% on in-sample (train), 85.59% on out-of-sample data (validate) and an accuracy of 86% on the test data.**

## Recommendations
- We would want to target those who have dependents or partners to be within our target market. Seeing that customers that have dependents or a partner churn at a lower rate.

- I would recommend promoting longer contracts for our customers, making them more appealing to customers.

## Next Steps

- Figure out why month to month and fiber optic customers are churning at such a high rate.
- Consider adding different hyperparameters to models for better results. 
