The Titanic survival prediction from Kaggle competition.

End_to_end Python based predictive modeling using modern Machine Learning techniques (Logistic Regression, K-nearest Neighbor, Decision Tree, Random Forest, Support Vector Classification (SVC), XGBoost) with cross-validation, Grid and Random search to tune each model, and eventually using voting system to create the best prediction. (Inspired by Ken Jee!)

The model reached 86% accuracy in the training data and 77% accuracy in the test data. One thing to mention is that, unlike many other practioners of Machine Learning, this model takes NO prior knowledge of the test data. 
All the parameters were built only using the training data, which is critical for industry applications.

Titanic still occuplies our mind 100 years after the disaster happened. Among the over 2000 passengers, about 1500 lost their lives. 
It would be vital to understand among the ones who died or survived, are their other factors that were associated with it. 
It would be interesting to see whether we can use other information, such as sex, fare, age etc to predict whether a person was likely to survive.

In this project, the training data has information on 891 participants including sex, age, amount of fare they paid, cabin number, ticket number, number of siblings or spouse onboard, number of children or parents onboard, class, embarked location, and whether they survived or not. 
The test data has inforamtion on 418 and is asked to predict every one of them whether he/she was giong to survive.
The project has four steps:

# 1.	EDA Study the training data and all the variables

## 1.1 check missing values, check variables in the datasets
<img src="https://github.com/sindhri/titanic/blob/master/images/img1.png" width="250"><img src="https://github.com/sindhri/titanic/blob/master/images/img2.png" width="700">

## 1.2 distribution of the numeric variables and their correlation
observation: 
* Age is pretty much normally distributed, the rest of the variables need normalization
* Parch (number of parents/children aboard) is positively correlated with SibSp(number of sibling/spouse aboard)
* age is negatiavelly correated with SibSp (number of siblings)  
<img src="https://github.com/sindhri/titanic/blob/master/images/img3.png" width="400"> <img src="https://github.com/sindhri/titanic/blob/master/images/img4.png" width="400">
<img src="https://github.com/sindhri/titanic/blob/master/images/img5.png" width="400"> <img src="https://github.com/sindhri/titanic/blob/master/images/img6.png" width="400">
<img src="https://github.com/sindhri/titanic/blob/master/images/img7.png" width="400">

## 1.3 bart plots of the categorical variables
observations:    
* more people died than survived
* more people are in the 3rd class cabin  
* more male than female
* more people embared from S than from C and Q  

<img src="https://github.com/sindhri/titanic/blob/master/images/img8.png" width="400"> <img src="https://github.com/sindhri/titanic/blob/master/images/img9.png" width="400"><br>
<img src="https://github.com/sindhri/titanic/blob/master/images/img10.png" width="400"><img src="https://github.com/sindhri/titanic/blob/master/images/img11.png" width="400">

## 1.4 compare the survival rate across all the numeric variables (Age, SibSp, Parch, and Fare) and categorical variables (Sex, Pclass, Embarked)
* observation, higher Faire has a higher survival rate
* high Parch has a higher survival rate
* lower SibSp and lower age has a higher survival rate
* Survivied female > male
* Survivied Pclass 1 > 2 > 3
* Survivied Embarked C > O > S  
<img src="https://github.com/sindhri/titanic/blob/master/images/img12.png" width="500"> <img src="https://github.com/sindhri/titanic/blob/master/images/img13.png" width="250">

## 1.5 Experimenting with feature engineering
### Simplify Cabin 1. by the number of cabins, NaN is 0, 2. 
people with 1, 2, 4 cabins have a higher survival propertion than nonsurvive  
<img src="https://github.com/sindhri/titanic/blob/master/images/img14.png" width="400">
### simplify Cabin 2. by the first letter of the cabin
More people in the following categories survivied: B, D, E, F  
<img src="https://github.com/sindhri/titanic/blob/master/images/img15.png" width="500">
### Simplify Tickets by the first letter of the ticket
More survival with the following ticket_firstletter: F, P  
Very little survival with the following ticket_firstletter: A, W  
moderatte survival rate with the following ticket_firstletter: C, None  
<img src="https://github.com/sindhri/titanic/blob/master/images/img16.png" width="500">  
### Simplify Name by extracting the title
<img src="https://github.com/sindhri/titanic/blob/master/images/img17.png" width="900">  

## 1.6 plot the survival rate in relation to multiple other features
### Survival rate ~ Sex + Age
* observation: male 20-40 yr many not survived
* female has a large survival rate across all ages  
<img src="https://github.com/sindhri/titanic/blob/master/images/img18.png" width="900">  

### Survival ~ Age + Sex + Pclass
* observation, male from age 20-40 in Pclass 2 and 3 mostly did not survive
<img src="https://github.com/sindhri/titanic/blob/master/images/img19.png" width="900">  

### Survival ~ Embarked + Age + Pclass
* Observations: Pclass 3 has a much lower survival rate than Pclass 1 and 2 across Sex and Embarked
* Male when Embarked from Q has a particular lower survival rate than Embarked S and C  
<img src="https://github.com/sindhri/titanic/blob/master/images/img20.png" width="900"> 

### Survival ~ Age + Fare
* observation: Higher fare has a higher survival rate across most of the age spectrum. 
* Younger age 0-10 has a higher survival rate
* older age 60 + has a lower survival rate  
<img src="https://github.com/sindhri/titanic/blob/master/images/img21.png" width="500"> 

### Survival ~ cabin_firstletter
* Observation: most people fall in the category of n, which means none for cabin.
* and in the n category the survival rate is lower than other categories  
<img src="https://github.com/sindhri/titanic/blob/master/images/img22.png" width="500"> 

### Survival ~ ticket_firstletter
* Observation: most people fall in the category of None, which means no ticket number.
* and the survival rates are lower in the following categories: None, A, S, C, W  
<img src="https://github.com/sindhri/titanic/blob/master/images/img23.png" width="500"> 

### Survival ~ name_title_adv
* Observation: Most people fall in the Mr. category and it has a low survival rate
* Category Msr, Miss, Master has a higher survival rate  
<img src="https://github.com/sindhri/titanic/blob/master/images/img24.png" width="900"> 

### conclusion: 
* based on EDA, the following variables should be included as features:
* Pclass, name_title_adv, Sex, Age, Sibsp, Parch, Fare, Embarked, cabin_total, cabin_firstletter, ticket_firstletter

# 2.	Preprocessing + Feature Engineer Extract feature from Name, Cabin, and Ticket
## Organized and prepared a helper module for feature engineering (according to EDA) so it can be readily applied for both the training and test sets.
* convert Pclass to categorical
* fill in the empty cells of 'Embarked'
* normalize then fill in the empty cells for 'Fare'
* Simplify Name by creating 'name_title_adv'
* Simplify Cabin by creating 'cabin_firstletter' and 'cabin_total'
* Simplify Ticket by creating 'ticket_firstletter'
* Replace the values in 'name_title_adv' in the test set that is absent in the training with training mode
* Fill the empty cells of Age by aggregrated values from 'name_title_adv'
* remove extra columns
* Merge training and test together to create a consistent dummy-variable set across train and test, then separate the datasets
* scale the numeric columns for both datasets  
<img src="https://github.com/sindhri/titanic/blob/master/images/img25.png" width="450"><img src="https://github.com/sindhri/titanic/blob/master/images/img26.png" width="450"> 

# 3.	Model buidling and tuning Several Machine Learning models
* sklearn
* Tested multiple ML models: Naive Bayes, Logistic Regression, Decision Tree, K Nearest Neighbors, Random Forest, SVC, XGBoost
* Used sklearn.ensemble to create a voting system
* Used the average accuracy from 5-folds cross-validtion
* Turned each model by either Grid Search or Random Search to improve accuracy

Accuracy improvment after tuning:
<table>
  <tr>
    <th>Algorithm</th>
    <th>Accuracy with Default Parameters</th>
    <th>Accuracy after Tuned</th>
  </tr>
  <tr>
    <td>naive bayes</td>
    <td>0.4668</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>Logistic regression</td>
    <td>0.8193</td>
    <td>0.8215</td>
  </tr>
  <tr>
    <td>Decision tree</td>
    <td>0.7924</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>k nearest neighbor</td>
    <td>0.8149</td>
    <td>0.8249</td>
  </tr>
  <tr>
    <td>Random Forest</td>
    <td>0.81</td>
    <td>0.8372</td>
  </tr>
  <tr>
    <td>SVC</td>
    <td>0.8306</td>
    <td>0.8350</td>
  </tr>
  <tr>
    <td>xgboost</td>
    <td>0.8305</td>
    <td>0.8451</td>
  </tr>
</table>

# 4. Prepare submission file using the algorithm of choice (xgboost after tuned)

## The final model accuracy was 85% for the training data and 77% for the test data. 

## Feature Importance
* Sex being male is the most important feature
* Then the next important feature is whether the person is called Master
* The next important feature is whether the passenger is in Pclass3
src="https://github.com/sindhri/titanic/blob/master/images/img27.png" width="900"> 

## More feature engining can be investigated to increase the accuracy.
