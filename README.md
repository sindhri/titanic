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

1.	EDA Study the training data and all the variables [Will add EDA plots] 

2.	Preprocessing Fill in the missing data for Age [Will add visualization] 

3.	Feature Engineer Extract feature from Name, Cabin, and Ticket [Will add visualization] 

4.	Model buidling and tuning Several Machine Learning models were built using the default settings using cross validation. Then parameters were set for Grid/Random search to find the best parameters for each model. At the end use a voting system to create the most appropriate decision for each prediction. 

Take away: The final model accuracy was 86% for the training data and 77% for the test data. More feature engining can be investigated to increase the accuracy.
