# Salary Classifier

## Link to the deployed API:
https://ml-salary-classifier.herokuapp.com

## Product Perspective
The ML salary classifier is a machine learning-based object classifier model which will help us to predict the classification of the salary of the subject based on the values of different parameters.

## PROPOSED SOLUTION
The solution proposed here is an XGBOOST model which can classify the salary by using the feature set described above. The model has also been hyper-parameter tuned using RandomCVSearch to provide the best results.

## Data Requirements
The user only needs to enter the data in the form of number for some fields and select options from drop down menu for the rest of the fields. The request form will then record the responses of the user and convert it to a NumPy array for the model to work on it.

## Tools Used
Python programming language and frameworks such as NumPy, Pandas, Scikit-learn, XGBOOST and Flask were used to build the whole model. For deployment purposes , Heroku was used.


<img src="https://user-images.githubusercontent.com/89142021/136641827-9e56c9a3-b86f-43d8-b811-3c2d72063ae7.png" width="38"> <img src="https://user-images.githubusercontent.com/89142021/136641861-54181fcd-2b80-476c-b956-1b1e2c776a24.png" width="58"> <img src="https://user-images.githubusercontent.com/89142021/136642111-710bb174-1325-47aa-8b01-a988418ea829.png" width="58"> <img src="https://user-images.githubusercontent.com/89142021/136642133-d9b6459b-9b5d-466d-8cf0-b0297419be2e.png" width="58"> <img src="https://user-images.githubusercontent.com/89142021/136642249-bd4f0efb-0a6c-4e24-86de-01b1b680a581.png" width="58"> <img src="https://www.logo.wine/a/logo/MySQL/MySQL-Logo.wine.svg" width="88">

-	Visual Studio Code is used as IDE. <br>
-	For visualization of the plots, Matplotlib and Seaborn are used. <br>
-	Heroku is used for deployment of the model.<br>
-	MySQL is used to retrieve, insert, delete, and update the database.<br>
-	Front end development is done using HTML/CSS <br>
-	Python Flask is used for backend development. <br>
-	GitHub is used as version control system.
	
## Process Flow
For identifying the different classes of the salary , we will use a machine learning base model. Below is the process flow diagram is as shown below.<br>
![image](https://user-images.githubusercontent.com/89142021/136642891-e57511b8-e4b2-4d70-abd6-7ed87ab0a843.png)<img src="https://user-images.githubusercontent.com/89142021/136642947-dd21c2d2-2e86-427b-a3d9-849742c631af.png" width="48" height="40">![image](https://user-images.githubusercontent.com/89142021/136643000-abd8e4a5-2f37-40db-941c-6b4bfa2c3a45.png)<img src="https://user-images.githubusercontent.com/89142021/136642947-dd21c2d2-2e86-427b-a3d9-849742c631af.png" width="48" height="40">![image](https://user-images.githubusercontent.com/89142021/136643015-ae87c4b1-cf0b-4d78-9aaf-c2097b44d07b.png)<img src="https://user-images.githubusercontent.com/89142021/136642947-dd21c2d2-2e86-427b-a3d9-849742c631af.png" width="48" height="40">![image](https://user-images.githubusercontent.com/89142021/136643035-b3b93a0d-03c7-43ec-9e1d-05ff95f0c19d.png)<img src="https://user-images.githubusercontent.com/89142021/136642947-dd21c2d2-2e86-427b-a3d9-849742c631af.png" width="48" height="40">![image](https://user-images.githubusercontent.com/89142021/136643081-0a559048-74c0-430f-9d6a-79aa39264021.png)


## Performance
The performance of the ML salary classifier is solely dependent upon the training dataset it uses to generate the XGBOOST model. This statement is supported by the fact that the model uses only a select few features from the total feature set as a consequence of feature selection, and the particular features which are selected totally depends upon the training dataset. Also, the hyperparameter tuning performed to select the specific XGBOOST parameters depends upon the composition of dataset as well.

## Potential Industry Of Use:
Such a model would be highly beneficial for use in finance industry.

## Column Descriptions

Age -> Age of the person <br>
Workclass -> Class of work <br>
Fnlwgt -> Final weight of how much of the population it represents <br>
Education -> Education level <br>
Education_num -> Numeric education level <br>
Marital_status -> Marital status  of the person <br>
Occupation -> Occupation  of the person <br>
Relationship -> Type of relationship <br>
Race -> Race of the person <br>
Sex -> Sex of the person <br>
Capital_gain -> Capital gains obtained <br>
Capital_loss -> Capital loss <br>
Hours_per_week -> Average number of hour working per week<br>
Native_country -> Country of origin<br>
Salary -> Income level (To be predicted)<br>






