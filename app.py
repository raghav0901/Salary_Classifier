import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import mysql.connector as sql

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
connection=sql.connect(host='us-cdbr-east-04.cleardb.com',user='b77648943f2114',password='517f5ad6',database='heroku_4fa29ab7f3558b6',connect_timeout=6000)
print(connection)

cursor=connection.cursor()

@app.route('/')
def home():
    global connection
    global cursor
    print(connection)

    try:
      cursor.execute("CREATE TABLE IF NOT EXISTS TestyData( age int , fnlwgt int, education varchar(255), education_num int, occupation varchar(255), capital_gain int, capital_loss int, hours_per_week int, country varchar(255), race varchar(255), relationship varchar(255), sex varchar(255), workclass varchar(255),prediction varchar(255) )")
      cursor.execute("show tables")
    except sql.Error as err:
      connection=sql.connect(host='us-cdbr-east-04.cleardb.com',user='b77648943f2114',password='517f5ad6',database='heroku_4fa29ab7f3558b6',connect_timeout=6000)
      cursor=connection.cursor()
      cursor.execute("CREATE TABLE IF NOT EXISTS TestyData( age int , fnlwgt int, education varchar(255), education_num int, occupation varchar(255), capital_gain int, capital_loss int, hours_per_week int, country varchar(255), race varchar(255), relationship varchar(255), sex varchar(255), workclass varchar(255),prediction varchar(255) )")
      cursor.execute("show tables")
     
    for x in cursor:
        print(x)

    return render_template('index.html')


@app.route('/View')
def View():
    global connection
    global cursor 
    print(connection)
    try:
      cursor.execute("select * from TestyData")
    except sql.Error as err:
      connection=sql.connect(host='us-cdbr-east-04.cleardb.com',user='b77648943f2114',password='517f5ad6',database='heroku_4fa29ab7f3558b6',connect_timeout=6000)
      cursor=connection.cursor()
      cursor.execute("select * from TestyData") 
    data=cursor.fetchall()
    for x in data:
        print(x)
    return render_template('template.html',output_data=data)

@app.route('/predict',methods=['POST'])
def predict():
    global connection
    global cursor 
    
    '''
    For rendering results on HTML GUI
    '''
    cols=['age', 'fnlwgt', 'education', 'education-num', 'occupation', 'capital-gain', 'capital-loss', 'hours-per-week', 'country', 'race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White', 'relationship_Husband', 'relationship_Not-in-family', 'relationship_Other-relative', 'relationship_Own-child', 'relationship_Unmarried', 'relationship_Wife', 'sex_Female', 'sex_Male', 'workclass_Federal-gov', 'workclass_Local-gov', 'workclass_Never-worked', 'workclass_Private', 'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc', 'workclass_State-gov', 'workclass_Without-pay']
    int_features = [int(x) for x in request.form.values()]
    final_features = np.array(int_features).reshape(1,30)

    df=pd.DataFrame(data=final_features,columns=cols)
    
    df = df.reindex(columns=['age', 'fnlwgt', 'education', 'education-num', 'occupation',
       'capital-gain', 'capital-loss', 'hours-per-week', 'country',
       'sex_Female', 'sex_Male', 'race_Amer-Indian-Eskimo',
       'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White',
       'relationship_Husband', 'relationship_Not-in-family',
       'relationship_Other-relative', 'relationship_Own-child',
       'relationship_Unmarried', 'relationship_Wife', 'workclass_Federal-gov',
       'workclass_Local-gov', 'workclass_Never-worked', 'workclass_Private',
       'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc',
       'workclass_State-gov', 'workclass_Without-pay'])

   
    prediction = model.predict(df)
  

    output = prediction[0]
    for x in df.index:
        educationdic={0:'10th',1:'11th',2:'12th',3:'1st-4th',4:'5th-6th',5:'7th-8th',6:'9th',7:'Assoc-acdm',8:'Assoc-voc',9:'Bachelors',10:'Doctorate',11:'HS-grad',12:'Masters',13:'Preschool',14:'Prof-school',15:'Some-college'}
        new_ed=educationdic[df.loc[x,'education']]
      

        occupdic={4069:'Adm-clerical',10:'Armed-Forces',4289:'Craft-repair',4224:'Exec-managerial',1066:'Farming-fishing',1444:'Handlers-cleaners',2092:'Machine-op-inspct',3687:'Other-service',156:'Priv-house-serv',4317:'Prof-specialty',669:'Protective-serv',3885:'Sales',963:'Tech-support',1666:'Transport-moving'}
        new_occup=occupdic[df.loc[x,'occupation']]
        

        new_age=df.loc[x,'age']
        
        new_wgt=df.loc[x,'fnlwgt']
        new_educationnum=df.loc[x,'education-num']

        newrace=None
        for y in ['race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Black',
       'race_Other', 'race_White']:
            if df.loc[x,y]==1:
                newrace=y.strip('race_')

        newcg=df.loc[x,'capital-gain']
        newloss=df.loc[x,'capital-loss']
        newhrs=df.loc[x,'hours-per-week']

        newrelation=None
        for y in ['relationship_Husband',
       'relationship_Not-in-family', 'relationship_Other-relative',
       'relationship_Own-child', 'relationship_Unmarried', 'relationship_Wife']:
            if df.loc[x,y]==1:
                newrelation=y.strip('relationship_')
        
        newsex=None
        for y in ['sex_Female', 'sex_Male']:
            if df.loc[x,y]==1:
                newsex=y.strip('sex_')
        
        newworkclass=None
        for y in ['workclass_Federal-gov',
       'workclass_Local-gov', 'workclass_Never-worked', 'workclass_Private',
       'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc',
       'workclass_State-gov', 'workclass_Without-pay']:
            if df.loc[x,y]==1:
                newworkclass=y.strip('workclass_')

        country_dic={23:'Cambodia',121:'Canada',79:'China',59:'Columbia',95:'Cuba',70:'Dominican Republic',28:'Ecuador',106:'El Salvadorr',90:'England',29:'France',137:'Germany',29:'Greece',62:'Guatemala',44:'Haiti',1:'Holand-Netherlands',13:'Honduras',24:'Hong',13:'Hungary',107:'India',43:'Iran',24:'Ireland',73:'Italy',81:'Jamaica',63:'Japan',20:'Laos',651:'Mexico',34:'Nicaragua',14:'Outlying-US(Guam-USVI-etc)',31:'Peru',220:'Philippines',60:'Poland',37:'Portugal',116:'Puerto-Rico',12:'Scotland',88:'South',35:'Taiwan',53:'Thailand',19:'Trinadad&Tobago',29663:'United States',70:'Vietnam',16:'Yugoslavia'}    
        new_contry=country_dic[df.loc[x,'country']]       

  
    query="insert into TestyData (age,fnlwgt,education, education_num,occupation,capital_gain,capital_loss,hours_per_week,country,race,relationship,sex,workclass,prediction) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"    
    values=(int(new_age),int(new_wgt),new_ed,int(new_educationnum),new_occup,int(newcg),int(newloss),int(newhrs),new_contry,newrace,newrelation,newsex,newworkclass,output)
    print(connection)
    try: 
      cursor.execute(query,values) 
    except sql.Error as err:
      connection=sql.connect(host='us-cdbr-east-04.cleardb.com',user='b77648943f2114',password='517f5ad6',database='heroku_4fa29ab7f3558b6',connect_timeout=6000)
      cursor=connection.cursor()
      cursor.execute(query,values)
    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
