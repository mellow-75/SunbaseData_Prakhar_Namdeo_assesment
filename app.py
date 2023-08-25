from flask import Flask,render_template,request
import pickle
import numpy as np
import sklearn

model=pickle.load(open('model.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def churn_predict():
    unnamed=float(request.form.get('unnamed'))
    age=float(request.form.get('AGE'))
    Subscription_months=float(request.form.get('Subscription_months'))
    Monthly_bill=float(request.form.get('Monthly_bill'))
    Total_usage_GB=float(request.form.get('Total_usage_GB'))
    female=float(request.form.get('female'))
    male=float(request.form.get('male'))
    chicago=float(request.form.get('chicago'))
    houston=float(request.form.get('houston'))
    losAngeles=float(request.form.get('losAngeles'))
    miami=float(request.form.get('miami'))
    NewYork=float(request.form.get('NewYork'))

    result=model.predict(np.array([unnamed,age,Subscription_months,Monthly_bill,Total_usage_GB,female,male,chicago,houston,losAngeles,miami,NewYork]).reshape(1,12))
    if result[0]==1:
        result="customer will churn"

    else:
        result="customer will not churn "
    return result
if __name__=='__main__':
    app.run(debug=True)
