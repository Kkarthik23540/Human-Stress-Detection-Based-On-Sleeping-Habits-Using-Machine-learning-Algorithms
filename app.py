import numpy as np
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, request, render_template
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

app = Flask(__name__)  #Initializing the flask App
stress = pickle.load(open('C:/Users/Karthik Reddy/Desktop/Human Stress Detection Based on Sleeping Habits using machine learning Algorithms/model/stress.pkl','rb'))

@app.route('/')  #default route

@app.route('/index')   # Route for the index page
def index():
	
	return render_template('index.html')
 
@app.route('/login')   # Route for the login page
def login():
	return render_template('login.html')

@app.route('/upload')    # Route for the upload page
def upload():
    return render_template('upload.html')  

@app.route('/preview',methods=["POST"])    # Route for the preview page
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)	

@app.route('/prediction', methods = ['GET', 'POST'])    # Route for the prediction page
def prediction():
    return render_template('prediction.html')

@app.route('/predict',methods=['POST'])       # Route for the predict page
def predict():
	int_feature = [x for x in request.form.values()]
	 
	final_features = [np.array(int_feature)]
	 
	result=stress.predict(final_features)
	if result == 0:
		result = "Low/Normal"
	elif  result == 1:
		result = "Medium Low"
	elif result == 2:
		result = "Medium"
	elif result == 3:
		result = "Medium High"
	elif result == 4:
		result = "High"			

	return render_template('prediction.html', prediction_text= result)
    
@app.route('/performance')      # Route for the performance page
def performance():
	return render_template('performance.html')
    
@app.route('/chart')     # Route for the chart page
def chart():
	return render_template('chart.html')

@app.route('/logout')     # Route for the logout page
def logout():
	return render_template('logout.html')    
    
if __name__ == "__main__":  # Running the app
	app.run(debug=False)  # Starting of server