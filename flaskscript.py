#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, render_template,url_for,redirect
from sklearn import pipeline
import re
import cPickle as pickle

#create app
app = Flask(__name__, template_folder='template')

filename='test_LR_test.pkl'
model=pickle.load(open(filename,'rb'))

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method=='POST':
    words=request.form['input']
     if words:
            temp=re.split(';|,|\n|\t',str(words).strip())
            words=[word.strip() for word in temp if word.strip()]
            if len(words) > 0:
                lr_pred_test=model.predict(words)   
                Y_Pro=model.predict_proba(words)
                confidence={}
                for i in range(len(Y_Pro)):
                    confidence[i]="{:0.2f}".format(100*max(Y_Pro[i])) 
                return redirect(url_for('index')) 

if __name__ == '__main__':
    app.run()


# In[ ]:




