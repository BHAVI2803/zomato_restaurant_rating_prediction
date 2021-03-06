import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template
import flask
import pickle
import bz2file as bz2


app = Flask(__name__, template_folder='templates',static_folder='static',static_url_path='/static')

def decompress_pickle(file):

    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data

model1 = pickle.load(open('model1.pkl', 'rb'))
#model2 = pickle.load(open('model2.pkl', 'rb'))
model2= decompress_pickle('model2.pbz2')
model3=pickle.load(open('model3.pkl','rb'))


   

@app.route('/')
def home():
    return (flask.render_template('index.html'))

@app.route('/prediction',methods=['POST', 'GET'])
def prediction():
    if flask.request.method == 'GET':
        return (flask.render_template('prediction.html'))
    if flask.request.method == 'POST':
    

        location_df=pd.read_csv("location.csv",index_col='location')

        rest_type_df=pd.read_csv("rest_type.csv",index_col='rest_type')

        cuisines_df=pd.read_csv("cuisines.csv",index_col='cuisines')

        listed_type_df=pd.read_csv("listed_type.csv",index_col='listed_in')



        online_order = int(request.form.get('online_order').replace("YES", '1').replace('NO', '0'))

        book_table = int(request.form.get('book_table').replace("YES", '1').replace('NO', '0'))

        votes=int(request.form.get('votes'))
        
        cost=int(request.form.get('cost'))

        location=request.form.get('location')
        if location in location_df.index:
            loc_code=int(location_df[location_df.index==location].location_code[0])


        Cuisines=request.form.get('Cuisines')
        if Cuisines in cuisines_df.index:
            cuis_code=int(cuisines_df[cuisines_df.index==Cuisines].cuisines_code[0])


        rest_type=request.form.get('rest_type')
        if rest_type in rest_type_df.index:
            rest_type_code=int(rest_type_df[rest_type_df.index==rest_type].rest_type_code[0])

        rest_style_list=request.form.get('rest_style_list')
        if rest_style_list in listed_type_df.index:
            listed_type_code=int(listed_type_df[listed_type_df.index==rest_style_list].listed_type_code[0])

        features=np.array([online_order,book_table,votes,cost,loc_code,cuis_code,rest_type_code,listed_type_code]).reshape(-1, 8)
        #print(features)

            # scaling our values before prediction
            # reading the train dataset 
        x_train=pd.read_csv("X_train.csv",usecols=range(1,9))
        cols=x_train.columns
        
        input_feature=pd.DataFrame(features,columns=cols)
        
        non_binary_cols=list(x_train.columns)
        non_binary_cols.remove('online_order')
        non_binary_cols.remove('book_table')
        
        
        scaler = StandardScaler()
        scaler.fit_transform(x_train[non_binary_cols])
        input_feature[non_binary_cols]= scaler.transform(input_feature[non_binary_cols])
        #final=np.append(features[:,0:2],feature_buffer).reshape(-1,8)
        
        predict1=model1.predict(input_feature)
        predict2=model2.predict(input_feature)
        predict3=model3.predict(input_feature)
        
        print(predict1,predict2,predict3)

        prediction = predict1+predict2+predict3/3



        output = round(prediction[0], 2)

        return render_template('prediction.html', prediction_text='Restaurant Rating would be  {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)