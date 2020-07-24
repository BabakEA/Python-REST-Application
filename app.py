import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import datetime
import os



app = Flask(__name__)
def model_Generator():
    if os.path.exists('./model/model.pkl'):
        print('A trained modle is existing....')
        model = pickle.load(open('model/model.pkl', 'rb'))
    else:
        print("training a New Model")
        import model_generatore as MG
        MG.Model_Generator()
        model = pickle.load(open('model/model.pkl', 'rb'))
    return model
    

@app.route('/')
def home():
    model=model_Generator()
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    #output = round(prediction[0], 2)
    output = prediction

    return render_template('index.html', prediction_text='the predicted class would be $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    model=model_Generator()

    app.run(debug=True)