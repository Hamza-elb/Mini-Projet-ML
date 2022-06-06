from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.form['gender'] == 'Male' :
        data1 = 1
    else :
        data1 = 0

    if request.form['parent'] == 'wa7d' :
        data2 = 1
    elif request.form['parent'] == 'joj' :
        data2 = 2
    elif request.form['parent'] == 'tlata' :
        data2 = 3
    elif request.form['parent'] == 'reb3a':
        data2 = 4
    elif request.form['parent'] == 'khemsa':
        data2 = 5
    else :
        data2 = 6

    if request.form['preparation'] == 'none':
        data3 = 0
    else :
        data3 = 1

    if request.form['restoration'] == 'standard':
        data4 = 1
    else:
        data4 = 0

    data55 = request.form['score_math']
    data5=  float(data55)*5

    # arr = np.array([[data1, data2, data3]])
    pred = model.predict([[data1, data2, data4, data3, data5]])
    return render_template('home.html', data=pred*20/100)


if __name__ == "__main__":
    app.run(debug=True)