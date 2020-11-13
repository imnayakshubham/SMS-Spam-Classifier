from flask import Flask,render_template,request
import pickle

model = pickle.load(open('spamdetectmodel2.pkl','rb'))
tfidf = pickle.load(open('tfidf1.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict',methods = ["Post"])
def predict():
    if request.method == 'POST':
        rawtext = request.form['textarea']
        msg = [rawtext]
        vector = tfidf.transform(msg).toarray()
        prediction = model.predict(vector)

    return render_template("index.html",result = prediction)


if __name__ == "__main__" :
    app.run(debug=(True))