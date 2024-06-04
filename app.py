from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the pre-trained IMDb rating prediction model
with open("imdb_prediction.pkl",'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/vn')
def nare():
    return "This is Naresh"

@app.route('/')
def naresh():
    return render_template("index.html")


@app.route('/predict', methods=['GET','POST'])
def predict():
    # Extracting values from the form
    duration = int(request.form['duration'])
    budget = int(request.form['budget'])
    gross = int(request.form['gross'])
    profit = int(request.form['profit'])
    Profit_Percentage = int(request.form['Profit_Percentage'])
    num_critic_for_reviews = int(request.form['num_critic_for_reviews'])
    num_user_for_reviews = int(request.form['num_user_for_reviews'])
    num_voted_users = int(request.form['num_voted_users'])
    director_facebook_likes = int(request.form['director_facebook_likes'])
    cast_total_facebook_likes = int(request.form['cast_total_facebook_likes'])
    actor_1_facebook_likes = int(request.form['actor_1_facebook_likes'])
    actor_2_facebook_likes = int(request.form['actor_2_facebook_likes'])
    actor_3_facebook_likes = int(request.form['actor_3_facebook_likes'])
    movie_facebook_likes = int(request.form['movie_facebook_likes'])

    # Creating an input array for prediction
    input_data = np.array([[duration,budget,gross,profit,Profit_Percentage,
                            num_critic_for_reviews,num_user_for_reviews,num_voted_users,
                            director_facebook_likes,cast_total_facebook_likes,
                            actor_1_facebook_likes,actor_2_facebook_likes,actor_3_facebook_likes,
                            movie_facebook_likes]])

    # Making predictions on the user input
    predicted_rating = model.predict(input_data)[0]
    predict= round(predicted_rating,1)
    # Rendering the template with the prediction result
    return render_template("index.html", predict=predict)

app.run()