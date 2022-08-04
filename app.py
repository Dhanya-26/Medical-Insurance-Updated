from flask import Flask, render_template, url_for, redirect,request
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user,  LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt

# //prediction code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# //prediction code

app = Flask(__name__)

# //prediction code
# # loading the data from csv file to a pandas DataFrame
insurance_dataset = pd.read_csv('insurance.csv')
# # encoding sex column
insurance_dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)

# encoding 'smoker' column
insurance_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)

# # encoding 'region' column
insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)

X = insurance_dataset.drop(columns='charges', axis=1)
#X = insurance_dataset.drop(columns='children', axis=1)
Y = insurance_dataset['charges']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
# loading the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train.values, Y_train.values)
# prediction on training data
training_data_prediction =regressor.predict(X_train)
# R squared value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
# prediction on test data
test_data_prediction =regressor.predict(X_test)
# R squared value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
# //prediction code


# creates database instance
db = SQLAlchemy(app) 
bcrypt  = Bcrypt(app)
# connect app file to database file
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'Thisisasecretkey'

# allows our app and flask login to work together to handle things when logging in
# loading users from ids etc.
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# reload the user objects from he user id stored in the session
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# table for database with three columns
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        # checking in user table if the username.data which is entered already in db
        existing_user_username = User.query.filter_by(
            username=username.data).first()

        # if already in database then error
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')

class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        # check if the user is in the database or not
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            # checks if they match, then log in
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))
    
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    
    if form.validate_on_submit():
        # hashing the entered password
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        # creating a new user
        new_user = User(username=form.username.data, password=hashed_password)
        # add the chnages to database
        db.session.add(new_user)
        # commit the chnages to db
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html',form=form)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template('predict.html')

@app.route('/submit',methods=['POST','GET'])
def submit():
    if request.method=='POST':
        age=float(request.form['age'])
        gender1=request.form['gender']
        if gender1=='Male':
            gender=0
        else:
            gender=1
        
        bmi=float(request.form['bmi'])
        #children=float(request.form['children'])
        smoker1=request.form['gender']
        if smoker1=='Yes':
            smoker=0
        else:
            smoker=1
        
        region1=request.form['region']
        if region1=='Southeast':
            region=1
        elif region1=='Northeast':
            region=2
        elif region1=='Southwest':
            region=3
        else:
            region=4
        
        input_data= (age,gender,bmi,smoker,region)
        # changing input_data to a numpy array
        input_data_as_numpy_array = np.asarray(input_data,dtype='float64')

        # reshape the array
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = regressor.predict(input_data_reshaped)
    return render_template('result.html',result=str(round(prediction[0], 2)))

if __name__ == '__main__':
    app.run(debug=True)