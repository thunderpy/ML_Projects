# Load model of support ticket

# Mounting Google Drive locally

#from google.colab import drive

#drive.mount('/content/gdrive')

#saveModelPath = '/content/gdrive/My Drive/SupportTicketData'

#ls -la '/content/gdrive/My Drive/SupportTicketData'

# To download file

#from google.colab import files

# files.download('/content/gdrive/My Drive/SupportTicketData/test.csv')

# Import required models

import pickle
import numpy as np
import pandas as pd
import sklearn
from flask import Flask, request, jsonify

# Read split test set
test_df = pd.read_csv('test.csv')

test_df.head()

# New mail Eg.
mail = "Internet Issue"

# Using Flask to create API.

app = Flask(__name__)

# Load CountVectorizer Model
loadCountVectorizer = pickle.load(open('savedModels/CountVectorizerModel.pkl', 'rb'))

# load the model
loaded_model = pickle.load(open('savedModels/support_ticket_model.pkl', 'rb'))

# Load label encorder model
loadLabelencorder = pickle.load(open('savedModels/labelEncorderModel.pkl', 'rb'))

@app.route('/index', methods=['POST'])
def index():
    data = request.get_json()
    text = data.get('text')
    predition = newValueCheck(text)
    return jsonify({'Predition':predition})

# Create a function to check new value from saved model.
def newValueCheck(text):
    x_val = loadCountVectorizer.transform([text])
    # transform:- Extract token counts out of raw text documents using the vocabulary fitted with fit or the one provided to the constructor.
    # transform:- Transform documents to document-term matrix.
    x_val = x_val.toarray()

    # Predict text
    model_output = loaded_model.predict(x_val)
    # print('Model output:- ', model_output)

    # inverse_transform label encorder
    result = loadLabelencorder.inverse_transform(model_output)
    # inverse_transform:- Transform labels back to original encoding.

    return result[0]

newValueCheck(mail)

# Create new colum for Predicted values
test_df['Predictions'] = test_df['Description'].apply(newValueCheck)

test_df.head()

# Verify Category and Predition colum.
test_df['Verify'] = test_df['Category'] == test_df['Predictions']

test_df.head()

totalValue = test_df.shape[0]
trueValue = (test_df['Verify'] == True).sum()
falseValue = (test_df['Verify'] == False).sum()

print('Total False value is {} and True value is {} out of total test value {}.'.format(falseValue, trueValue, totalValue))

# pandas groupby
test_df.groupby(('Verify')).count()

# test_df.to_csv('testPredicted.csv')


# Run Flask 
app.run(debug=True)