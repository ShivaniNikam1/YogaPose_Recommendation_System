# from flask import Flask, render_template, request
# app = Flask(__name__)
# import pickle

# # Load the model
# with open('E:/flask_tutorial/yogarecom/trained.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

# data = ['Option 1', 'Option 2', 'Option 3', 'Option 4']


# @app.route('/')
# def hello_world():
#     return 'Hello, World!'

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     # Get the selected option from the form
#     selected_option = request.form['selected_option']

#     # Process the selected option using your model
#     # Replace this with your actual model logic
#     # For demonstration purposes, just return the selected option
#     recommended_output = selected_option

#     # Render a template to display the recommendation
#     return render_template('recommendation.html', recommended_output=recommended_output)


# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, render_template, request
# import pandas as pd
# import pickle

# app = Flask(__name__)

# # Load the trained model
# with open('E:/flask_tutorial/yogarecom/trained.pkl', 'rb') as f:
#     model = pickle.load(f)

# # Load the data
# data = pd.read_csv('E:/flask_tutorial/yogarecom/final_asan1_1.csv')

# @app.route('/')
# def index():
#     # Get unique values from the 'Benefits' column for dropdown options
#     dropdown_options = data['Benefits'].unique()
#     return render_template('recommendation.html', options=dropdown_options)

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     # Get the selected pain from the dropdown
#     pain_name = request.form['pain']

#     # Use the trained model to make recommendations
#     recommended_poses = model.predict(pain_name)

#     if recommended_poses:
#         return render_template('recommendation.html', poses=recommended_poses, selected_pain=pain_name)
#     else:
#         return render_template('recommendation.html', message="No recommendations available for the selected pain.", selected_pain=pain_name)

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, render_template, request
# import pandas as pd
# import difflib
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# app = Flask(__name__)

# # Load the dataset
# data = pd.read_csv('E:/flask_tutorial/yogarecom/final_asan1_1.csv')

# # Preprocess the data
# columns = ['AName', 'Description', 'Benefits']
# for feature in columns:
#     data[feature] = data[feature].fillna('')
# combined_features = data['AName'] + ' ' + data['Description'] + ' ' + data['Benefits']

# # Vectorize the text data
# vectorizer = TfidfVectorizer()
# feature_vector = vectorizer.fit_transform(combined_features)
# similarity = cosine_similarity(feature_vector)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     pain_name = request.form['pain']
#     list_of_all_benefits = data['Benefits'].tolist()
#     find_close_match = difflib.get_close_matches(pain_name, list_of_all_benefits)
#     if find_close_match:
#         close_match = find_close_match[0]
#         index_of_the_benefit = data[data['Benefits'] == close_match].index[0]
#         similarity_score = list(enumerate(similarity[index_of_the_benefit]))
#         sorted_similar_poses = sorted(similarity_score, key=lambda x: x[1], reverse=True)
#         recommended_poses = [data.loc[index, 'AName'] for index, _ in sorted_similar_poses[:5]]
#         return render_template('recommend.html', poses=recommended_poses)
#     else:
#         return render_template('recommend.html', message="No recommendations available for the selected pain.")

# if __name__ == '__main__':
#     app.run(debug=True)

# ------------------------------------------------------------------------------------------------------------------
from flask import Flask, render_template, request
import pandas as pd
import difflib
import pickle

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('E:/flask_tutorial/yogarecom/yoga - Sheet1 (5).csv')

# Path to your pickled model (assuming it's in the same directory as your script)
# model_path = 'trained_model.pkl'

# # Load the model components
# with open(model_path, 'rb') as f:
#   model_components = pickle.load(f)


# data['Pain'] = data['Pain'].fillna('')
# Load the trained model
# with open('E:/flask_tutorial/yogarecom/trained_model.pkl', 'rb') as f:
#     model = pickle.load(f)
    
def load_model_components():
    """Loads the Asana recommendation model components from a pickle file.

    Returns:
        dict: A dictionary containing the loaded model components (vectorizer, similarity, data).
    """

    with open('final_trained_model.pkl', 'rb') as f:
        model_components = pickle.load(f)
    return model_components

# # Load model components (optional, call before using them)
# vectorizer = None
# similarity = None
# data = None  # Optional
model_components = load_model_components()  # Uncomment to load on startup

@app.route('/')
def index():
    return render_template('index.html', data=data)

@app.route('/recommend', methods=['POST'])
def recommend():
    global vectorizer, similarity, data  # Declare and modify global variables

    vectorizer = model_components['vectorizer']
    similarity = model_components['similarity']
    data = model_components['data']  # Access components from dictionary

    pain_name = request.form['pain']
    list_of_all_pain = data['Pain'].tolist()
    find_close_match = difflib.get_close_matches(pain_name, list_of_all_pain)
    
    if find_close_match:
        close_match = find_close_match[0]
        index_of_close_match = data[data['Pain'] == close_match].index[0]
        similarity_scores = list(enumerate(similarity[index_of_close_match]))
        sorted_similar_poses = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get recommended poses and their image URLs
        recommended_data = []
        for index, _ in sorted_similar_poses[:6]:
            pose_name = data.loc[index, 'AName']
            image_url = data.loc[index, 'Photo']  # Assuming 'ImageURL' is the column containing image URLs
            recommended_data.append({'pose': pose_name, 'image': image_url})

        return render_template('recommend.html', data=recommended_data)
    else:
        return render_template('recommend.html', message="No recommendations available for the selected pain.")
if __name__ == '__main__':
    app.run(debug=True)
