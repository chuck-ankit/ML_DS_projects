# Load the saved model using pickle
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Now, you can use the loaded_model for making predictions on new data
new_data = pd.DataFrame(...)  # Provide your new data
predictions = loaded_model.predict(new_data)