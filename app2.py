import joblib

# ... (your existing training code) ...

# Save the trained Decision Tree model
joblib.dump(dt, 'final_model.joblib')

# ALSO SAVE THE PREPROCESSOR!
joblib.dump(preprocessor, 'preprocessor.joblib') # Add this line

print("Model and preprocessor saved successfully!")
