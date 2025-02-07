import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")
    return X, y

# Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

# Predict species
def predict_species(model, input_data):
    prediction = model.predict([input_data])
    return prediction

# Streamlit app
def main():
    st.title("Iris Species Prediction")

    # Load data
    X, y = load_data()

    # Display dataset
    if st.checkbox("Show dataset"):
        st.write(X)

    # Train the model
    model, accuracy = train_model(X, y)
    st.write(f"Model Accuracy: {accuracy:.2f}")

    # User input for prediction
    st.subheader("Enter flower measurements for prediction:")
    sepal_length = st.slider("Sepal Length (cm)", float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()))
    sepal_width = st.slider("Sepal Width (cm)", float(X['sepal width (cm)'].min()), float(X['sepal width (cm)'].max()))
    petal_length = st.slider("Petal Length (cm)", float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()))
    petal_width = st.slider("Petal Width (cm)", float(X['petal width (cm)'].min()), float(X['petal width (cm)'].max()))

    # Button to predict
    if st.button("Predict"):
        input_data = [sepal_length, sepal_width, petal_length, petal_width]
        prediction = predict_species(model, input_data)

        iris = load_iris()
        st.write(f"Predicted species: {iris.target_names[prediction][0]}")

if __name__ == '__main__':
    main()


st.title("demo session of GEC  K R PETE")

