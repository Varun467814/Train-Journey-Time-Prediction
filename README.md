# Train Journey Prediction

This project predicts train journey times using machine learning.

## Setup

1. Install dependencies: pip install -r requirements.txt
2. Prepare data in the data/ folder
3. Train a model and save it using the helper in `src/train_model.py`.
   Example:
   ```python
   from src.train_model import train_model, save_model

   model = train_model(X_train, y_train)
   save_model(model, "saved_model.pkl")
   ```
4. Run the Streamlit app: python app.py

## Project Structure

- data/ - Data files
- src/ - Source code
- notebooks/ - Jupyter notebooks for exploration
