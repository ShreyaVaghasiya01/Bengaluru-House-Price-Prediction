# 🏠 Bengaluru House Price Prediction

This is a simple **Machine Learning project** that predicts the price of a house in Bengaluru based on key features like area type, location, size, bathrooms, balconies, and availability. The project uses a **Linear Regression model** trained on the Bengaluru House Price dataset.

---

## Features

- Predict house prices in **lakhs** based on user inputs.
- Handles **categorical features** like area type, location, BHK size, and availability using one-hot encoding.
- Clean and interactive **Streamlit web app** for easy usage.
- Quick and accurate predictions for potential buyers, sellers, and real estate enthusiasts.

---

## Dataset

- The dataset used: `Bengaluru_House_Data.csv`
- Features include:
  - `area_type`
  - `location`
  - `Size` (BHK)
  - `total_sqft`
  - `bath`
  - `balcony`
  - `availability`
  - `price` (target variable)

---

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/ShreyaVaghasiya01/Bengaluru-House-Price-Prediction.git
    
    ```

2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

4. Open the URL shown in the terminal (usually `http://localhost:8501`) to interact with the app.


## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib
- streamlit

