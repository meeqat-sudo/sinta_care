# How to Run the Topic Modeling Application

## Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

## Installation Steps

1. Install required packages:
   pip install -r requirements.txt

2. Prepare your data:
   - Place your CSV file with customer reviews in the project folder
   - Name it webmd_dataset.csv or update the FILE_PATH variable in the code

3. Add your logo:
   - Place a logo image in the project folder
   - Name it logo.png or update the LOGO_PATH variable in the code

## Running the Application

1. Start the Streamlit server:
   streamlit run app.py

2. Access the application:
   - Open your web browser
   - Go to http://localhost:8501 (the URL shown in your terminal)

## Using the Application

1. Data Viewer Page:
   - View your dataset
   - Apply filters using the sidebar

2. Topic Analysis Page:
   - Click "Run Topic Analysis" to process your reviews
   - Explore the interactive visualizations
   - Click on topics to see sample reviews
   - Download results as CSV

## Note
The first run will download the language model (~80MB) which may take a few minutes.