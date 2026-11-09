# Booking Cancellation Predictor

Predicts whether a hotel booking will be canceled and shows cancellation patterns on a dashboard.

## Quick start on Windows 10

1. Download hotel-predictor.zip, right-click it, choose “Extract All,” and click “Extract.”
2. Open the extracted hotel-predictor folder in File Explorer, click the address bar, type cmd, and press Enter. A Command Prompt window opens in that folder.
3. Create a virtual environment inside the project folder by typing the following command and pressing Enter: Python -m venv .venv
4. Activate the virtual environment by typing .venv\Scripts\activate and pressing Enter. The prompt then begins with (.venv), and every subsequent command runs in the virtual environment.
5. Install the required libraries by typing the following command and pressing Enter: Python -m pip install -r requirements.txt
6. Start the application by typing the following command and pressing Enter: Python -m streamlit run app.py. The default web browser opens the app at http://localhost:8501. If it does not, open that address manually. If Streamlit asks for an email address, press Enter to skip it.
7. Sign in with username manager and password Ventira2026!
8. On the Dashboard page, use the Hotel and Market segment filters in the sidebar to update the charts, and open “Explore the data” to see summary statistics, correlations, and sample bookings.
9. Select “Check a booking” in the sidebar, enter the booking details, and click “Check risk.”
10. Select “Model accuracy” to review the test results, confusion matrix, top factors, hypothesis results, and tuning results.
11. Select “Monitoring” to see the model’s status, the logged predictions, and the activity log, or to retrain the model.
12. Optionally, run the unit tests by typing Python -m pytest in the Command Prompt.
13. To stop the application, return to the Command Prompt and press Ctrl+C. To start it again later, open a Command Prompt in the project folder, activate the virtual environment as in step 5, and run the start command from step 7.

## Maintenance

- Run the unit tests: `python -m pytest`
- Add a user or change a password: `python security.py add <username>`
- Retrain from the command line: `python train_model.py` (about one minute). The trained model is already included.
- Sign-ins and predictions are written to `logs/app.log`, and every prediction is also saved in the `predictions` table of `app.db`.

## Files

- `app.py`: the dashboard
- `train_model.py`: cleans the data, trains and evaluates the models, tests the hypotheses, saves the results
- `security.py`: password hashing (PBKDF2-SHA256) and user management
- `database.py`: SQLite connection and tables
- `app.db`: SQLite database with user accounts (salted password hashes only) and the prediction log
- `tests/test_app.py`: unit tests
- `data/hotel_bookings_raw.csv`: Hotel Booking Demand dataset (Antonio, Almeida, and Nunes, 2019)
- `data/hotel_bookings_clean.csv`: cleaned data produced by `train_model.py`
- `models/model.joblib`, `models/metrics.json`: trained model and its test results



