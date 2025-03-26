import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect("users.db")

# Fetch users
users_df = pd.read_sql_query("SELECT * FROM users", conn)
print("Registered Users:\n", users_df)

# Fetch predictions
predictions_df = pd.read_sql_query("SELECT * FROM predictions", conn)
print("\nStored Predictions:\n", predictions_df)

# Close connection
conn.close()
