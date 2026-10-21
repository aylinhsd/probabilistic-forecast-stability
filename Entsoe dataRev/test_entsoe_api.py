
import os
import pandas as pd
from entsoe import EntsoePandasClient

# 1. Read API key from environment variable
api_key = os.environ.get("ENTSOE_API_KEY")

if not api_key:
    raise RuntimeError(
        "ENTSOE_API_KEY is missing. "
        "Set it in your terminal first."
    )

# 2. Connect to ENTSO-E
client = EntsoePandasClient(api_key=api_key)

# 3. Define the period to download
# A small sample from 2023 for testing
start = pd.Timestamp("2023-01-01", tz="Europe/Rome")
end = pd.Timestamp("2023-01-08", tz="Europe/Rome")

# 4. Download day-ahead prices for Italy North
prices = client.query_day_ahead_prices(
    country_code="IT_NORD",
    start=start,
    end=end
)

# 5. Display results
print("\nFirst 10 observations:")
print(prices.head(10))

print("\nTotal observations:", len(prices))
print("First timestamp:", prices.index.min())
print("Last timestamp:", prices.index.max())

# 6. Save sample to CSV
prices.to_csv("entsoe_api_test.csv")

print("\nSuccess! Data saved to entsoe_api_test.csv")
