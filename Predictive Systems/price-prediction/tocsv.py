import pandas as pd

# Define the data
data = [
    ["Paddy", "Common", 1310, 1360, 1410, 1470, 1550, 1750, 1815, 1868, 1940, 2040, 2183, 2300, "West Bengal", 1800],
    ["Paddy", "Grade A", 1345, 1400, 1450, 1510, 1590, 1770, 1835, 1888, 1960, 2060, 2203, 2320, "West Bengal", 1800],
    ["Jowar", "Hybrid", 1500, 1530, 1570, 1625, 1700, 2430, 2550, 2620, 2738, 2970, 3180, 3371, "Maharashtra", 1200],
    ["Jowar", "Maldandi", 1520, 1550, 1590, 1650, 1725, 2450, 2570, 2640, 2758, 2990, 3225, 3421, "Maharashtra", 1200],
    ["Bajra", "", 1250, 1250, 1275, 1330, 1425, 1950, 2000, 2150, 2250, 2350, 2500, 2625, "Rajasthan", 400],
    ["Ragi", "", 1500, 1550, 1650, 1725, 1900, 2897, 3150, 3295, 3377, 3578, 3846, 4290, "Karnataka", 900],
    ["Maize", "", 1310, 1310, 1325, 1365, 1425, 1700, 1760, 1850, 1870, 1962, 2090, 2225, "Karnataka", 900],
    ["Tur (Arhar)", "", 4300, 4350, 4625, 5050, 5450, 5675, 5800, 6000, 6300, 6600, 7000, 7550, "Madhya Pradesh", 1100],
    ["Moong", "", 4500, 4600, 4850, 5225, 5575, 6975, 7050, 7196, 7275, 7755, 8558, 8682, "Rajasthan", 400],
    ["Urad", "", 4300, 4350, 4625, 5000, 5400, 5600, 5700, 6000, 6300, 6600, 6950, 7400, "Madhya Pradesh", 1100],
]

# Create a DataFrame
columns = ["Commodity", "Variety", "2013-14", "2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
           "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "State", "Annual Rainfall (mm)"]

df = pd.DataFrame(data, columns=columns)

# Save to CSV file
df.to_csv("updated_crop_prices.csv", index=False)

print("CSV file 'updated_crop_prices.csv' has been created successfully!")


