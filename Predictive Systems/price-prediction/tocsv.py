import pandas as pd

# Expanded data with more states and commodities
data = [
    # Punjab
    ["Wheat", "Common", 1350, 1400, 1450, 1525, 1625, 1735, 1840, 1925, 2015, 2125, 2275, 2425, "Punjab", 650],
    ["Rice", "Basmati", 2500, 2600, 2750, 2900, 3100, 3300, 3500, 3700, 3900, 4100, 4350, 4600, "Punjab", 650],
    ["Maize", "Hybrid", 1310, 1360, 1400, 1450, 1520, 1600, 1700, 1800, 1900, 2000, 2225, 2500, "Punjab", 650],
    ["Paddy", "Common", 1310, 1360, 1410, 1470, 1550, 1750, 1815, 1868, 1940, 2040, 2183, 2325, "Punjab", 650],

    # Haryana
    ["Wheat", "Common", 1350, 1400, 1450, 1525, 1625, 1735, 1840, 1925, 2015, 2125, 2275, 2425, "Haryana", 600],
    ["Rice", "Pusa 1121", 2400, 2500, 2650, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4250, 4500, "Haryana", 600],
    ["Bajra", "Hybrid", 1250, 1300, 1350, 1400, 1475, 1550, 1650, 1750, 1850, 1950, 2150, 2350, "Haryana", 600],
    ["Mustard", "Common", 2500, 2600, 2700, 2800, 2950, 3100, 3250, 3400, 3550, 3700, 3850, 4000, "Haryana", 600],

    # Karnataka
    ["Ragi", "Common", 1500, 1550, 1650, 1725, 1900, 2100, 2500, 2897, 3150, 3378, 3846, 4290, "Karnataka", 900],
    ["Maize", "Hybrid", 1310, 1360, 1400, 1450, 1520, 1600, 1700, 1760, 1850, 1962, 2090, 2345, "Karnataka", 900],
    ["Tur (Arhar)", "Common", 4300, 4350, 4625, 5050, 5450, 5675, 5800, 6000, 6300, 6600, 7000, 7550, "Karnataka", 900],
    ["Jowar", "Hybrid", 1550, 1575, 1625, 1710, 1850, 2100, 2200, 2350, 2550, 2750, 2975, 3180, "Karnataka", 900],

    # Andhra Pradesh
    ["Paddy", "Common", 1310, 1360, 1410, 1470, 1550, 1750, 1815, 1868, 1940, 2040, 2183, 2325, "Andhra Pradesh", 1200],
    ["Groundnut", "Common", 4000, 4100, 4200, 4300, 4450, 4550, 4650, 4750, 4850, 4950, 5050, 5150, "Andhra Pradesh", 1200],
    ["Tur (Arhar)", "Common", 4300, 4350, 4625, 5050, 5450, 5675, 5800, 6000, 6300, 6600, 7000, 7550, "Andhra Pradesh", 1200],
    ["Maize", "Hybrid", 1310, 1360, 1400, 1450, 1520, 1600, 1700, 1800, 1900, 2000, 2225, 2500, "Andhra Pradesh", 1200],

    # Telangana
    ["Paddy", "Fine", 1360, 1410, 1470, 1550, 1650, 1750, 1830, 1880, 1950, 2050, 2200, 2350, "Telangana", 1100],
    ["Cotton", "Medium Staple", 3700, 3750, 3800, 3860, 4020, 4220, 4420, 4620, 4820, 5020, 5220, 5420, "Telangana", 1100],
    ["Maize", "Hybrid", 1310, 1360, 1400, 1450, 1520, 1600, 1700, 1800, 1900, 2000, 2225, 2500, "Telangana", 1100],
    ["Soybean", "Common", 2600, 2750, 2900, 3050, 3390, 3600, 3750, 3880, 3950, 4300, 4600, 4800, "Telangana", 1100],

    # Maharashtra
    ["Sugarcane", "Common", 210, 220, 230, 240, 255, 275, 285, 295, 305, 315, 325, 335, "Maharashtra", 850],
    ["Cotton", "Medium Staple", 3700, 3750, 3800, 3860, 4020, 4220, 4420, 4620, 4820, 5020, 5220, 5420, "Maharashtra", 850],
    ["Soybean", "Common", 2600, 2750, 2900, 3050, 3390, 3600, 3750, 3880, 3950, 4300, 4600, 4800, "Maharashtra", 850],
    ["Jowar", "Hybrid", 1550, 1575, 1625, 1710, 1850, 2100, 2200, 2350, 2550, 2750, 2975, 3180, "Maharashtra", 850],

    # Madhya Pradesh
    ["Wheat", "Common", 1350, 1400, 1450, 1525, 1625, 1735, 1840, 1925, 2015, 2125, 2275, 2425, "Madhya Pradesh", 700],
    ["Soybean", "Common", 2600, 2750, 2900, 3050, 3390, 3600, 3750, 3880, 3950, 4300, 4600, 4800, "Madhya Pradesh", 700],
    ["Mustard", "Common", 2500, 2600, 2700, 2800, 2950, 3100, 3250, 3400, 3550, 3700, 3850, 4000, "Madhya Pradesh", 700],
    ["Urad", "Common", 4000, 4100, 4350, 4625, 5050, 5450, 5625, 5800, 6000, 6300, 6600, 6950, "Madhya Pradesh", 700],

    # Tamil Nadu
    ["Paddy", "Common", 1310, 1360, 1410, 1470, 1550, 1750, 1815, 1868, 1940, 2040, 2183, 2325, "Tamil Nadu", 1300],
    ["Sugarcane", "Common", 210, 220, 230, 240, 255, 275, 285, 295, 305, 315, 325, 335, "Tamil Nadu", 1300],
    ["Groundnut", "Common", 4000, 4100, 4200, 4300, 4450, 4550, 4650, 4750, 4850, 4950, 5050, 5150, "Tamil Nadu", 1300],
    ["Sunflower", "Hybrid", 3600, 3700, 3850, 4000, 4340, 4510, 4700, 4880, 5100, 5400, 5650, 5880, "Tamil Nadu", 1300],

    # Gujarat
    ["Cotton", "Medium Staple", 3700, 3750, 3800, 3860, 4020, 4220, 4420, 4620, 4820, 5020, 5220, 5420, "Gujarat", 800],
    ["Groundnut", "Common", 4000, 4100, 4200, 4300, 4450, 4550, 4650, 4750, 4850, 4950, 5050, 5150, "Gujarat", 800],
    ["Tur (Arhar)", "Hybrid", 4300, 4350, 4625, 5050, 5450, 5675, 5800, 6000, 6300, 6600, 7000, 7550, "Gujarat", 800],
    ["Mustard", "Common", 2500, 2600, 2700, 2800, 2950, 3100, 3250, 3400, 3550, 3700, 3850, 4000, "Gujarat", 800],

    # Uttar Pradesh
    ["Wheat", "Common", 1350, 1400, 1450, 1525, 1625, 1735, 1840, 1925, 2015, 2125, 2275, 2425, "Uttar Pradesh", 1000],
    ["Sugarcane", "Common", 210, 220, 230, 240, 255, 275, 285, 295, 305, 315, 325, 335, "Uttar Pradesh", 1000],
    ["Paddy", "Fine", 1310, 1360, 1410, 1470, 1550, 1750, 1815, 1868, 1940, 2040, 2183, 2300, "Uttar Pradesh", 1000],
    ["Mustard", "Common", 2500, 2600, 2700, 2800, 2950, 3100, 3250, 3400, 3550, 3700, 3850, 4000, "Uttar Pradesh", 1000],

    # Rajasthan
    ["Bajra", "Hybrid", 1250, 1300, 1350, 1400, 1475, 1550, 1650, 1750, 1850, 1950, 2150, 2350, "Rajasthan", 400],
    ["Moong", "Green Gram", 4500, 4600, 4850, 5225, 5575, 5800, 6200, 6975, 7196, 7755, 8558, 8682, "Rajasthan", 400],
    ["Mustard", "Common", 2500, 2600, 2700, 2800, 2950, 3100, 3250, 3400, 3550, 3700, 3850, 4000, "Rajasthan", 400],
    ["Wheat", "Common", 1350, 1400, 1450, 1525, 1625, 1735, 1840, 1925, 2015, 2125, 2275, 2425, "Rajasthan", 400],
]

# Create a DataFrame
columns = ["Commodity", "Variety", "2013-14", "2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
           "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "State", "Annual Rainfall (mm)"]

df = pd.DataFrame(data, columns=columns)

# Save to CSV file
df.to_csv("expanded_crop_prices_extended.csv", index=False)

print("CSV file 'expanded_crop_prices_extended.csv' has been created successfully!")