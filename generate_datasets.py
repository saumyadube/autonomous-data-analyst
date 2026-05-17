import pandas as pd, numpy as np, os

rng = np.random.default_rng(42)
os.makedirs('data/sample', exist_ok=True)

datasets = {}

# ── 1. Titanic ────────────────────────────────────────────────
n=891
datasets['titanic'] = pd.DataFrame({
    'PassengerId': range(1,n+1),
    'Survived': rng.integers(0,2,n),
    'Pclass': rng.choice([1,2,3],n,p=[0.24,0.21,0.55]),
    'Sex': rng.choice(['male','female'],n,p=[0.65,0.35]),
    'Age': np.where(rng.random(n)>0.2, rng.normal(29,14,n).clip(1,80), np.nan),
    'SibSp': rng.integers(0,5,n),
    'Parch': rng.integers(0,4,n),
    'Fare': rng.exponential(33,n).round(2),
    'Embarked': rng.choice(['S','C','Q'],n,p=[0.72,0.19,0.09]),
    'Cabin': np.where(rng.random(n)>0.77,
                      [f"{c}{rng.integers(1,150)}" for c in rng.choice(list('ABCDEFG'),n)],
                      None),
    'Ticket': [f'PC {rng.integers(10000,99999)}' for _ in range(n)],
})

# ── 2. Heart Disease ─────────────────────────────────────────
n=303
datasets['heart_disease'] = pd.DataFrame({
    'age': rng.integers(29,78,n),
    'sex': rng.integers(0,2,n),
    'cp': rng.integers(0,4,n),
    'trestbps': rng.integers(94,200,n),
    'chol': rng.integers(126,564,n),
    'fbs': rng.integers(0,2,n),
    'restecg': rng.integers(0,3,n),
    'thalach': rng.integers(71,202,n),
    'exang': rng.integers(0,2,n),
    'oldpeak': rng.uniform(0,6.2,n).round(1),
    'slope': rng.integers(0,3,n),
    'ca': rng.integers(0,4,n),
    'thal': rng.integers(0,4,n),
    'target': rng.integers(0,2,n),
})

# ── 3. House Prices ───────────────────────────────────────────
n=1460
hoods = ['CollgCr','Veenker','Crawfor','NoRidge','Mitchel',
         'Somerst','NWAmes','OldTown','BrkSide','Sawyer']
datasets['house_prices'] = pd.DataFrame({
    'Id': range(1,n+1),
    'MSSubClass': rng.choice([20,30,40,50,60,70,80,90,120,160],n),
    'LotArea': rng.integers(1300,215245,n),
    'OverallQual': rng.integers(1,11,n),
    'OverallCond': rng.integers(1,10,n),
    'YearBuilt': rng.integers(1872,2010,n),
    'YearRemodAdd': rng.integers(1950,2010,n),
    'TotalBsmtSF': rng.integers(0,6110,n),
    'GrLivArea': rng.integers(334,5642,n),
    'FullBath': rng.integers(0,4,n),
    'BedroomAbvGr': rng.integers(0,8,n),
    'TotRmsAbvGrd': rng.integers(2,14,n),
    'GarageCars': rng.integers(0,4,n),
    'GarageArea': rng.integers(0,1418,n),
    'Neighborhood': rng.choice(hoods,n),
    'BldgType': rng.choice(['1Fam','2fmCon','Duplex','TwnhsE','Twnhs'],n,p=[0.83,0.03,0.05,0.07,0.02]),
    'SalePrice': rng.integers(34900,755000,n),
})

# ── 4. Bike Sharing ───────────────────────────────────────────
n=500
datasets['bike_sharing'] = pd.DataFrame({
    'dteday': pd.date_range('2011-01-01',periods=n,freq='D').strftime('%Y-%m-%d'),
    'season': rng.integers(1,5,n),
    'yr': rng.integers(0,2,n),
    'mnth': rng.integers(1,13,n),
    'holiday': rng.choice([0,1],n,p=[0.97,0.03]),
    'weekday': rng.integers(0,7,n),
    'workingday': rng.integers(0,2,n),
    'weathersit': rng.choice([1,2,3,4],n,p=[0.46,0.38,0.15,0.01]),
    'temp': rng.uniform(0.02,1,n).round(4),
    'atemp': rng.uniform(0,1,n).round(4),
    'hum': rng.uniform(0,1,n).round(4),
    'windspeed': rng.uniform(0,0.85,n).round(4),
    'casual': rng.integers(2,3410,n),
    'registered': rng.integers(20,6946,n),
    'cnt': rng.integers(22,8714,n),
})

# ── 5. Customer Churn ─────────────────────────────────────────
n=500
datasets['customer_churn'] = pd.DataFrame({
    'customerID': [f'CUST_{i:05d}' for i in range(n)],
    'gender': rng.choice(['Male','Female'],n),
    'SeniorCitizen': rng.integers(0,2,n),
    'Partner': rng.choice(['Yes','No'],n),
    'Dependents': rng.choice(['Yes','No'],n),
    'tenure': rng.integers(0,73,n),
    'PhoneService': rng.choice(['Yes','No'],n,p=[0.90,0.10]),
    'InternetService': rng.choice(['DSL','Fiber optic','No'],n,p=[0.34,0.44,0.22]),
    'Contract': rng.choice(['Month-to-month','One year','Two year'],n,p=[0.55,0.21,0.24]),
    'PaperlessBilling': rng.choice(['Yes','No'],n,p=[0.59,0.41]),
    'PaymentMethod': rng.choice(['Electronic check','Mailed check','Bank transfer','Credit card'],n),
    'MonthlyCharges': rng.uniform(18.25,118.75,n).round(2),
    'TotalCharges': np.where(rng.random(n)>0.02, rng.uniform(18.8,8684.8,n).round(2), np.nan),
    'Churn': rng.choice(['Yes','No'],n,p=[0.265,0.735]),
})

# ── 6. Wine Quality ───────────────────────────────────────────
n=500
datasets['wine_quality'] = pd.DataFrame({
    'fixed_acidity': rng.uniform(3.8,15.9,n).round(1),
    'volatile_acidity': rng.uniform(0.08,1.58,n).round(2),
    'citric_acid': rng.uniform(0,1.66,n).round(2),
    'residual_sugar': rng.uniform(0.6,65.8,n).round(1),
    'chlorides': rng.uniform(0.009,0.611,n).round(3),
    'free_sulfur_dioxide': rng.uniform(1,289,n).round(0),
    'total_sulfur_dioxide': rng.uniform(6,440,n).round(0),
    'density': rng.uniform(0.9871,1.039,n).round(4),
    'pH': rng.uniform(2.72,4.01,n).round(2),
    'sulphates': rng.uniform(0.22,2.0,n).round(2),
    'alcohol': rng.uniform(8.0,14.9,n).round(1),
    'type': rng.choice(['red','white'],n,p=[0.25,0.75]),
    'quality': rng.integers(3,10,n),
})

# ── 7. IBM HR Analytics ───────────────────────────────────────
n=500
roles = ['Sales Executive','Research Scientist','Laboratory Technician',
         'Manufacturing Director','Healthcare Representative','Manager',
         'Sales Representative','Research Director','Human Resources']
datasets['ibm_hr_analytics'] = pd.DataFrame({
    'Age': rng.integers(18,60,n),
    'Attrition': rng.choice(['Yes','No'],n,p=[0.16,0.84]),
    'BusinessTravel': rng.choice(['Non-Travel','Travel_Rarely','Travel_Frequently'],n,p=[0.10,0.71,0.19]),
    'DailyRate': rng.integers(102,1499,n),
    'Department': rng.choice(['Sales','Research & Development','Human Resources'],n,p=[0.31,0.65,0.04]),
    'DistanceFromHome': rng.integers(1,29,n),
    'Education': rng.integers(1,6,n),
    'Gender': rng.choice(['Male','Female'],n,p=[0.60,0.40]),
    'HourlyRate': rng.integers(30,100,n),
    'JobLevel': rng.integers(1,6,n),
    'JobRole': rng.choice(roles,n),
    'JobSatisfaction': rng.integers(1,5,n),
    'MaritalStatus': rng.choice(['Single','Married','Divorced'],n,p=[0.32,0.46,0.22]),
    'MonthlyIncome': rng.integers(1009,19999,n),
    'NumCompaniesWorked': rng.integers(0,10,n),
    'OverTime': rng.choice(['Yes','No'],n,p=[0.28,0.72]),
    'PercentSalaryHike': rng.integers(11,25,n),
    'PerformanceRating': rng.integers(3,5,n),
    'TotalWorkingYears': rng.integers(0,40,n),
    'TrainingTimesLastYear': rng.integers(0,7,n),
    'WorkLifeBalance': rng.integers(1,5,n),
    'YearsAtCompany': rng.integers(0,40,n),
    'YearsInCurrentRole': rng.integers(0,18,n),
})

# ── 8. COVID-19 Country Stats ─────────────────────────────────
countries = [
    'USA','India','Brazil','Russia','France','UK','Germany','Italy','Spain',
    'Argentina','Colombia','Poland','Turkey','South Africa','Mexico',
    'Indonesia','Netherlands','Iran','Philippines','Ukraine','Romania',
    'Belgium','Chile','Hungary','Canada','Peru','Sweden','Portugal',
    'Switzerland','Austria','Israel','Japan','Pakistan','Malaysia',
    'Bangladesh','Thailand','Morocco','Saudi Arabia','Australia','Serbia',
    'Vietnam','Kazakhstan','Denmark','Norway','Finland','Croatia',
    'Slovakia','Lithuania','Bolivia','Lebanon','Jordan','Egypt','Nigeria',
    'Kenya','Ethiopia','Ghana','Tanzania','Uganda','Cameroon','Senegal',
    'Tunisia','Zimbabwe','Mozambique','Rwanda','Angola','Zambia','Nepal',
    'Sri Lanka','Myanmar','Cambodia','Cuba','Panama','Guatemala','Ecuador',
    'Venezuela','Paraguay','Uruguay','Haiti','Algeria','Sudan','Libya',
    'Iraq','Syria','Yemen','Afghanistan','Uzbekistan']
n2 = len(countries)
continents = {
    'USA':'North America','Canada':'North America','Mexico':'North America',
    'Brazil':'South America','Argentina':'South America','Colombia':'South America',
    'Peru':'South America','Chile':'South America','Venezuela':'South America',
    'Paraguay':'South America','Uruguay':'South America','Bolivia':'South America',
    'Ecuador':'South America','Panama':'North America','Guatemala':'North America',
    'Haiti':'North America','Cuba':'North America',
    'UK':'Europe','Germany':'Europe','France':'Europe','Italy':'Europe','Spain':'Europe',
    'Russia':'Europe','Poland':'Europe','Ukraine':'Europe','Romania':'Europe',
    'Belgium':'Europe','Hungary':'Europe','Sweden':'Europe','Portugal':'Europe',
    'Switzerland':'Europe','Austria':'Europe','Netherlands':'Europe','Denmark':'Europe',
    'Norway':'Europe','Finland':'Europe','Croatia':'Europe','Slovakia':'Europe',
    'Lithuania':'Europe','Serbia':'Europe','Czech Republic':'Europe',
    'India':'Asia','China':'Asia','Japan':'Asia','Pakistan':'Asia','Indonesia':'Asia',
    'Bangladesh':'Asia','Philippines':'Asia','Vietnam':'Asia','Thailand':'Asia',
    'Malaysia':'Asia','Iran':'Asia','Turkey':'Asia','Israel':'Asia','Kazakhstan':'Asia',
    'Saudi Arabia':'Asia','Iraq':'Asia','Syria':'Asia','Yemen':'Asia',
    'Afghanistan':'Asia','Nepal':'Asia','Sri Lanka':'Asia','Myanmar':'Asia',
    'Cambodia':'Asia','Laos':'Asia','Mongolia':'Asia','Uzbekistan':'Asia',
    'South Africa':'Africa','Nigeria':'Africa','Kenya':'Africa','Ethiopia':'Africa',
    'Ghana':'Africa','Tanzania':'Africa','Uganda':'Africa','Cameroon':'Africa',
    'Senegal':'Africa','Tunisia':'Africa','Zimbabwe':'Africa','Mozambique':'Africa',
    'Rwanda':'Africa','Angola':'Africa','Zambia':'Africa','Algeria':'Africa',
    'Sudan':'Africa','Libya':'Africa','Egypt':'Africa','Morocco':'Africa','DRC':'Africa',
    'Australia':'Oceania','Jordan':'Asia','Lebanon':'Asia',
}
datasets['covid19_country'] = pd.DataFrame({
    'country': countries,
    'continent': [continents.get(c,'Asia') for c in countries],
    'total_cases': rng.integers(500,50000000,n2),
    'new_cases': rng.integers(0,100000,n2),
    'total_deaths': rng.integers(10,900000,n2),
    'new_deaths': rng.integers(0,5000,n2),
    'total_recovered': rng.integers(400,45000000,n2),
    'active_cases': rng.integers(100,5000000,n2),
    'total_tests': rng.integers(10000,500000000,n2),
    'population': rng.integers(1000000,1400000000,n2),
    'gdp_per_capita': rng.integers(500,65000,n2),
    'median_age': rng.uniform(15,48,n2).round(1),
    'hospital_beds_per_thousand': rng.uniform(0.1,14.0,n2).round(2),
    'life_expectancy': rng.uniform(52,84,n2).round(1),
    'vaccination_rate_pct': rng.uniform(1,98,n2).round(1),
    'stringency_index': rng.uniform(0,100,n2).round(2),
})

# ── 9. NYC Taxi Trips ─────────────────────────────────────────
n=500
base_ts = pd.Timestamp('2023-01-01')
pickup_dt = [base_ts + pd.Timedelta(seconds=int(s))
             for s in rng.integers(0, 86400*30, n)]
dropoff_dt = [t + pd.Timedelta(minutes=int(m))
              for t, m in zip(pickup_dt, rng.integers(2,90,n))]
datasets['nyc_taxi'] = pd.DataFrame({
    'VendorID': rng.integers(1,3,n),
    'pickup_datetime': [t.strftime('%Y-%m-%d %H:%M:%S') for t in pickup_dt],
    'dropoff_datetime': [t.strftime('%Y-%m-%d %H:%M:%S') for t in dropoff_dt],
    'passenger_count': np.where(rng.random(n)>0.05, rng.integers(1,7,n).astype(float), np.nan),
    'trip_distance': rng.uniform(0.1,35.0,n).round(2),
    'PULocationID': rng.integers(1,266,n),
    'DOLocationID': rng.integers(1,266,n),
    'payment_type': rng.integers(1,6,n),
    'fare_amount': rng.uniform(2.5,200,n).round(2),
    'tip_amount': rng.uniform(0,50,n).round(2),
    'tolls_amount': rng.uniform(0,20,n).round(2),
    'total_amount': rng.uniform(3,250,n).round(2),
    'trip_duration_minutes': rng.integers(2,90,n),
    'hour_of_day': rng.integers(0,24,n),
    'day_of_week': rng.integers(0,7,n),
    'is_airport': rng.integers(0,2,n),
    'congestion_surcharge': rng.choice([0.0,2.5],n,p=[0.10,0.90]),
    'rate_code': rng.choice([1,2,3,4,5,6],n,p=[0.90,0.04,0.01,0.002,0.04,0.008]),
    'store_and_fwd': rng.choice(['Y','N'],n,p=[0.01,0.99]),
})

# ── 10. E-commerce Orders ─────────────────────────────────────
n=500
products_list = ['Laptop','Phone','Tablet','Headphones','Camera','Watch',
                 'Keyboard','Monitor','Speaker','Mouse','Charger','Cable',
                 'Router','Printer','Projector','Smart TV','Gaming Console']
cities = ['Mumbai','Delhi','Bangalore','Chennai','Kolkata',
          'Hyderabad','Pune','Ahmedabad','Jaipur','Lucknow',
          'Surat','Bhopal','Nagpur','Indore','Kanpur']
datasets['ecommerce_orders'] = pd.DataFrame({
    'order_id': [f'ORD{i:06d}' for i in range(n)],
    'customer_id': [f'CUST{rng.integers(1,5001):05d}' for _ in range(n)],
    'order_date': pd.date_range('2023-01-01',periods=n,freq='16h').strftime('%Y-%m-%d'),
    'product_category': rng.choice(['Electronics','Clothing','Books','Home & Garden','Sports','Beauty','Toys','Food'],n),
    'product_name': rng.choice(products_list,n),
    'quantity': rng.integers(1,11,n),
    'unit_price': rng.uniform(5,2000,n).round(2),
    'discount_pct': rng.choice([0,5,10,15,20,25,30],n,p=[0.40,0.15,0.15,0.10,0.10,0.05,0.05]),
    'shipping_cost': rng.uniform(0,50,n).round(2),
    'total_amount': rng.uniform(5,5000,n).round(2),
    'payment_method': rng.choice(['Credit Card','Debit Card','UPI','Net Banking','COD','Wallet'],n),
    'shipping_city': rng.choice(cities,n),
    'order_status': rng.choice(['Delivered','Shipped','Processing','Cancelled','Returned'],n,p=[0.70,0.12,0.08,0.06,0.04]),
    'customer_rating': np.where(rng.random(n)>0.1, rng.integers(1,6,n).astype(float), np.nan),
    'return_flag': rng.integers(0,2,n),
    'days_to_deliver': np.where(rng.random(n)>0.08, rng.integers(1,15,n).astype(float), np.nan),
    'customer_age_group': rng.choice(['18-25','26-35','36-45','46-55','55+'],n),
    'is_repeat_customer': rng.integers(0,2,n),
    'platform': rng.choice(['Mobile App','Website','In-Store Kiosk'],n,p=[0.55,0.38,0.07]),
    'seller_id': [f'SELL{rng.integers(1,201):04d}' for _ in range(n)],
    'region': rng.choice(['North','South','East','West','Central'],n),
})

# ── 11. Credit Card Fraud ─────────────────────────────────────
n=500
fraud_mask = rng.random(n) < 0.0017
cc_data = {f'V{i}': rng.normal(0,1,n).round(4) for i in range(1,29)}
cc_data['Time']   = rng.uniform(0,172792,n).round(0)
cc_data['Amount'] = np.where(fraud_mask,
                              rng.uniform(0.01,25691,n).round(2),
                              rng.exponential(88,n).round(2))
cc_data['Class']  = fraud_mask.astype(int)
datasets['credit_card_fraud'] = pd.DataFrame(cc_data)

# ── 12. Rossmann Store Sales ──────────────────────────────────
n=500
datasets['rossmann_sales'] = pd.DataFrame({
    'Store': rng.integers(1,1116,n),
    'DayOfWeek': rng.integers(1,8,n),
    'Date': pd.date_range('2013-01-01',periods=n,freq='D').strftime('%Y-%m-%d'),
    'Sales': np.where(rng.random(n)>0.15, rng.integers(0,41551,n), 0),
    'Customers': np.where(rng.random(n)>0.15, rng.integers(0,7388,n), 0),
    'Open': rng.choice([0,1],n,p=[0.14,0.86]),
    'Promo': rng.integers(0,2,n),
    'StateHoliday': rng.choice(['0','a','b','c'],n,p=[0.94,0.03,0.02,0.01]),
    'SchoolHoliday': rng.integers(0,2,n),
    'StoreType': rng.choice(['a','b','c','d'],n,p=[0.53,0.17,0.17,0.13]),
    'Assortment': rng.choice(['a','b','c'],n,p=[0.53,0.01,0.46]),
    'CompetitionDistance': np.where(rng.random(n)>0.003,
                                    rng.integers(20,75860,n).astype(float), np.nan),
    'CompetitionOpenSinceMonth': np.where(rng.random(n)>0.32,
                                          rng.integers(1,13,n).astype(float), np.nan),
    'Promo2': rng.integers(0,2,n),
    'Promo2SinceWeek': np.where(rng.random(n)>0.49,
                                 rng.integers(1,53,n).astype(float), np.nan),
    'Promo2SinceYear': np.where(rng.random(n)>0.49,
                                 rng.integers(2009,2016,n).astype(float), np.nan),
    'Sales_per_Customer': np.where(rng.random(n)>0.15,
                                    rng.uniform(0,100,n).round(2), np.nan),
})

# ── Save all ──────────────────────────────────────────────────
for name, df in datasets.items():
    path = f'data/sample/{name}.csv'
    df.to_csv(path, index=False)
    print(f'  ✅  {name:<30}  {df.shape[0]:>5} rows  ×  {df.shape[1]:>2} cols')

print(f'\n✅  All {len(datasets)} datasets saved to data/sample/')
