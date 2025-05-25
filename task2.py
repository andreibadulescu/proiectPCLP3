import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import preprocessing
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

# import dataset
trainingDataset = pd.read_csv('Autostrazi_Antrenare.csv', encoding='utf-8-sig', sep=',', on_bad_lines='skip')
testingDataset = pd.read_csv('Autostrazi_Testare.csv', encoding='utf-8-sig', sep=',', on_bad_lines='skip')

# task 2.1 - prelucrarea datelor
trainingDataset = trainingDataset.replace(-1.0, np.nan)
trainingDataset = trainingDataset.rename(columns = {' Station ': 'Station'})

testingDataset = testingDataset.replace(-1.0, np.nan)
testingDataset = testingDataset.rename(columns = {' Station ': 'Station'})

# variabilele numerice nu pot fi standardizate, intrucat acestea releva
# gradul de aglomeratie din fiecare punct de-a lungul unui tronson de
# drum de mare viteza, iar maximul este variabil in functie de categoria sa

# inlocuim valorile lipsa cu valoarea medie
# eliminam astfel cazurile in care valorile pot introduce entropie
# (valorile de trafic -1 sunt imposibile) in modelul de antrenament

trainingDataset_kfz = trainingDataset[trainingDataset['Fahrzeugklasse'] == 'Kfz']
kfzMean = trainingDataset_kfz.mean(numeric_only = True, skipna = True).drop(axis = 0, index=['Station', 'Zählstellen-nummer'])
trainingDataset_kfzOben = trainingDataset[trainingDataset['Fahrzeugklasse'] == 'Kfz > 3,5t hzG']
kfzObenMean = trainingDataset_kfzOben.mean(numeric_only = True, skipna = True).drop(axis = 0, index=['Station', 'Zählstellen-nummer'])
trainingDataset_kfzUnter = trainingDataset[trainingDataset['Fahrzeugklasse'] == 'Kfz <= 3,5t hzG']
kfzUnterMean = trainingDataset_kfzUnter.mean(numeric_only = True, skipna = True).drop(axis = 0, index=['Station', 'Zählstellen-nummer'])

trainingDataset = trainingDataset.reset_index(drop=True)

for index, row in trainingDataset.iterrows():
	if index == 0:
		continue
	kfzCategory = row[6]

	if row['Datengüte'] == 'Ausfall':
		match kfzCategory:
			case 'Kfz':
				for j in range(7):
					trainingDataset.iloc[index, 7 + j] = kfzMean.iloc[j]
			case 'Kfz > 3,5t hzG':
				for j in range(7):
					trainingDataset.iloc[index, 7 + j] = kfzObenMean.iloc[j]
			case 'Kfz <= 3,5t hzG':
				for j in range(7):
					trainingDataset.iloc[index, 7 + j] = kfzUnterMean.iloc[j]

testingDataset_kfz = testingDataset[testingDataset['Fahrzeugklasse'] == 'Kfz']
kfzMean = testingDataset_kfz.mean(numeric_only = True, skipna = True).drop(axis = 0, index=['Station', 'Zählstellen-nummer'])
testingDataset_kfzOben = testingDataset[testingDataset['Fahrzeugklasse'] == 'Kfz > 3,5t hzG']
kfzObenMean = testingDataset_kfzOben.mean(numeric_only = True, skipna = True).drop(axis = 0, index=['Station', 'Zählstellen-nummer'])
testingDataset_kfzUnter = testingDataset[testingDataset['Fahrzeugklasse'] == 'Kfz <= 3,5t hzG']
kfzUnterMean = testingDataset_kfzUnter.mean(numeric_only = True, skipna = True).drop(axis = 0, index=['Station', 'Zählstellen-nummer'])

testingDataset = testingDataset.reset_index(drop=True)

for index, row in testingDataset.iterrows():
	if index == 0:
		continue
	kfzCategory = row[6]

	if row['Datengüte'] == 'Ausfall':
		match kfzCategory:
			case 'Kfz':
				for j in range(7):
					testingDataset.iloc[index, 7 + j] = kfzMean.iloc[j]
			case 'Kfz > 3,5t hzG':
				for j in range(7):
					testingDataset.iloc[index, 7 + j] = kfzObenMean.iloc[j]
			case 'Kfz <= 3,5t hzG':
				for j in range(7):
					testingDataset.iloc[index, 7 + j] = kfzUnterMean.iloc[j]

# encodam variabilele categorice:
# modelul poate fi astfel antrenat pe trei clase diferite:
# vehicule, autoturisme si autocamioane / autocare / vehicule grele;
# de asemenea, dupa encodarea Datengüte modelul poate oferi rezultate
# corecte in functie de precizia numaratorului
label_encoder = preprocessing.LabelEncoder()
trainingDataset['Fahrzeugklasse'] = label_encoder.fit_transform(trainingDataset['Fahrzeugklasse'])
trainingDataset['Datengüte'] = label_encoder.fit_transform(trainingDataset['Datengüte'])

label_encoder = preprocessing.LabelEncoder()
testingDataset['Fahrzeugklasse'] = label_encoder.fit_transform(testingDataset['Fahrzeugklasse'])
testingDataset['Datengüte'] = label_encoder.fit_transform(testingDataset['Datengüte'])

## task 2.2 - Analiza exploratorie a datelor
print()

# nu exista valori lipsa, acestea au fost tratate la taskul 2.1
# verificare existenta date invalide
print("Does the training set contain invalid values?")
print(trainingDataset.isnull().values.any())

print("Does the testing set contain invalid values?")
print(testingDataset.isnull().values.any())

print()

# statistici generale despre subseturile de antrenament, respectiv testare
print("General stats about the training set:")
print(trainingDataset.describe())

print()

print("General stats about the testing set:")
print(testingDataset.describe())

print()

# analiza distributiei variabilelor
print("Opening histograms...")
numericColumns = trainingDataset.select_dtypes(include=['Float64', 'Int64']).columns
numericColumns = numericColumns.drop('Station').drop('Zählstellen-nummer').drop('Fahrzeugklasse').drop('Datengüte')

for col in numericColumns:
	plt.figure(figsize=(10, 5))
	plt.hist(trainingDataset[col], bins=30, color='blue', edgecolor='black')
	plt.title(f'Histogramă pentru {col} (Training Dataset)')
	plt.xlabel(col)
	plt.ylabel('Valori numerice')
	plt.grid(axis='y', alpha=0.75)

numericColumns = testingDataset.select_dtypes(include=['Float64', 'Int64']).columns
numericColumns = numericColumns.drop('Station').drop('Zählstellen-nummer').drop('Fahrzeugklasse').drop('Datengüte')

for col in numericColumns:
	plt.figure(figsize=(10, 5))
	plt.hist(testingDataset[col], bins=30, color='blue', edgecolor='black')
	plt.title(f'Histogramă pentru {col} (Testing Dataset)')
	plt.xlabel(col)
	plt.ylabel('Valori numerice')
	plt.grid(axis='y', alpha=0.75)

plt.show()

print("Opening barplots...")
categoryColumns = trainingDataset[['Fahrzeugklasse', 'Datengüte']]
for col in categoryColumns:
	plt.figure(figsize=(10, 5))
	counts = trainingDataset[col].value_counts()
	plt.bar(counts.index.astype(str), counts.values, color='lightgreen', edgecolor='black')
	plt.title(f'Barplot pentru {col} (Training Dataset)')
	plt.xlabel(col)
	plt.ylabel('Număr apariții')
	plt.xticks(rotation=45, ha='right')
	plt.tight_layout()

categoryColumns = testingDataset[['Fahrzeugklasse', 'Datengüte']]
for col in categoryColumns:
	plt.figure(figsize=(10, 5))
	counts = testingDataset[col].value_counts()
	plt.bar(counts.index.astype(str), counts.values, color='lightgreen', edgecolor='black')
	plt.title(f'Barplot pentru {col} (Testing Dataset)')
	plt.xlabel(col)
	plt.ylabel('Număr apariții')
	plt.xticks(rotation=45, ha='right')
	plt.tight_layout()

plt.show()

# detectarea outlierilor
print('Opening boxplots for anomaly detection...')
outlierDetectionColumns = trainingDataset.select_dtypes(include=['Float64', 'Int64']).columns
outlierDetectionColumns = outlierDetectionColumns.drop('Station').drop('Zählstellen-nummer').drop('Fahrzeugklasse').drop('Datengüte')
for col in outlierDetectionColumns:
	plt.figure(figsize=(10, 5))
	plt.boxplot(trainingDataset[col], vert=False)
	plt.title(f'Boxplot pentru {col} (Training Dataset)')
	plt.xlabel(col)
	plt.grid(True)

outlierDetectionColumns = testingDataset.select_dtypes(include=['Float64', 'Int64']).columns
outlierDetectionColumns = outlierDetectionColumns.drop('Station').drop('Zählstellen-nummer').drop('Fahrzeugklasse').drop('Datengüte')
for col in outlierDetectionColumns:
	plt.figure(figsize=(10, 5))
	plt.boxplot(testingDataset[col], vert=False)
	plt.title(f'Boxplot pentru {col} (Testing Dataset)')
	plt.xlabel(col)
	plt.grid(True)

plt.show()

# analiza corelatiilor
print('Opening heatmaps...')
heatmap = trainingDataset[numericColumns].corr()
plt.figure(figsize=(10, 10))
plt.imshow(heatmap, cmap='coolwarm')
plt.colorbar(label='Coeficient de corelație')
plt.yticks(ticks=range(len(heatmap.columns)), labels=heatmap.columns)
plt.title("Heatmap (Training Dataset)")

heatmap = testingDataset[numericColumns].corr()
plt.figure(figsize=(10, 10))
plt.imshow(heatmap, cmap='coolwarm')
plt.colorbar(label='Coeficient de corelație')
plt.yticks(ticks=range(len(heatmap.columns)), labels=heatmap.columns)
plt.title("Heatmap (Testing Dataset)")
plt.show()

# analiza relatiilor cu variabila tinta
print('Opening scatter plots...')
plt.figure(figsize=(10, 5))
plt.scatter(trainingDataset['DTVMS Mo-Fr Kfz/24h'], trainingDataset['DTVMS Mo-So Kfz/24h'], alpha=0.6, color='red')
plt.title('Scatter Plot - Mean traffic values during workdays and general mean traffic values (Training)')
plt.xlabel('DTVMS Mo-Fr Kfz/24h')
plt.ylabel('DTVMS Mo-So Kfz/24h')
plt.grid(True)

plt.figure(figsize=(10, 5))
plt.scatter(trainingDataset['DTVMS Mo Kfz/24h'], trainingDataset['DTVMS Mo-So Kfz/24h'], alpha=0.6, color='red')
plt.title('Scatter Plot - Mean traffic values on Mondays and general mean traffic values (Training)')
plt.xlabel('DTVMS Mo Kfz/24h')
plt.ylabel('DTVMS Mo-So Kfz/24h')
plt.grid(True)

plt.figure(figsize=(10, 5))
plt.scatter(testingDataset['DTVMS Mo-Fr Kfz/24h'], testingDataset['DTVMS Mo-So Kfz/24h'], alpha=0.6, color='red')
plt.title('Scatter Plot - Mean traffic values during workdays and general mean traffic values (Testing)')
plt.xlabel('DTVMS Mo-Fr Kfz/24h')
plt.ylabel('DTVMS Mo-So Kfz/24h')
plt.grid(True)

plt.figure(figsize=(10, 5))
plt.scatter(testingDataset['DTVMS Mo Kfz/24h'], testingDataset['DTVMS Mo-So Kfz/24h'], alpha=0.6, color='red')
plt.title('Scatter Plot - Mean traffic values on Mondays and general mean traffic values (Testing)')
plt.xlabel('DTVMS Mo Kfz/24h')
plt.ylabel('DTVMS Mo-So Kfz/24h')
plt.grid(True)

plt.show()
print()

# comentariile referitoare la rezultatele analizei exploratorii a datelor
# pot fi găsite în README

# task 2.3 + 2.4
# 3) antrenarea si compararea a 3 algoritmi de ML
# 4) evaluarea performantei

# pregatim datele pentru ML
targetColumn = 'DTVMS Mo-So Kfz/24h'
predictorsColumns = ['DTVMS Mo-Fr Kfz/24h', 'DTVMS Mo Kfz/24h']
predictorsTraining = trainingDataset[predictorsColumns]
targetsTraining = trainingDataset[targetColumn]
predictorsTesting = testingDataset[predictorsColumns]
targetsTesting = testingDataset[targetColumn]

# rulam si evaluam performanta algoritmului linear regression
linearRegModel = LinearRegression()
linearRegModel = linearRegModel.fit(predictorsTraining, targetsTraining)
linearRegPredictions = linearRegModel.predict(predictorsTesting)

R2Linear = r2_score(targetsTesting, linearRegPredictions)
RMSELinear = root_mean_squared_error(targetsTesting, linearRegPredictions)
print("Linear Regression")
print("R2 score:")
print(R2Linear)

print("RMSE score:")
print(RMSELinear)
print()

# rulam si evaluam performanta algoritmului ridge regression
ridgeRegModel = Ridge()
ridgeRegModel = ridgeRegModel.fit(predictorsTraining, targetsTraining)
ridgeRegPredictions = ridgeRegModel.predict(predictorsTesting)

R2Ridge = r2_score(targetsTesting, ridgeRegPredictions)
RMSERidge = root_mean_squared_error(targetsTesting, ridgeRegPredictions)
print("Ridge Regression")
print("R2 score:")
print(R2Ridge)

print("RMSE score:")
print(RMSERidge)
print()

# rulam si evaluam performanta algoritmului decision tree regressor
decTreeRegModel = DecisionTreeRegressor()
decTreeRegModel = decTreeRegModel.fit(predictorsTraining, targetsTraining)
decTreeRegPredictions = decTreeRegModel.predict(predictorsTesting)

R2DecTree = r2_score(targetsTesting, decTreeRegPredictions)
RMSEDecTree = root_mean_squared_error(targetsTesting, decTreeRegPredictions)
print("Decision Tree Regressor")
print("R2 score:")
print(R2DecTree)

print("RMSE score:")
print(RMSEDecTree)
print()

# construim un tabel cu scorurile obtinute
print("Table with ML model scores:")
mlModelsScores = {}
mlModelsScores['LinearReg'] = [R2Linear, RMSELinear]
mlModelsScores['RidgeReg'] = [R2Ridge, RMSERidge]
mlModelsScores['DecTreeReg'] = [R2DecTree, RMSEDecTree]
scoreDF = pd.DataFrame(mlModelsScores)
print(scoreDF)
print()

# prima linie reprezinta scorul R2
# a doua linie reprezinta valorile root mean squared error

# afisarea unor diagrame de tip scatter pentru evaluarea predictiilor
# in raport cu rezultatele asteptate

print("Opening scatter diagrams for prediction analysis...")
plt.figure(figsize=(10, 5))
plt.scatter(targetsTesting, linearRegPredictions, alpha=0.4, color='green')
plt.title('Scatter Plot - Actual expected values vs predicted values (via Linear Regression)')
plt.xlabel('Expected Values')
plt.ylabel('Predicted Values')
plt.grid(True)

plt.figure(figsize=(10, 5))
plt.scatter(targetsTesting, ridgeRegPredictions, alpha=0.4, color='green')
plt.title('Scatter Plot - Actual expected values vs predicted values (via Ridge Regression)')
plt.xlabel('Expected Values')
plt.ylabel('Predicted Values')
plt.grid(True)

plt.figure(figsize=(10, 5))
plt.scatter(targetsTesting, decTreeRegPredictions, alpha=0.4, color='green')
plt.title('Scatter Plot - Actual expected values vs predicted values (via Decision Tree Regressor)')
plt.xlabel('Expected Values')
plt.ylabel('Predicted Values')
plt.grid(True)

plt.show()
print("Program ended.")
