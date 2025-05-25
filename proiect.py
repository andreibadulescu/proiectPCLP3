import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

if __name__=="__main__":
	# citesc fisierul
	data = pd.read_csv('AaaaaUTOSTRAZI.csv', encoding='utf-8-sig', sep=',', on_bad_lines='skip')

	# setez pentru fiecare coloana tipul variabilelor
	data['Autostrada'] = data['Autostrada'].astype(str)
	data[' Station '] = data[' Station '].astype('Float64')
	data['Zählstellenname'] = data['Zählstellenname'].astype(str)
	data['Zählstellen-nummer'] = data['Zählstellen-nummer'].astype('Int64')
	data['Abschnitt (von - bis)'] = data['Abschnitt (von - bis)'].astype(str)
	data['Richtung'] = data['Richtung'].astype(str)
	data['Fahrzeugklasse'] = data['Fahrzeugklasse'].astype(str)
	data['DTVMS Mo-So Kfz/24h'] = data['DTVMS Mo-So Kfz/24h'].str.replace(',', '.').astype('Float64')
	data['DTVMS Mo-Fr Kfz/24h'] = data['DTVMS Mo-Fr Kfz/24h'].str.replace(',', '.').astype('Float64')
	data['DTVMS Mo Kfz/24h'] = data['DTVMS Mo Kfz/24h'].str.replace(',', '.').astype('Float64')
	data['DTVDD Di-So Kfz/24h'] = data['DTVDD Di-So Kfz/24h'].str.replace(',', '.').astype('Float64')
	data['DTVFR Fr Kfz/24h'] = data['DTVFR Fr Kfz/24h'].str.replace(',', '.').astype('Float64')
	data['DTVSA Kfz/24h'] = data['DTVSA Kfz/24h'].str.replace(',', '.').astype('Float64')
	data['DTVSF Sonn- und Feiertage Kfz/24h'] = data['DTVSF Sonn- und Feiertage Kfz/24h'].str.replace(',', '.').astype('Float64')
	data['Datengüte'] = data['Datengüte'].astype(str)
	data = data.drop(columns=['Unnamed: 15'])


	# daca in database erau si valori lipsa
	# parcurgeam fiecare coloana
	for col in data.columns:
		print(data[col].isna().sum()) # verificam daca coloana are valori lipsa
		missing_percent = data[col].isna().mean() * 100 # fac si media
		print(missing_percent) # afisez procentajul
		if missing_percent != 0.0: # daca e diferit de 0 inseamna ca am valori lipsa
			data = data.dropna(subset=[col]) # si atunci sterg randul respectiv

	# Împarte datele
	data_800 = data.iloc[:800]
	data_400 = data.iloc[800:1200]

	# le scriu in fisiere separate
	data_800.to_csv('Autostrazi_Antrenare.csv', index=False)
	data_400.to_csv('Autostrazi_Testare.csv', index=False)

	# aceasta este coloana tinta DTVMS Mo-So Kfz/24h (reprezinta traficul mediu pe toata saptamana pentru toate tipurile de autovehicule)

	# afisez detalii despre setul de date
	print(data.describe())

	# daca vreau sa vad ce tipuri de date am pe fiecare coloana
	#for col in data.columns:
	#	print(f"{col}: {data[col].dtype}")

	# pentru a putea face histogramele pe datele numerice, trebuie sa iau intai coloane care au datele respective
	numericColumns = data.select_dtypes(include=['Float64', 'Int64']).columns # le selectez doar pe cele float si int
	for col in numericColumns:
		plt.figure(figsize=(8, 4)) # imi fac o noua fereastra pentru fiecare coloana
		plt.hist(data[col].dropna(), bins=30, color='blue', edgecolor='black') # deseneaza histograma si ignora randurile Nan
		plt.title(f'Histogramă pentru {col}') # ii pun si titlu
		plt.xlabel(col) # pune o eticheta pe fiecare coloana ca sa aiba nume
		plt.ylabel('Numar valori') # pun eticheta si pe axa OY
		plt.grid(axis='y', alpha=0.75) # imi pune si gridul ca sa fie mai usor de citit
		plt.show() # afisez graficul

	# fac un boxplot pentru fiecare coloana numerica pentru a putea determina aberatiile
	for col in numericColumns:
		plt.figure(figsize=(8, 4))
		plt.boxplot(data[col].dropna(), vert=False)
		plt.title(f'Boxplot pentru {col}')
		plt.xlabel(col)
		plt.grid(True)
		plt.show()

	# barplot pentru variabile categorice
	objectColumns = data.select_dtypes(include=['object']).columns
	for col in objectColumns:
		plt.figure(figsize=(8, 4))
		counts = data[col].value_counts()
		plt.bar(counts.index.astype(str), counts.values, color='lightgreen', edgecolor='black')
		plt.title(f'Barplot pentru {col}')
		plt.xlabel(col)
		plt.ylabel('Număr apariții')
		plt.xticks(rotation=45, ha='right')
		plt.tight_layout()
		plt.show()

	# pentru matricea de corelatii folosesc metoda .corr() din libraria pandas
	# aceasta metoda gaseste corelatia pe perechi a tuturor coloanelor din dataframe
	correlation_matrix = data[numericColumns].corr()
	# afisez heatmap-ul
	plt.figure(figsize=(10, 8))
	plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
	plt.colorbar(label='Coeficient de corelație') # am si o bara in degrade care evidentiaza gradul
	# vreau sa adaug etichete cu numele coloanelor
	plt.yticks(ticks=range(len(correlation_matrix.columns)), labels=correlation_matrix.columns)
	# ii dau si un titlu
	plt.title("Matricea de corelații între variabilele numerice")
	plt.show()


	# fac si Scatter plotul intre Station si coloana tinta
	plt.figure(figsize=(8, 4))
	plt.scatter(data[' Station '], data['DTVMS Mo-So Kfz/24h'], alpha=0.5, color='teal')
	plt.title('Scatter plot între Station și DTVMS Mo-So Kfz/24h')
	plt.xlabel('Station')
	plt.ylabel('DTVMS Mo-So Kfz/24h')
	plt.grid(True)
	plt.show()


	# vreau sa citesc seturile de antrenare si testare
	train_data = pd.read_csv('Autostrazi_Antrenare.csv')
	test_data = pd.read_csv('Autostrazi_Testare.csv')

	# setez coloana tinta
	target_column = 'DTVMS Mo-So Kfz/24h'

	# pentru model selectez coloanele numerice
	numeric_columns = train_data.select_dtypes(include=['float64', 'int64']).columns.drop(target_column)

	# separ variabilele si tinta
	X_train = train_data[numeric_columns]
	y_train = train_data[target_column]
	X_test = test_data[numeric_columns]
	y_test = test_data[target_column]


	# normez variabilele
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	# antrenez modul (regresie liniara)
	model = LinearRegression()
	model.fit(X_train, y_train)

	# prezicerea pe setul de testare
	y_pred = model.predict(X_test)

	# utilizez metrici specifice pentru pb de regresie liniara R2
	r2 = r2_score(y_test, y_pred)
	print(f"R2 score: {r2}")

	# afisez si un grafic
	plt.figure(figsize=(8, 5))
	plt.scatter(y_test, y_pred, alpha=0.6, color='purple')
	plt.xlabel("Valori reale")
	plt.ylabel("Predicții")
	plt.title("Predicții vs. Valori reale")
	plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
	plt.grid(True)
	plt.show()


