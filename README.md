# Proiect PCLP3 - Analiza informatiilor rutiere de pe anumite Autostrazi

### Descarcarea datelor

Setul de date propus este destinat unei probleme de regresie. Fisierul .csv contine informatii despre traficul pe anumite autostrazi si a fost preluat de pe site-ul https://www.asfinag.at/verkehr-sicherheit/verkehrszaehlung/. Sunt colectate date din mai multe intervale de timp (zile lucratoare, weekend, sarbatori legale) si pe mai multe categorii de autovehicule. Coloana tinta am ales-o ca fiind DTVMS Mo-So Kfz/24h si reprezinta traficul mediu pe toata saptamana pentru toate tipurile de autovehicule.

### Construirea dataset-ului tabelar

Ulterior pentru fiecare coloana setez tipul de date. Implicit au fost trecute ca object. In cazul in care am randuri cu celule libere, 'Int64' si 'Float64' permit convertirea chiar daca exista aceste celule problema. Daca foloseam int sau float obisnuit primeam mesaj de eroare in terminal. Apoi impart tabelul original in 2 fisiere .csv separate. Nu le-am luat intr-o ordine specifica, pur si simplu primele 800 de linii si urmatoarele 400 pentru testul de testare. Tabelul pentru testele de antrenare este mult mai mare pentru o prezicere mai exacta. In tabel avem tipuri de date precum float, int si stringuri.

In continuare parcurg fiecare coloana din tabelul initial si verific daca aceasta are valori lipsa. De asemenea calculez si procentajul pentru cate valori lipsa am. In situatia in care procentajul este diferit de 0, atunci am folosit ca metoda de stergere .dropna, care sterge randul pe care se afla spatiul gol.

### Interpretarea datelor din tabel

Pe setul de date intial aplic metoda describe() care imi ofera informatii suplimentare despre datele din tabel. Aceasta metoda ne ofera informatii despre numarul total de valori non NaN, media aritmetica, deviatia standard si valori minime / maxime. Astfel, pentru setul nostru de date putem trage urmatoarele concluzii:
- Media traficului zilnic este de 78 autovechicule/zi
- Deviatia standard care masoara gradul de raspandire fata de valorile medii este aproximativ 90, asta inseamna ca valorile sunt foarte raspandite, iar valorile sunt variate
- Am o valoare minima de 0.4 si una maxima de 379.2, adica s-au inregistrat minim 0.4 vehicule pe zi in cea mai libera zi si 379.2 autovehicule in cea mai aglomerata zi
- De asemenea, jumatate din statii au masurat 38.6 de autovehicule pe zi, 25% din statii au aproape 13 vehicule pe zi, iar 75% din acestea au 118.2


### Realizarea graficelor si interpretarea acestora

Pentru urmatoarele grafice am folosit libraria matplotlib.pyplot care are numeroase metode pentru grafice si tabele.

Am facut si o histograma pentru datele numerice din tabel. In continuare voi analiza histograma pentru Station care este asimetrica si extrem de aglomerata pe partea stanga. Varful se afla in jurul valorii 0-10 pe axa OX, unde numarul valorilor trece de 600 de exemplare. Dupa varf, valorile scad extrem de repede si poate sugera ca autostrada este mai veche sau conectata la mai putine drumuri principale motiv pentru care este evitata. Astfel, se poate trage concluzia ca majoritatea observatiilor sunt facute de statiile 0-50. Avand in vedere ca la inceput valorile sunt foarte mari si scad extrem de repede ar putea aparea suspiciuni in legatura cu corectitudinea datelor. Ar trebui verificat daca exista duplicate care pot cauza astfel de probleme sau pur si simplu sunt valori aberante.

Am realizat si Barplot pentru variabilele categorice. Voi analiza in detaliu graficul pentru coloana Autostrada care prezinta distributia numarului de aparitii pentru diferite Autostrazi. Se poate observa ca Autostrada A02 are cel mai mare numar de aparitii, putin sub 350. Ca si in histograma anterioara, se observa o scadere a valorilor spre stanga, dar mult mai lina. Spre finalul axei OX traficul este mult mai redus. Sunt destul de multe autostrazi cu valori mici si poate inseamna ca nu sunt drumuri principale si nu merita neaparat sa fie luate in calcul. Am putea pune un prag sa le filtram si astfel graficul va ilustra doar drumurile care sunt frecventate regulat si care reprezinta un punct de interes. Totusi, am putea sa le grupam pe zone si sa vedem pentru fiecare cate drumuri imporante si cate mai mici avem. Astfel, daca fiecare zona are si drumuri importante si drumuri secundare, acestea ar trebui eliminate sau trecute in grafice separate ca metode de backup in situatia in care cele principale se supraaglomereaza.

Boxplotul pentru coloana Station imi arata ca valorile aberante se afla undeva in intervalul 270-370 si sunt reprezentate prin acele cercuri rotunde. Valoarea medie se afla putin sub 50 in grafic si este trasata cu o bara rosie. In acest caz, valorile aberante ar trebui investigate separat.

Am mentionat si anterior ca am ales drept coloana tinta coloana DTVMS Mo-So Kfz/24h. Astfel, am realizat graficul Scatter plots pentru relatia dintre statii si traficul mediu pe toata saptmana pentru toate tipurile de autovehicule. Pentru acest grafic se poate observa ca datele se concentreaza in partea stanga, iar traficul este in medie cuprins intre 0 si 200 pentru toate statiile. Exista si o valoare 'maxima', adica traficul nu depaseste 1000 autovehicule pe zi pentru nici o statie. Statiile cu numere mai mici au trafic mai intens, adica sunt drumuri principale. Distribuita pentru acest grafic este bimodala, adica avem statii unde traficul este foarte redus si statii cu valori mari de autovehicule. In urma analizei, ar trebui sa investigam suplimentar autostrazile cu numar redus de masini intrucat acestea pot avea defectiuni pentru care soferii tind sa le evite. Am putea face si grafice separate pentru zonele de urban si rural ca sa vedem ce solutii avem pentru soselele care nu sunt la fel de frecventate (in cazul in care acestea sunt defecte).

Ultimul grafic compara valorile prezise cu valorile reale. Se poate observa ca valorile sunt extrem de apropiate de bara rosie, ceea ce inseamna ca exista o corelatie foarte buna. Relatia liniara este foarte buna, scorul se apropie foarte tare de 1 si nu exista deviatii fata de linia rosie. Erorile sunt extrem de mici si rare, sunt foarte putine puncte mai departate de linia rosie care sunt distribuite uniform pe esantionul de testare. Pentru ca rezultatul este asa de bun, cred ca ar trebui investigat suplimentar pentru a nu il testa pe aceleasi date cu care l-am si antrenat. Daca acest lucru nu se intampla, atunci modelul este foarte bun si precis.

### Resurse

Pentru realizarea acestui proiect am folosit informatiile din curs si laborator, dar si urmatoarele surse:
	https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html
	https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
	https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html
	https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html
	https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
	https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
	https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
	https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html


### Task 2

https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/
https://pandas.pydata.org/docs/reference/index.html
https://stackoverflow.com/questions/17097236/replace-invalid-values-with-none-in-pandas-dataframe
