# Proiect PCLP3 - Analiza informatiilor rutiere de pe anumite Autostrazi

<p>
Proiect realizat de Karina-Marina Antoniu și Andrei-Marcel Bădulescu,
Universitatea Politehnica București. <br>
Data – 25 mai 2025
</p>

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

### Prelucrarea datelor
<p> Subseturile de date au fost preluate din task-ul 1. În urma importării acestora
din fișierele .csv (mai exact <i>'Autostrazi_Antrenare.csv'</i> și
<i>'Autostrazi_Testare.csv'</i>), nu au fost normate / standardizate variabilele
numerice, întrucât în urma unei astfel de operațiuni, ar fi fost pierdută relevanța
modelului (valoarea maximă de trafic pe un tronson nu este cunoscută, și atunci
nu pot fi făcute ajustări / normări / standardizări). Valorile care reprezentau
categorii (clasa de autovehicule analizată și tipul de măsurătoare / numărare)
au fost codificate pentru a economisi memorie sub forma unor numere întregi.
Valorile lipsă au fost înlocuite cu media valorilor de trafic calculate pentru
fiecare clasă în parte pentru a nu afecta antrenarea modelurilor (dacă valorile
-1 erau păstrate, atunci astfel de anomalii pot afecta semnificativ comportamentul
modelului de predicție).
</p>

### Analiza exploratorie a datelor
<p>
În urma prelucrării datelor, este afișată o analiză asupra valorilor lipsă / nule,
însă aceasta va indica că setul de date este complet, întrucât valorile nule au fost
înlocuite de medie în cadrul pasului de prelucrare al datelor. Sunt afișate date
generale referitoare la subsetul de antrenament și cel de testare, iar mai apoi
sunt afișate diferite grafice ce au scopul de a analiza variabilele numerice și cele
ce reprezintă categorii. Detecția anomaliilor este posibilă prin interpretarea
boxplot-urilor, analiza corelațiilor dintre variabilele numerice este ilustrată cu
ajutorul heatmap-urilor și relațiile dintre valorile țintă și predictori reies
din diagramele de tip scatter.
</p>

#### Analiză detaliată a diagramelor

<p>
Histograma pentru DTSV Sonn - und Feiertage este asimetrica, in jurul
valorii 0 a axei OX s-au inregistrat putin sub 250 de autovehicule.
Valorile scad brusc cu cat ne departam de origine pe axa OX, indicand
ca majoritatea zilelor de weekend si sarbatori legale traficul auto
este redus. Avand in vedere ca exista o singura coloana cu valori
foarte mari, iar celelalte sunt aproape nule poate aparea problema
unei erori de masurare. Pentru o analiza mai buna putem sa verificam
corectitudinea datelor, pentru a nu repeta valori sau sa nu existe
valori aberante. De asemenea, se pot trage concluzii cu privire la
calitatea drumului sau importanta acestuia. Spre exemplu, acolo unde
avem valori aproape nule, pot aparea motive tehnice care trebuie
investigate (exemplu drumuri stricate sau blocate). Aceste sosele
reprezinta alternative in situatia supraaglomerarii.
</p>

<p>
In comparatie, histograma care prezinta traficul zilnic mediu
scade treptat. Exista si aici un maxim, care este atins pentru
traficul cu media cuprinsa intre 0 si 20, unde numarul de observatii
este trecut de 80. Graficul sugereaza ca traficul se concentreaza
in intervale medii spre ridicate. Pe masura ce traficul creste,
numarul de observatii scade. Astfel, prima coloana poate fi o zona
aglomerata, pe care multi soferi o aleg in detrimentul celorlalte
datoria conexiunilor cu alte drumuri princiapale. Avand in vedere
ca scade liniar, nu se pune se pune problema unui set de date
gresit sau predominat de valori aberante. De asemenea, diferenta
poate proveni din cauze perioadelor in care s-au efectuat
observatiile (spre exemplu perioada de vacanta vs perioada obisnuita).
Un mod de preprocesare ar putea fi tratarea variabilelor lipsa,
adica inlocuirea acestora cu valori medii sau pur si simplu ignorarea acestora.
</p>

<p>
Pentru histogramele DTVSA (traficul mediu pentru transportul greu),
DTVMS (transportul rutier motorizat greu) si DTVDD (densitatea
traficului de tranzit), scaderea este brusca. Prima coloana are
valori foarte mari urmand ca celelalte sa se apropie de 0. De
aici putem trage concluzia ca exista erori de numarare sau date
aberante care strica corectitudinea graficului. In cazul in care
acestea se dovedesc a fi corecte putem vorbi de un trafic accentuat
in anumite perioade sau pe anumite sosele pe care soferii de autovechicule
ce transporta marfuri grele le aleg. Din grafice, se observa ca in general
toate vehiculele sunt usoare (precum masinile personale), iar
vehiculele putin mai grele precum autobuzele si autocarele sunt extrem
de rare. Una din suspiciuni poate fi reprezentata de zona in care a fost
efectuata colectarea de date. Daca este o zona rezidentiala traficul de
marfa nu se va face pe rutele respective.
</p>

<p>
Barploturile sunt relevate pentru a observa calitatea datelor. Spre
exemplu, Barplotul pentru Datengute (atat pentru Testing cat si pentru
Training) sugereaza ca datele sunt bune. Graficul isi atinge maximul
in coloana 20 acolo unde se fac si cele mai multe observatii. Astfel,
dintre toate datele colectate cele mai multe sunt corecte. Acest lucru
poate ridica suspiciuni: daca datele sunt invechite sau colectate din
zonele ideale. O valoare slaba (din grafice rezulta ca sunt putine
cazuri in care se indeplineste) poate duce la subestimari sau
supraestimari ale traficului din anumite zone.
</p>

<p>
Barploturile pentru Fahrzeugklasse sunt echilibrate, sugestie a faptului
ca datele colectate provin de la toate tipurile de masini. Avem
autovehicule cu dimensiuni variate, norme de poluare diferite si greutati
si aspecte tehnice diverse.
</p>

<p>
Boxploturile evidentiaza ca valorile aberante sunt in partea dreapta a
graficelor si se intind pe intervale restranse. Acestea sunt reprezentate
cu cerculete, iar valoarea medie este trasata cu rosu in partea stanga.
Avand in vedere distanta (uneori prea mare) dintre valoarea medie si
valorile aberante ar trebui sa tratam separat cazurile respective.
</p>

<p>
Pentru Scatter Plotul din Figura 4, majoritatea datelor sunt concentrate
in coltul din stanga jos si indica o frecventa scazuta a traficului greu
atat la orele de varf, cat si in intreaga zi, dar in partea dreapta sus
avem si cateva exceptii. Datele sugereaza ca traficul greu este sporadic,
iar vehiculele grele circula rar in acea zona. Punctele extreme pot
corespunde unor evenimente speciale, care ar trebui investigate separat
pentru a nu exista erori.
</p>

<p>
Diagrama de tip Scatter pentru transportul rutier motorizat greu indica
valori mari pentru statiile numerotate intre 0 si 1000, dar exista
si cateva pentru statiile apropiate de 6000. Asta poate sugera
faptul ca drumurile pe care sunt amplasate statiile 1001 - 5999 sunt
in zone urbane care nu sunt frecventate de masinile de marfa.
</p>

### Antrenarea și compararea a trei algoritmi de regresie
<p>
Pentru modelele de antrenament am ales algoritmii de regresie liniară,
Ridge Regression și Decision Tree Regressor. După ce am antrenat fiecare
model, am calculat și predicțiile oferite de fiecare model de Machine
Learning. Analiza referitoare la rezultatele obținute în comparație cu
rezultatele așteptate este realizată în următoarea secțiune.
</p>

### Evaluarea performanței
<p>
Folosind predicțiile și rezultatele așteptate (expected vs.
predicted) am calculat metricile R squared și Root Mean Squared Error,
pe care le-am salvat într-un Data Frame care este afișat utilizatorului.
De asemenea, am afișat și o diagramă de tip scatter pentru a ilustra
deviația predicțiilor modelului față de rezultatele așteptate. Modelul
de regresie liniară obține rezultate foarte asemănătoare cu Ridge Regression,
ambele dintre ele fiind în aparență mai performante ca Decision Tree Regressor.
Cu toate acestea, cele două scoruri calculate (R squared și Root Mean
Squared Error) indică faptul că Decision Tree Regressor este mai precis decât
celelalte două.
</p>

### Bibliografie

Următoarele resurse au fost utilizate spre realizarea soluției prezentate:
- materialele de curs și laborator
- https://pandas.pydata.org
- https://scikit-learn.org/
- https://stackoverflow.com/questions/57348495/what-is-the-good-rmse-root-mean-square-error-value-range-to-justify-the-effici
- https://stackoverflow.com/questions/26778899/how-to-convert-matrix-to-pandas-data-frame
- https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/
- https://pandas.pydata.org/docs/reference/index.html
- https://stackoverflow.com/questions/17097236/replace-invalid-values-with-none-in-pandas-dataframe
- https://discuss.datasciencedojo.com/t/how-to-check-for-missing-values-in-a-pandas-dataframe-using-python/1046/2