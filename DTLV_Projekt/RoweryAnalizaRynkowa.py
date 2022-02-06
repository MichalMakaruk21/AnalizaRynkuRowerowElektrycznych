import pandas as pd
import numpy as np
from IPython.display import display
from functools import reduce
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

#Ładowanie plików z danymi źródłowymi pobranymi z Banku Danych Loakalnych GUS
dfCwiczacy = pd.read_csv('DaneGUSPobrane/CwiczacyOgolem.csv', names=['KodTeryt', 'IloscCwiczacych'], decimal=',',sep=';') #Osoby aktywnie ćwiczące w klubach sportowych
dfS_Rowerowe = pd.read_csv('DaneGUSPobrane/DlugoscSciezekRowerowych.csv', names=['KodTeryt', 'DlugoscSciezekRowerowych'], decimal=',',sep=';')#Łączna długość ścieżek rowerowych w powiecie
dfP_Wynagrodzenie = pd.read_csv('DaneGUSPobrane/PrzecietnieWynagrodzenie.csv', names=['KodTeryt', 'SrednieWynagrodzenie'], decimal=',',sep=';')#Średnie wynagrodzenie brutto na jedną osobę
dfT_Lipiec = pd.read_csv('DaneGUSPobrane/TurystykaONoclegoweLipiec.csv', names=['KodTeryt', 'IloscNoclegowWSezonie'],decimal=',',sep=';')#Ilość miejsc noclegowych w obiektach turystcznych otwartych w lipcu
dfLudnosc = pd.read_csv('DaneGUSPobrane/Ludnosc.csv', names=['KodTeryt', 'IloscMieszkancow'], decimal=',',sep=';')#Ilość ludności powiatu międzu 15 a 45 lat
dfKodTertyt_Powiat = pd.read_csv('DaneGUSPobrane/KodTeryt_Powiat.csv', names=['KodTeryt', 'Powiat'], decimal=',',sep=';')#Powiat i jego kod

#Przygotowanie danych do mergowania
dane = [dfLudnosc, dfCwiczacy, dfP_Wynagrodzenie, dfS_Rowerowe, dfT_Lipiec]

#Złączenie w jeden dataframe wg. Kodu Terytorialnego
dfRowery = reduce(lambda left, right: pd.merge(left, right, on=['KodTeryt'], how='outer'), dane)

#Zastąpienie pustych rekordów zerem(nie każdy powiat ma ścieżki rowerowe i miejsca noclogowe)
dfRowery['DlugoscSciezekRowerowych'].fillna(0, inplace=True)
dfRowery['IloscNoclegowWSezonie'].fillna(0, inplace=True)

#wycięcie z dataframe Kodu terytorialnego i przechowanie go pod zmienną
dfKodTertyt = dfRowery['KodTeryt']

# Metoda indeksujaca dane od 0 do 100 wg maksymalnej i minimalnej wartosci, przechwowywana pod zmienną
scaler = MinMaxScaler(feature_range=(0, 100))
#Stworzenie datarame z danymi liczbowymi, który będzie normalizowany przez funkcje MinMaxScaler
dfRoweryNumbers = dfRowery.drop(columns='KodTeryt')


#Wywołanie metody MinMaxScaler(Skaluje oddzielnie każdą kolumne w przedziale 0-100)
dfRoweryNumbers[dfRoweryNumbers.columns] = scaler.fit_transform(dfRoweryNumbers)

#Dodanie kolumny z indeksem dotyczących salonów rowerowych, oblicznym na podstawie zeskalowanych danych z uwzględnieniem wag.
#Zakładamu że największą wagę ma wynagrodzenie, następnie ilość mieszkańców a, potem ilość osób aktywnie ćwiczących oraz długośc ścieżek rowerowych.
dfRoweryNumbers['SalonIndeks'] = dfRoweryNumbers['IloscMieszkancow'] * 2 + dfRoweryNumbers['SrednieWynagrodzenie'] * 3 + dfRoweryNumbers['IloscCwiczacych'] + dfRoweryNumbers['DlugoscSciezekRowerowych']

#Dodanie kolumny z indeksem oblicznym dotyczących wypożyczalni rowerowych, na podstawie zeskalowanych danych z uwzględnieniem wag.
#Zakładamu że największą wagę ma ilość miejsc noclegowych w sezonie, ale uwzględniamy tez długość ściezek rowerowych
dfRoweryNumbers['WypozyczalniaIndeks'] = dfRoweryNumbers['IloscNoclegowWSezonie'] * 3 + dfRoweryNumbers['DlugoscSciezekRowerowych'] * 2

#Ponowne skalowanie, które zeskaluje indeks w wartościach 0-100
dfRoweryNumbers[dfRoweryNumbers.columns] = scaler.fit_transform(dfRoweryNumbers)
#Zaokrąglenie wartości od licz całkowitych
dfRoweryNumbers = dfRoweryNumbers.round(decimals=0)

#Przygowtowanie dataframe zawierającego ostateczny wynik. Listę kodów terytorialnych z indeksem potencjału powiatu pod względem otwarcia salonu rowerowego i wypożyczalni
dfSalonFinal = pd.DataFrame([dfKodTertyt, dfRoweryNumbers.SalonIndeks.astype(np.int64)]).transpose()
dfWypozyczalniaFinal = pd.DataFrame([dfKodTertyt, dfRoweryNumbers.WypozyczalniaIndeks.astype(np.int64)]).transpose()

#Dodanie nazw powiatu do wg.kodów terytorialych
dfSalonFinal = pd.merge(dfKodTertyt_Powiat, dfSalonFinal, on='KodTeryt', how='inner')
dfWypozyczalniaFinal = pd.merge(dfKodTertyt_Powiat, dfWypozyczalniaFinal, on='KodTeryt', how='inner')

#Usunięcie hisorycznego pliku i ponowny zapis pliku
fileIndeksSalon = 'Dane_Indeks/SalonIndeks.csv'
if(os.path.exists(fileIndeksSalon) and os.path.isfile(fileIndeksSalon)): os.remove(fileIndeksSalon)
dfSalonFinal.to_csv(fileIndeksSalon, sep=';')

fileIndeksWypozyczalnia = 'Dane_Indeks/WypozyczalniaIndeks.csv'
if(os.path.exists(fileIndeksWypozyczalnia) and os.path.isfile(fileIndeksWypozyczalnia)): os.remove(fileIndeksWypozyczalnia)
dfWypozyczalniaFinal.to_csv(fileIndeksWypozyczalnia, sep=';')

#Wyświtlenie na konsoli 5 powiatów rekomendowaych do otwacia salonu wraz z indeksem
display('Powiaty z największym potencjałem, reokmendowane do otwarcia salonów rowerów elektrycznych')
display(dfSalonFinal.sort_values(by=['SalonIndeks'], ascending=False).head(5))

#Wyświtlenie na konsoli 5 powiatów rekomendowaych do otwacia salonu wraz z indeksem
display('Powiaty z największym potencjałem, reokmendowane do otwarcia wypożyczalni rowerów elektrycznych')
display(dfWypozyczalniaFinal.sort_values(by=['WypozyczalniaIndeks'], ascending=False).head(5))

#Przygowtowanie danych i modelowanie histogramu
fig, ((ax0), (ax1)) = plt.subplots(nrows=2, ncols=1)

ax0.hist(dfSalonFinal['SalonIndeks'], density=True, histtype='bar', bins=50, stacked=True)
ax0.set_title('Dystrybucja indeksu potencjału salonu')

ax1.hist(dfWypozyczalniaFinal['WypozyczalniaIndeks'], density=True, histtype='bar', bins=50, stacked=True)
ax1.set_title('Dystrybucja indeksu potencjału wypożyczalni')

fig.text(0.5, 0.01, 'Indeks potencjału powiatu', ha='center')
fig.text(0.01, 0.5, 'Ilość powiatów', va='center', rotation='vertical')

fig.tight_layout()

#Zapis histogramu do pliku
fileHistSalon = 'Dane_Indeks/SalonHistogram.png'
if(os.path.exists(fileHistSalon) and os.path.isfile(fileHistSalon)): os.remove(fileHistSalon)
plt.savefig(fileHistSalon)

#Prezentacja histogramu
plt.show()
