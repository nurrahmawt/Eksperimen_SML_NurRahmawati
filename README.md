# Eksperimen SML - Nur Rahmawati

Dokumentasi ini hanya mencakup isi folder `Eksperimen_SML_NurRahmawati`.

## Tujuan

Proyek ini digunakan untuk proses eksperimen data preprocessing pada dataset harga rumah sebagai bagian dari tujuan proyek Dicoding.

## Struktur Folder (Scope Lokal)

- `houseprices_raw/`
	- `house_prices.csv`: dataset mentah.
- `preprocessing/`
	- `automate_NurRahmawati.py`: script preprocessing otomatis.
	- `Eksperimen_NurRahmawati.ipynb`: notebook eksperimen preprocessing.
	- `houseprices_preprocessing/`
		- `house_data_processed.csv`: hasil preprocessing.

## Cara Menjalankan Preprocessing

Jalankan perintah dari folder `Eksperimen_SML_NurRahmawati/preprocessing`:

```bash
python automate_NurRahmawati.py
```

Script akan membaca file:
- `../houseprices_raw/house_prices.csv`

Dan menghasilkan file:
- `houseprices_preprocessing/house_data_processed.csv`

## Catatan

Project ini sebagai syarat submission di Dicoding Indonesia
