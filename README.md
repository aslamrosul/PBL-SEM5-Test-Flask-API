cara run aplikasi

1. python -m venv venv

2. venv\Scripts\activate

note: Jika eror jalankan ini
2.1 Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass


Jika berhasil, terminal akan berubah jadi:

(venv) C:\project\api_daun>

3. pip install -r requirements.txt

4. python app.py

Jika berhasil, akan muncul:

 * Running on http://0.0.0.0:5000


Coba buka di browser:

http://localhost:5000/predict