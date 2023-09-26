FROM python:3.10.11

WORKDIR /klasifikasi_laporan/model

COPY klasifikasi_laporan .

COPY klasifikasi_laporan/model/requirements.txt .

RUN pip3 install -r requirements.txt

CMD ["python3", "batching.py"]