FROM python:3.11

WORKDIR container/

COPY . .

RUN pip install -r requirements.txt

CMD python backend/api.py