FROM python:3.7.6

ADD ./packages/ /packages/
RUN pip install --upgrade pip
RUN pip install -r /packages/requirements.txt

EXPOSE 5000

WORKDIR /packages/
ENV PYTHONPATH /packages/titanic/
CMD ["python", "api/api/app.py"]
