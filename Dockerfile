FROM python:3.7.6

ADD ./packages/ /packages/
RUN pip install --upgrade pip
RUN pip install -r /packages/requirements.txt

EXPOSE 5000

ENV PYTHONPATH /packages/api/:/packages/titanic/
WORKDIR /packages/titanic/
RUN python titanic/train_pipeline.py
WORKDIR /packages/api/
CMD ["gunicorn", "-b", ":5000", "run:application"]
