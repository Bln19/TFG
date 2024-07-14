FROM jupyter/base-notebook:latest


RUN pip install --upgrade pip


# Copiar el archivo requirements.txt
COPY requirements.txt /tmp/

# Instalar las bibliotecas desde requirements.txt
RUN pip install -r /tmp/requirements.txt

# Descargar recurss adicionales
RUN python -m pip install nltk
RUN python -m nltk.downloader all
RUN python -m spacy download es_core_news_sm

# Copiar el archivo de configuración de Jupyter
COPY jupyter_notebook_config.py /home/jovyan/.jupyter/

USER root

# puerto Jupyter
EXPOSE 8888
