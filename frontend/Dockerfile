FROM python:3.9-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos los archivos de requisitos al contenedor
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# copiamos todo el codigo
COPY . .

# Puerto que se expone de Flask dentro del contenedor. En el docker compose pondremos el que se expone fuera (el mismo)
EXPOSE 5000

# Para ejecutar la aplicación hacemos:
CMD ["python", "app.py"]
