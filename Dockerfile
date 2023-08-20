FROM python:3.8.10

WORKDIR /app

COPY . /app

RUN pip install virtualenv

# Create a virtual environment
RUN virtualenv venv

# Activate the virtual environment
RUN /bin/bash -c "source venv/bin/activate"

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python3","app.py"]