FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of application code
COPY . .

# Expose the application port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]




# build command - docker build -t iamsaurav/langconv-api .
# docker login command - docker login
# push command - docker push iamsaurav/langconv-api
# pull command - docker pull iamsaurav/langconv-api
# run command - docker run -p 8000:8000 iamsaurav/langconv-api
# stop command - docker stop <container_id>