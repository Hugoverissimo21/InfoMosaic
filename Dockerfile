# Use a lightweight base image with Python
FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# Copy environment file and install dependencies
COPY environment.yaml .

# Copy the assets and data into the container

COPY assets /app/assets
COPY data /app/data

# Copy the script into the container
COPY data02.py .

# Set environment name directly (change to your environment name)
ENV ENV_NAME=infomosaic

# Create the environment from environment.yaml
RUN conda env create -f environment.yaml

# Set the default environment (by using the predefined environment name)
RUN echo "source activate $ENV_NAME" > ~/.bashrc && \
    echo "export PATH=/opt/conda/envs/$ENV_NAME/bin:$PATH" >> ~/.bashrc

# Make RUN commands use the new environment:
RUN conda run -n infomosaic python -m spacy download pt_core_news_sm





# Run the Python script (use one CMD only)
CMD ["conda", "run", "--no-capture-output", "-n", "infomosaic", "python", "data02.py"]
