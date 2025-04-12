# Use the dustynv/nano_llm:r36.2.0 as the base image
FROM dustynv/nano_llm:r36.4.0

# Set the working directory inside the container
WORKDIR /app

# Copy the 'app' directory from the host to the container's working directory
COPY app /app

# Copy specific files to new locations inside the container
RUN cp /app/dynamic_plugin.py /opt/NanoLLM/nano_llm/plugins \
 && cp /app/robot.py /opt/NanoLLM/nano_llm/plugins/tools