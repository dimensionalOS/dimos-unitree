FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY dimos/web/dimos_interface/package.json dimos/web/dimos_interface/yarn.lock ./

# Install dependencies
RUN yarn install

# Expose development port
EXPOSE 3000

# Start development server with host 0.0.0.0 to allow external connections
CMD ["yarn", "dev", "--host", "0.0.0.0"] 