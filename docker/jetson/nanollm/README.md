# NanoDog Container Setup Instructions

Follow these steps to build and run the **NanoDog** container:

---

## 1. Install Jetson Containers

Install **Jetson Containers** by following the instructions here:

👉 [Jetson Containers GitHub](https://github.com/dusty-nv/jetson-containers/tree/master)

---

## 2. Build the Docker Container

In the project folder, run:

```bash
docker build -t nanodog .

## 3. Run the Container

Run the container using Jetson Containers with your Hugging Face token:

```bash
jetson-containers run --env HUGGINGFACE_TOKEN=<> $(autotag nanodog)

## 4. Start Agent Studio in the Container

Inside the container, run:

```bash
python3 -m nano_llm.studio --verbose --load NousHermes-Pro

##5. Access Agent Studio

Open your browser and navigate to the IP address and port displayed in the container logs where Agent Studio is running.

##6. Attach Plugin Tool

Within Agent Studio, attach your desired plugin tool to test its functionality.

##7. Reset Chat

To reset the conversation in Agent Studio to update the model to your tools/plugin, type:

```bash
/reset
