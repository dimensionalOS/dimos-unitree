FROM dimensionalos/unitree-agents-interface

RUN apt-get update && apt-get install -y \
    fish \
    git 

COPY motd /etc/motd
COPY bash.sh /root/.bash.sh

RUN echo "cat /etc/motd" >> /root/.bashrc
RUN echo "source /root/.bash.sh" >> /root/.bashrc
