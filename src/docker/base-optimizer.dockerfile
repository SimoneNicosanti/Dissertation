FROM python:3.9-bookworm

COPY ./Other/cplex_installer.bin /root/cplex_installer.bin
COPY ./docker/cplex.properties /root/cplex.properties

RUN chmod +x /root/cplex_installer.bin

RUN /root/cplex_installer.bin -i silent -f /root/cplex.properties

RUN rm /root/cplex_installer.bin