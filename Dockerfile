FROM carlasim/carla:0.9.16
USER root
RUN apt update && apt install -y python3-pip curl

USER carla
ENTRYPOINT [ "./CarlaUE4.sh" ]
CMD [ "-RenderOffScreen" ]
