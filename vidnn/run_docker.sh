docker run -it \
--ipc=host \
--gpus=all \
--privileged \
--network=host \
--log-driver json-file \
--log-opt max-size=10m \
--log-opt max-file=3 \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /home/robotegra/michael:/mnt/michael \
-w /mnt/michael \
--name vidnn \
rtg/vidnn:v0

