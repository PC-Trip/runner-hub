=============
Docker Images
=============
# Show
docker images
# Remove
docker image prune
docker image rm
docker rmi

=================
Docker Containers
=================
# Show
docker container ls
# Show with stopped
docker container ls - a
# Remove
docker container prune
docker rm
docker rm --volumes

==============
Docker Volumes
==============
# Show
docker volume ls
# Remove
docker volume rm

==============
Docker Compose
==============
# docker-compose
https://docs.docker.com/compose/gettingstarted/
# docker-compose up
Builds, (re)creates, starts, and attaches to containers for a service.
https://docs.docker.com/compose/reference/up/
# memory/cpu
https://docs.docker.com/config/containers/resource_constraints/

============
Docker Build
============
# NOTE Run ./get_fenia.sh first!
docker-compose build
docker-compose up --build --no-start

==========
Docker Run
==========
docker-compose up
docker-compose -f docker-compose.yml up 

========================
Docker Run Interactive
========================
docker run -it IMAGE 
# Remove container after run
docker run -it --rm IMAGE 

==============
Docker Service
==============
# Show all
docker service ls
# Show one
docker service inspect --pretty SERVICE
# Create
docker service create [OPTIONS] IMAGE [COMMAND] [ARG...]
# Remove
docker service rm SERVICE
# Update
docker service update [OPTIONS] SERVICE
docker service update --replicas SERVICE
docker service update --replicas-max-per-node SERVICE
https://docs.docker.com/engine/reference/commandline/service_update/

============
Docker Stack
============
# docker stack deploy
When running Docker Engine in swarm mode, you can use docker stack deploy to deploy a complete application stack to the swarm. The deploy command accepts a stack description in the form of a Compose file.
https://docs.docker.com/engine/swarm/stack-deploy/
1.1 Test the app with Compose
docker-compose up -d . 
docker-compose up --build --no-start . 
1.2 Check
docker-compose ps
1.3 Stop the app
docker-compose down --volumes
2. Push the generated image to the registry
docker-compose push
3.1 Deploy the stack to the swarm
docker stack deploy --compose-file docker-compose.yml SERVICE_TAG
3.2 Check
docker stack services SERVICE_TAG
3.3 Stop
docker stack rm SERVICE_TAG


===============
Docker Registry
===============
https://docs.docker.com/registry/deploying/


============
Docker Swarm
============
# https://docs.google.com/document/d/12-JfAjAYF1LZqcuIIPvy6SCeNkN-qzFivJo25-W_Pws/edit
# Init https://docs.docker.com/engine/reference/commandline/swarm_init/
# Add nodes https://docs.docker.com/engine/reference/commandline/swarm_join/
docker swarm join [OPTIONS] HOST:PORT
docker swarm join-token [OPTIONS] (worker|manager)
# Remove node
docker swarm leave
# See nodes
docker node ls
# Inspect node
ocker node inspect --pretty <NODE-ID>
# Drain nodes https://docs.docker.com/engine/swarm/swarm-tutorial/drain-node/
docker node update --availability drain <NODE-ID>
# Undrain node
docker node update --availability active <NODE-ID>
# Visualizer https://github.com/dockersamples/docker-swarm-visualizer
git clone https://github.com/dockersamples/docker-swarm-visualizer
cd docker-swarm-visualizer
docker-compose up --no-start
docker service create \
  --name=viz \
  --publish=8080:8080/tcp \
  --constraint=node.role==manager \
  --mount=type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
  dockersamples/visualizer




