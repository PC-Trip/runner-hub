# 1. Install requirements
```console
pip install -r requirements.txt
```

# 2. Go to dag directory
```console
cd dag
```

# 3. Plot dag  (e.g. grid sampler)
```console
python -m runner grid.yml --plot
```

# 4. Run dag
```console
python -m runner grid.yml
```

# 5. Get results (results will be on work directory)
```console
python -m runner results.yml
```

# 6. Docker
## 6.0 Build
```console
docker build -t tenex_container_simple .
```
## 6.1 Run
https://docs.docker.com/engine/reference/commandline/run/
### 6.1.1 Run test
```console
docker run -it --rm --env-file .env --entrypoint /bin/bash tenex_container_simple
python -m runner random_env.yml
```
### 6.1.2 Run
```console
docker run -it --rm --env-file .env tenex_container_simple
```
### 6.1.3 Run with mounting current directory to OPTUNA_WORKDIR in .env
```console
docker run -it --rm --env-file .env -v ${PWD}:/app/dag/work_env tenex_container_simple
```
### 6.1.4 Run with mounting current directory to directory with DAG
```console
docker run -it --rm --env-file .env -v ${PWD}:/app/dag tenex_container_simple
```
### 6.1.5 Run with changing input file from random_env.yml to random.yml
```console
docker run -it --rm --env-file .env tenex_container_simple random.yml
```
### 6.1.6 Run with changing input file from random_env.yml to random.yml from current directory
```console
docker run -it --rm --env-file .env -v ${PWD}:/app/dag tenex_container_simple random.yml
```
# 6.2 Compose up
https://stackoverflow.com/questions/65484277/access-env-file-variables-in-docker-compose-file
### 6.2.1 Compose up
```console
docker-compose up
```
### 6.2.2 Compose up with build
```console
docker-compose up --build
```
### 6.2.4 Compose down
```console
docker-compose down
```
## 6.3 Compose stack
https://docs.docker.com/engine/swarm/stack-deploy/
https://github.com/moby/moby/issues/29133#issuecomment-442912378
https://github.com/docker/compose/issues/7771
### 6.3.1 Deploy
```console
docker-compose build
docker-compose push
docker stack deploy -c <(docker-compose config|sed -E "s/cpus: ([0-9\\.]+)/cpus: '\\1'/") tenex_container_simple_stack
```
### 6.3.2 Update environment variable
https://docs.docker.com/engine/reference/commandline/service_update/
```console
docker service update --env-add OPTUNA_TRIALS=42 tenex_container_simple_stack
```
### 6.3.3 Remove
```console
docker service rm tenex_container_simple_stack
```
