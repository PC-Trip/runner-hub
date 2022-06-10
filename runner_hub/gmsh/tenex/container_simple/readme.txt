1. Install requirements
pip install -r requirements.txt

2. Go to dag directory
cd dag

3. Plot dag  (e.g. grid sampler)
python -m runner grid.yml  --plot

4. Run dag
python -m runner grid.yml

5. Get results (results will be on work directory)
python -m runner results.yml

6. Docker
https://stackoverflow.com/questions/65484277/access-env-file-variables-in-docker-compose-file
https://github.com/moby/moby/issues/29133#issuecomment-442912378
6.1 Run
docker run
6.2 Compose up
docker-compose up
6.3 Compose stack
docker stack deploy -c <(docker-compose config) stack-name-here
