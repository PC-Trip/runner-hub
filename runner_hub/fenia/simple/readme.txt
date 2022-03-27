0. Run 
./main.sh

or

1. Get FENIA
./get_fenia.sh
2. Build Docker image
docker-compose build
3. Save image to file
docker save --output fenia_latest.tar.gz 10.254.55.75/fenia:latest
4. Test https://docs.docker.com/engine/reference/run
4.1 Simple (Foreground) -v $(pwd):/work - map current directory to container's /work
cd work
docker run -v $(pwd):/work 10.254.55.75/fenia [COMMAND]
4.2 Detached (Background)
cd work
docker run -d -v $(pwd):/work 10.254.55.75/fenia [COMMAND]
4.3 Clean up the container
cd work
docker run --rm -d -v $(pwd):/work 10.254.55.75/fenia [COMMAND]
4.4 Check
docker ps
docker container ls
5. Clean results
rm -r [1-9]*
