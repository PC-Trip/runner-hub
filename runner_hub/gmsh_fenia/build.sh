rm -r OpenFEM-0.0.x -f \
&& git clone -b release git@svn100.ibrae:/home/git/opf/OpenFEM-0.0.x.git \
&& docker-compose build --no-cache \
&& docker-compose push