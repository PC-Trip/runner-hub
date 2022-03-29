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