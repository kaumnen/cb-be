# cb-be

# local development

conda create --name cb-be python=3.12

conda activate cb-be

pip install -r requirements.txt

python main.py

# run as a docker container

docker-compose up --build
