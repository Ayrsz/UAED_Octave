docker build .  -t edge_detection
docker run --name UAED -v "$(pwd)/data:/code/datasets" --gpus all --privileged -d -it edge_detection:latest 
#Utilizando volume do dataset do UAED