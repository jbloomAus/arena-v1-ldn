docker build -t shakespeare:latest . --progress=plain
docker run -p 5000:5000 shakespeare:latest  