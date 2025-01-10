# littoral_refine

## Docker Setup

Build the Docker image:

```
docker build -t littoral_refine .
```

Run refine boundary test:

```
docker run --platform linux/amd64 -it --rm \
    -v $(pwd)/output:/output \
    littoral_refine
```
