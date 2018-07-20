# Build gensim image

In docker directory run the following command to build the image locally:

```
docker build -t gensim .
```

# Run ipython notebook with installed gensim

Just execute:

```
docker run -p 9000:9000 gensim
```

# Run the interactive bash mode

```
docker run -it gensim /bin/bash
```
