# Build gensim image

In docker directory run the following command to build the image locally:

```
docker build -t gensim .
````

After that, you will need to search the gensim_image_id and tag the image using:

```
# check the image
docker image ls

# generage the tag
docker tag [gensim_image_id] [my_user]/gensim:latest
```

Run the interactive bash mode:

```
docker run -it [my_user]/gensim /bin/bash
```

Run gensim tests in the interactive bash mode:

```
# run tests with desired python version
cd gensim
python2 setup.py test
```

To push the image to docker hub:

```
# login to docker hub
docker login

# push image to docker hub
docker push [my_user]/gensim
```

# Run gensim image from anywhere

Just execute:

```
docker run -p 9000:9000 [my_user]/gensim
```