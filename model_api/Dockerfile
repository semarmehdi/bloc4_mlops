FROM python:3.11-slim

RUN apt-get update -y 
RUN apt-get install nano unzip curl -y

# # THIS IS SPECIFIC TO HUGGINFACE
# # We create a new user named "user" with ID of 1000
# RUN useradd -m -u 1000 user
# # We switch from "root" (default user when creating an image) to "user" 
# USER user
# # We set two environmnet variables 
# # so that we can give ownership to all files in there afterwards
# # we also add /home/user/.local/bin in the $PATH environment variable 
# # PATH environment variable sets paths to look for installed binaries
# # We update it so that Linux knows where to look for binaries if we were to install them with "user".
# ENV HOME=/home/user \
#     PATH=/home/user/.local/bin:$PATH

# We set working directory to $HOME/app (<=> /home/user/app)
WORKDIR /home/app

# Leverage layer caching: copy only reqs first
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# # Copy all local files to /home/user/app with "user" as owner of these files
# # Always use --chown=user when using HUGGINGFACE to avoid permission errors
# COPY --chown=user . $HOME/app

COPY app.py /home/app/app.py

CMD ["bash","-lc","fastapi run app.py --host 0.0.0.0 --port ${PORT}"]