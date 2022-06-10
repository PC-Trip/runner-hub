FROM ubuntu:20.04

# libxcursor1 - OSError: libXcursor.so.1: cannot open shared object file: No such file or directory
# libxft2 - OSError: libXft.so.2: cannot open shared object file: No such file or directory
# libxinerama1 - OSError: libXinerama.so.1: cannot open shared object file: No such file or directory
# libglu1 - OSError: libGLU.so.1: cannot open shared object file: No such file or directory
# libxrender1 - OSError: libXrender.so.1: cannot open shared object file: No such file or directory
# libpq-dev - pip3 postgres install error
RUN apt-get update && apt-get install -y \
    git \
    libglu1 \
    libpq-dev \
    libxcursor1 \ 
    libxft2 \
    libxinerama1 \
    libxrender1 \
    python3-pip \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /app/dag
ENTRYPOINT ["python", "-m", "runner"]
CMD ["random_env.yml"]