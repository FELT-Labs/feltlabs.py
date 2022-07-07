# syntax=docker/dockerfile:1

FROM python:3.9-slim-bullseye
# slim docker doesn't contain gcc
# this install gcc for build of some packages and remove it afterwards
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && pip install feltlabs \
    && apt-get purge -y --auto-remove gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/* 
