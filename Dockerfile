FROM ubuntu:20.04
ARG JUPYTER_PASSWORD="feature_selection"
ENV BROWSER=/browser \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8
COPY Makefile.docker Makefile
COPY . feature_selection/

RUN apt-get update && \
	apt-get install -y --no-install-suggests --no-install-recommends make cmake && \
    make install-python3.8 && \
    make install-common-dependencies && \
    make install-python-libs

RUN cd feature_selection \
    && python3 -m pip install -U pip \
    && pip3 install -r requirements-lint.txt  \
    && pip3 install -r requirements-test.txt  \
    && pip3 install -r requirements.txt  \
    && pip install ipython jupyter \
    && pip3 install -e . \
    && git config --global init.defaultBranch master \
    && git config --global user.name "Whoever" \
    && git config --global user.email "whoever@fragile.tech"

RUN make remove-dev-packages

RUN mkdir /root/.jupyter && \
    echo 'c.NotebookApp.token = "'${JUPYTER_PASSWORD}'"' > /root/.jupyter/jupyter_notebook_config.py
CMD jupyter notebook --allow-root --port 8080 --ip 0.0.0.0
