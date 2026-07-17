FROM tensorflow/tensorflow:2.18.0-jupyter

COPY requirements.txt /tmp/requirements.txt
# blinker 1.4 (distutils) bloque l'upgrade voulu par streamlit : on le force a part
RUN pip install --no-cache-dir --ignore-installed blinker \
    && pip install --no-cache-dir -r /tmp/requirements.txt
