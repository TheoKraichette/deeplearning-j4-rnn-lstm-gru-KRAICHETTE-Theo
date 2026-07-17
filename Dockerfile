FROM tensorflow/tensorflow:2.18.0-jupyter

COPY requirements.txt /tmp/requirements.txt
# blinker 1.4 est installe via distutils dans l'image de base : pip ne peut pas le
# desinstaller pour l'upgrade requis par streamlit. On force une version recente a part.
RUN pip install --no-cache-dir --ignore-installed blinker \
    && pip install --no-cache-dir -r /tmp/requirements.txt
