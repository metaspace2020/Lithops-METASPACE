# Python 3.7.5
FROM python:3.8.5-slim-buster

RUN apt-get update && apt-get install -y \
        zip \
        && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade setuptools six pip \
    && pip install --no-cache-dir \
        gunicorn==19.9.0 \
        glob2 \
        PyYAML \
        kubernetes

RUN pip install --upgrade setuptools six \
    && pip install --no-cache-dir \
        simplejson==3.16.0 \
        httplib2==0.12.1 \
        kafka_python==1.4.4 \
        lxml==4.3.1 \
        python-dateutil==2.7.5 \
        pika==0.13.1 \
        flask==1.1.1 \
        gevent==1.4.0 \
        ibm-cos-sdk==2.5.3 \
        redis==3.3.8 \
        # Modules for annotation-pipeline (These should match setup.py)
        pandas==1.1.3 \
        pyarrow==1.0.1 \
        scipy==1.5.3 \
        pyImagingMSpec==0.1.4 \
        cpyImagingMSpec==0.2.4 \
        pyMSpec==0.1.2 \
        cpyMSpec==0.3.5 \
	    pyimzML==1.4.1 \
        requests==2.22.0 \
        msgpack==0.6.2 \
        msgpack-numpy==0.4.4.3 \
	    pypng==0.0.19


# Copy PyWren proxy and lib to the container image.
ENV APP_HOME /pywren
WORKDIR $APP_HOME

COPY pywren_knative.zip .
RUN unzip pywren_knative.zip && rm pywren_knative.zip

CMD exec gunicorn --bind :$PORT --workers 1 --timeout 600 pywrenproxy:proxy
