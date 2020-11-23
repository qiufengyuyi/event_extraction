FROM tensorflow/tensorflow:1.12.0-gpu-py3
RUN mkdir /event_extraction
WORKDIR /event_extraction
COPY requirements.txt /event_extraction/
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple