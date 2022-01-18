FROM yidarvin/bmcv-base:latest

# Project Specific
RUN pip install fvcore omegaconf grad-cam h5py

# Path File
ENV PATH="/home/bmcv/firstaid_classification:${PATH}"
ENV PYTHONPATH="/home/bmcv/firstaid_classification:${PYTHONPATH}"

CMD ["bash"]
