# -*- coding: utf-8 -*-
"""downloadBAMforest.py
"""
# This script downloads the BAMForests dataset to drive.

from google.colab import drive
import zipfile

drive.mount('/content/drive')

!wget -c -O "/content/drive/MyDrive/Bamberg_coco1024.zip" "https://pba-freesoftware.eoc.dlr.de/Bamberg_coco1024.zip"

!ls -lh /content/drive/MyDrive/Bamberg_coco1024.zip

drive.flush_and_unmount()