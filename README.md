# Introduction

Проект посвящен вычислению высот объектов на изображениях СЗМ-микроскопии с помощью сверточной нейросети с архитектурой U-net, на вход которой подается двухканальное изображение - анализируемое изображение и бинарная маска, оставляющая на изображении только наивысшие точки объектов, - а функцией потерь и метрикой является среднеквадратичное отклонение (MSE) между предсказанием модели и заранее размеченной картой высот объектов (вычисляется только по пикселям, выделенным маской). Проведено обучение модели на 187 изображениях, валидация и тестирование модели проводилось на 39 и 18 изображениях, соответственно. Модель выдает предсказание в виде одноканальной карты высот той же размерности, что и входное изображение, при этом на выходное изображение накладывается та же самая бинарная маска, которая подавалась в качестве второго канала на вход, она оставляет только искомые значения максимальных пикселей объектов.
В результате обучения была достигнута метрика на тестовом датасете ~ 0.78 нм2.


# Installation

For local run.

Clone the repo:

    git clone https://github.com/dve2/Heights.git

Make [virtual environment](https://docs.python.org/3/library/venv.html):

    python -m venv .venv
    

Activate .venv and install required packages:

    source venv/bin/activate
    pip install -r requirements.txt
    

For use in colab:

    TODO add colab notebook for inference


# Inference

1. Open project folder in terminal

        cd Heights

2. Copy **input files** (TODO : change to proper names of microscope outputs)
to [Inference folder](Inference)
3. Activate virtual environment

       source venv/bin/activate
4. 
5. Run [predict.py](predict.py)

       python predict.py
6. Results will be saved into 




Example of working inference (one file only)
[inference example](https://colab.research.google.com/github/dve2/Heights/blob/main/notebooks/inference_new.ipynb)

Example of working inference (any file from test dataset; downloads big file):
[inference example](https://colab.research.google.com/github/dve2/Heights/blob/main/notebooks/inference.ipynb)


# Training

[train example](https://colab.research.google.com/github/dve2/Heights/blob/main/notebooks/Train_2ch_ml_dm.ipynb)

Training dataset available by request

