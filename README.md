# Introduction

Проект посвящен вычислению высот объектов на изображениях СЗМ-микроскопии с помощью сверточной нейросети с архитектурой U-net, на вход которой подается двухканальное изображение - анализируемое изображение и бинарная маска, оставляющая на изображении только наивысшие точки объектов, - а функцией потерь и метрикой является среднеквадратичное отклонение (MSE) между предсказанием модели и заранее размеченной картой высот объектов (вычисляется только по пикселям, выделенным маской). Проведено обучение модели на 187 изображениях, валидация и тестирование модели проводилось на 39 и 18 изображениях, соответственно. Модель выдает предсказание в виде одноканальной карты высот той же размерности, что и входное изображение, при этом на выходное изображение накладывается та же самая бинарная маска, которая подавалась в качестве второго канала на вход, она оставляет только искомые значения максимальных пикселей объектов.
В результате обучения была достигнута метрика на тестовом датасете ~ 0.78 нм2.

# Inference
Example of working inference (one file only)
[inference example](https://colab.research.google.com/github/dve2/Heights/blob/main/notebooks/inference_new.ipynb)

Example of working inference (any file from test dataset; downloads big file):
[inference example](https://colab.research.google.com/github/dve2/Heights/blob/main/notebooks/inference.ipynb)


# Training

[train example](https://colab.research.google.com/github/dve2/Heights/blob/main/notebooks/Train_2ch_ml_dm.ipynb)

Training dataset available by request

