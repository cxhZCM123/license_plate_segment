# 车牌分割
利用数字图像处理的方法，依赖python的open-cv库实现的车牌分割

通过修改路径读取包含车牌的图片

```python
car = cv.imread(local_path, 1)
```

通过修改HSV颜色空间中的阈值分割不同颜色的车牌

```python
lower_blue = np.array([100, 110, 110])
upper_blue = np.array([130, 255, 255])
lower_yellow = np.array([15, 55, 55])
upper_yellow = np.array([50, 255, 255]) #预置的蓝色和黄色阈值
```

显示结果

```python
cv.imshow('segment', segment)
```

