X光片检测患者肺炎

2020-3-2

2020-3-3 train。csv test。csv都被改了，读取不了。试着读取本地文件看看

2020-3-3 修正了wangyiUtil的bug，可以正常使用了

评测集答案分布

0-32.75

1-17.4

2-16.34

3-33.51

2020-3-3 灰度图片？ channel不能再用3，试试

2020-3-4 image = cv2.imread(path , cv2.COLOR_GRAY2BGR)

2020-3-5 predict to csv 写成了

del model保存权重，重置模型，读取权重。dataset shufle