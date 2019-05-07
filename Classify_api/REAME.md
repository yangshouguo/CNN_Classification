#Classify_api目录

该目录是为了将训练的模型加载和封装成接口

## compile_level_classi.py

使用 python compile_level_classi.py -h 查看使用帮助

* 该脚本模型放在 checkpoints 文件夹下
* 模型路径在代码中指定，初始化类时作为参数传入

    cla = Classifier('./checkpoints/model-182800.meta')
    
* python compile_level_classi.py -f path_to_binary


## test_optimization_recog.py

该文件是为了测试模型的性能，如果一个文件分成几份之后的预测结果的众数占预测结果的一半以上
那我们认为这次预测是有效的

