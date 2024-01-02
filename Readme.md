# Enhancing Hyperspectral Image Classification:Leveraging Unsupervised Information With Guided Group Contrastive Learning

[Ben Li](), [Leyuan Fang](), [Ning Chen](), [Jitong Kang](), [Jun Yue]()
___________

The code in this toolbox implements the ["Enhancing Hyperspectral Image Classification:Leveraging Unsupervised Information With Guided Group Contrastive Learning"](https://ieeexplore.ieee.org/document???). 

**The codes for this research include the GGCL network model: ./src/models/cross_transformer.py.**

More specifically, it is detailed as follow.

![alt text](./framework.png)

Citation
---------------------

**Please kindly cite the papers if this code is useful and helpful for your research.**

```


```

```


```


How to use it?
---------------------
1. Prepare raw data
   * Raw data is origin HSI data, likes IP, PU, SA datasets, You need to separate the training and test sets in advance. Or you can download ours from [baiduyun](https://pan.baidu.com/s/19-YNNIjQxEOz-gl3vCLuDg), extract codes is ```pabk```.
   * The classification module requires providing the features extracted by the diffusion module as input. We provide the diffusion features extracted in our experiments for researchers to reproduce the results. For the convenience of testing, we have provided all diffusion features data before PCA. Please download the specific data from [baiduyun](https://pan.baidu.com/s/19-YNNIjQxEOz-gl3vCLuDg), extract codes is ```pabk```. 
2. Configure the name, path and the parameters of dataset in ./src/params and ./src/workflow.py
3. Run the code to train and use the GGCL model.
   ```
   python workflow.py
   ```

Others
----------------------
If you want to run the code in your own data, you can accordingly change the input (e.g., data, labels) and tune the parameters.

If you encounter the bugs while using this code, please do not hesitate to contact us.

Licensing
---------

Copyright (C) 2023 Ning Chen

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
