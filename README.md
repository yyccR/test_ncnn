## Test NCNN c++ in PC

### 1. yolov5-segment (torchScript->pnnx->ncnn)

![dog](/data/traffic_road_seg.jpg)

Detect层后处理修改如下, 直接return`x_cat`:
```python
def forward(self, x):
    """Concatenates and returns predicted bounding boxes and class probabilities."""
    shape = x[0].shape  # BCHW
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    if self.training:
        return x
    elif self.dynamic or self.shape != shape:
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        self.shape = shape

    x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
    return x_cat
    # if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
    #     box = x_cat[:, :self.reg_max * 4]
    #     cls = x_cat[:, self.reg_max * 4:]
    # else:
    #     box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    # dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
    # y = torch.cat((dbox, cls.sigmoid()), 1)
    # return y if self.export else (y, x)
```

### 2. yolov8-detect (torchScript->pnnx->ncnn)

![dog](/data/traffic_road_detect_v8.jpg)

Detect层后处理修改如下, 直接return`x_cat`:
```python
def forward(self, x):
    """Concatenates and returns predicted bounding boxes and class probabilities."""
    shape = x[0].shape  # BCHW
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    if self.training:
        return x
    elif self.dynamic or self.shape != shape:
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        self.shape = shape

    x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
    return x_cat
    # if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
    #     box = x_cat[:, :self.reg_max * 4]
    #     cls = x_cat[:, self.reg_max * 4:]
    # else:
    #     box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    # dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
    # y = torch.cat((dbox, cls.sigmoid()), 1)
    # return y if self.export else (y, x)
```

### 3. yolov8-segment (torchScript->pnnx->ncnn)

![dog](/data/traffic_road_seg_v8.jpg)

Detect层后处理修改如下, 直接return`x_cat`:
```python
def forward(self, x):
    """Concatenates and returns predicted bounding boxes and class probabilities."""
    shape = x[0].shape  # BCHW
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    if self.training:
        return x
    elif self.dynamic or self.shape != shape:
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        self.shape = shape

    x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
    return x_cat
    # if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
    #     box = x_cat[:, :self.reg_max * 4]
    #     cls = x_cat[:, self.reg_max * 4:]
    # else:
    #     box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    # dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
    # y = torch.cat((dbox, cls.sigmoid()), 1)
    # return y if self.export else (y, x)
```

### 4. yolov8-pose (torchScript->pnnx->ncnn)

![dog](/data/coco128_625-pose.jpg)

Detect层后处理修改如下, 直接return`x_cat`:
```python
def forward(self, x):
    """Concatenates and returns predicted bounding boxes and class probabilities."""
    shape = x[0].shape  # BCHW
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    if self.training:
        return x
    elif self.dynamic or self.shape != shape:
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        self.shape = shape

    x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
    return x_cat
    # if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
    #     box = x_cat[:, :self.reg_max * 4]
    #     cls = x_cat[:, self.reg_max * 4:]
    # else:
    #     box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    # dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
    # y = torch.cat((dbox, cls.sigmoid()), 1)
    # return y if self.export else (y, x)
```

Pose层后处理修改如下, `kpts_decode`注释掉,将该处理放在c++推理处实现
```python
def forward(self, x):
    """Perform forward pass through YOLO model and return predictions."""
    bs = x[0].shape[0]  # batch size
    kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
    x = self.detect(self, x)
    if self.training:
        return x, kpt
    # pred_kpt = self.kpts_decode(bs, kpt)
    # return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))
    # return torch.cat([x, kpt], 1) if self.export else (torch.cat([x[0], kpt], 1), (x[1], kpt))
```

### 4. real-sr (torchScript->pnnx->ncnn)

![real_sr_test.png](/data/real_sr_test.png)

1. 下载 [DF2K.pth](https://drive.google.com/open?id=1pWGfSw-UxOkrtbh14GeLQgYnMLdLguOF) 和 [DPED.pth](https://drive.google.com/open?id=1zZIuQSepFlupV103AatoP-JSJpwJFS19) 模型
2. 拷贝模型到 `pretrained_model` 目录下
3. 修改 `/codes/options/df2k/test_df2k.yml` , `/codes/options/dped/test_dped.yml` 中 `path: pretrain_model_G` 参数.
4. 在 `/codes/test.py` 脚本在 `model = create_model(opt)` 之后加上如下, 同级目录可看到转换后的ncnn模型文件:
```python
import pnnx
x = torch.rand(1, 3, 320, 320)
opt_model = pnnx.export(model.netG, "dped.pt", x)
result = opt_model(x)
```
5. `python3 test.py -opt options/df2k/test_df2k.yml` 
6. `python3 test.py -opt options/dped/test_dped.yml`