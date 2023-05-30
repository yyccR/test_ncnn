## Test NCNN c++ in PC

### yolov5-segment (torchScript->pnnx->ncnn)

![dog](/data/traffic_road_seg.jpg)

### yolov8-segment (torchScript->pnnx->ncnn)

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
