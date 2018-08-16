# Result
### image
- tensorBoard图
- tensorBoardX 18层网络 batchSize = 100
    - 大图
    ![](SENet-batch100-18-tensorboardX-big.png)
    - 小图
    ![](SENet-batch100-18-tensorboardX-small.png)

- log图
- 18层网络
    - batchSize = 64 最终结果
![](./image/SENet-batch64-last-18.png)
    - batchSize = 64 最大结果
![](./image/SENet-batch64-max-18.png)

    - batchSize = 128 最终结果
![](./image/SENet-batch128-last-18.png)

- 34层网络
    - batchSize = 64 最终结果
![](./image/SENet-batch64-last-34.png)
    - batchSize = 128 最终结果
![](./image/SENet-batch128-last-34.png)



### visdom
- visdom启动
```python
>>> python -m visdom.server
```
- 可视化查看地址:http://localhost:8097/ or http://serverIP:8097/


### tensorBoardX
- 安装
```python
>>> pip install tensorboardX
```
- 使用
```python
 from tensorboardX import SummaryWriter
 writer = SummaryWriter('log')
```
- 添加曲线
```python
 writer.add_scalars('data/group', {'train_loss':loss, 'train_acc':acc}, train_step/100)
```
