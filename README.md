#Mnist using StackedAutoencoder

chainerで何かとやりづらいStackedAutoencoderを作りました．  
MnistをStackedAutoencoderで学習します．  
でもぶっちゃけ精度が良くなったりとかはしません．  
##実行
    $python SdA.py -g 0

層の構造を変える時はSdA.pyの77行目  
    77     lsizes=[784,50,50,10]`
これを変えるだけでOKです．  
net.pyを変える必要はありません．  
層の活性化関数はsigmoidを使っているので，変えたい場合は`
net.py   
```
    40             if idx == self.size-1: break
    41             h = F.sigmoid(self.__getitem__("l{0}".format(idx + 1))(h))
    42         if self.pretrain:
    43             d = F.sigmoid(self.__getitem__("b{0}".format(idx + 1))(h))
    44             self.loss = F.mean_squared_error(hb, d)
```
のsigmoidをreluに変えるなどすればOKです．  
最適化関数はAdamを使っていますが，SGDなどを使いたい場合は  
SdA.py
```
    95     optimizer = optimizers.Adam()
```
をSGDなどに変えてみましょう．
以上です．

