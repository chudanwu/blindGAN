<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

## 问题建模
把去模糊和超分问题一起考虑，可以得到如下建模：
$$ 已知：退化图 L；未知：退化核 k；下采样操作：D；斯噪声 n；高清图 H；待求解：重建图 X。$$
- 在blind-deblur中，他们的关系是：
$$ H*k+n=L $$
- 在super-resolution中，他们的关系是：
$$ D \circ (H*k)=L$$


### 超分重建
- SCSR (08cvpr稀疏编码):
$$ argmin_X ||X-H|| \quad s.t.\quad D \circ (H*k)=L $$
- SRCNN (15cvpr深度卷积)：
$$ argmin_{\theta}MSE(X_\theta-H) $$
- SRGAN (17cvpr深度生成对抗)：
$$ argmin_{\theta} ( MSE(\phi(X_\theta )-\phi(H)) + 对抗loss)$$

均在文中明确指出，目标是使恢复的图像X逼近H。

### 盲去模糊
- DL blur kernel估计（2016IP）：

    这篇文章中，由于估计的是核的参数，因此最小化的是参数和参数groundTruth的误差。
- text deblur by L0 正则化（2017pami）：
$$ argmin_{X,k} ||X*k-L||^2_2 + \lambda l_0(X) + \gamma l_2(k)$$

    迭代优化求解X,k。其中，求k的时候，用的是（X的梯度*k）和（L的梯度）。这里之所以和SR反过来，我觉得是因为这是个无监督的方法(用不上H)，所以只能和L比较。
- direct text deblur：
    
    作者明确指出，由于真实模糊比模糊核卷积更复杂,所以比起模拟模糊过程，模拟去模糊过程更实际：
    $$ argmin_\theta MSE(X-H) + 0.0005 ||\theta||^2_2$$


## Loss

$$单个样本。输入：x,标签：y,输出：F(x),网络参数：\theta,特征：\phi(x)$$
下面的loss都是适用于batch的，即对一个batch，采取了均值loss。

|名称|表达式|作用|适用情况|
|:--|:-----|:--|:-------|
|L1 loss|$$ \|F(x)-y\|_1 $$|输出大体上逼近y|普适|
|mse loss|$$ \frac{1}{CHW}\|F(x)-y\|^2_2 $$|输出大体上平滑的逼近y|普适|
|perceptual loss|$$ \|\phi(F(x))-\phi(y)\|_2 $$|输出的内容逼近y的内容|语义信息|
|adversaral loss|$$ argmin_{G}max_{D}  [\log D(x,y)+\log(1-D(x,F(x)))]$$|判别网络无法区分输出和y是否condition的区别在于D是两个输入还是一个|在判别网络中|
|adversaral loss|$$ argmin_{G}max_{D}  [\| D(x,y)\|_2+\|1-D(x,F(x))\|_2]$$|替换negative log likelihood为least quare，
|gram loss|$$ \|gram(\phi(F(x)))-gram( \phi(y))\|_2 $$ |输出的纹理逼近y的纹理|风格转换中不含语义|
|Lp loss|$$ \|F(x)-y\|_p $$|在L1与L2中折中|比L1、L2效果都要好时|
|identity loss|$$ \|x-F(x)\|_1 $$|当输入本身就是输出同一类时，输入=输出|有助于在transfer任务中保留颜色|
|TotalVariant loss|$$\|\nabla _H F(x)\|_1+\| \nabla _W F(x)\|_1$$ |平滑输出，削弱artifact|去噪中常见|
|l1/l2 piror|$$ \|F(x)\|_1 \quad or \quad \|F(x)\|^2_2 $$|不知道有没有用
|l2 reg|$$ \|\theta\|_1  \quad or \quad \|\theta\|^2_2 $$|参数稀疏/用小参数可以帮助收敛|非必须




## 结构
|网络结构|作用|适用情况|出自|
|:------|:--|:-------|:--|
|瓶颈结构|减少参数|图片较大，追求速度，中间层数较多|>-< 形状流行于 fast-style-transfer|
|skip connection|有选择性的保留输入|重建精细的任务|出自 U-net，流行于后来的SR、GAN任务|
|resisual learning|只学习残差，帮助收敛|普遍适用|已广泛应用于CNN|
|IN/BN|降低特征表达学习难度，帮助收敛|？？？因为有文章说可以去掉|已广泛应用于CNN|
|GAN|获得一个非设计的loss：对抗loss|设计的loss效果不好|流行于生成式任务|
|buffer判别式|在GAN训练D时，不选用最新G输出而是保留历史G输出作为“fake”|避免模型参数震荡|来自pix2pix的引用文章|
|deconv|有upconv，transconv，subpixel三种|第一种效果最好也最慢，第三种最快|第一种SRCNN在用，第三种来自ESPCN，第二种fast-style-transfer系列在用|
|patchGAN|判别器输出非1维而是dxd矩阵，即感受域只对应输入一部分而非全部，加速训练|假设像素符合马尔科夫特性|pix2pix引用的文章|

## 实验及对应结果

1. 非盲去模糊可行性实验：

    参数： 
    motion blur:len=7,;length=

    数据集：
    training set=COCO 2014;test set=9 img from yang
    
    网络结构：
    瓶颈结构+skipconnect+IN+res learning+ 8 res_block +subpixel。

    loss：
    L1 loss。其他reg项已经实现但还没用

    优化方法：
    adam

    核参数：运动=0.gauss=1.7
|img-name |blurPSNR | +PSNR |
|:--------|--------:|------:|
|comic.bmp	 |-1.876827	|32.462749|
|pepper.bmp	 |2.274908	|33.841355|
|man.bmp	 |-1.232231	|33.332980|
|flowers.bmp	 |-0.823303	|35.407222|
|zebra.bmp	 |-2.251888	|36.726342|

2. pair blindGAN ：
    
    loss：
    $$ ldentity\ loss:\ argmin_G = E|| G(H)-\delta ||_1$$
    $$ generate\ loss:\ argmin_G = E|| H*G(L)-L ||_2$$
    $$ GAN\ loss:\ argmin_Gmax_D = E[\log(D(L,k))] + E[1-\log(D(L,G(x)))]$$

    结构：

    G：
        scaling：1x116x116->1x1x29
        resblock：conv+norm+relu
        lastactivate：如果输入norm到（mean=0.5,std=0.5），用tanh；norm到（mean=0,std=1），用sigmoid。
    
    D：
        fully convolution layer: conv+norm+leakyrelu(sigmoid for last 0-1 output)

    预处理：
        G的输入random crop
        D用上history pool
        注意使用detach避免不必要的梯度反向


    