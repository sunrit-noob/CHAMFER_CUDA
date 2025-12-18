# CHAMFER_CUDA

+ author: [sunrit2022pers@gmail.com](sunrit2022pers@gmail.com)

#### Performance

+ dim   = 5
+ loop  = 5
+ Intel(R) Xeon(R) CPU @ 2.20GHz
+ Tesla T4

| PC1     | PC2     | sklearn  | CUDA     | SpeedUpX |
| :---:   | :---:   | :---:    | :---:    | :---:    |
| 100     | 100     | 2.128 ms | 1.465 ms | 1.45X    |
| 1000    | 1000    | 24.12 ms | 7.449 ms | 3.24X    |
| 10000   | 10000   | 241.3 ms | 28.07 ms | 8.60X    |
| 30000   | 50      | 102.4 ms | 38.53 ms | 2.65X    |

#### Install

+ from source

<!-- ```bash
git clone https://github.com/unlimblue/KNN_CUDA.git
cd KNN_CUDA
make && make install
``` -->
<!-- 
+ for windows

You should use branch `windows`:

```bash
git clone --branch windows https://github.com/unlimblue/KNN_CUDA.git
cd C:\\PATH_TO_KNN_CUDA
make
make install
``` -->

#### Usage

```python
import torch

# Cuda needs to be available
assert torch.cuda.is_available()

from chamferCuda import CD
"""
if transpose_mode is True, 
    pc1 is Tensor [bs x nr x dim]
    pc2 is Tensor [bs x nq x dim]
    
    return 
        cd_dist is Tensor [bs]
else
    pc1 is Tensor [bs x dim x nr]
    pc2 is Tensor [bs x dim x nq]
    
    return 
        cd_dist is Tensor [bs]
"""

cd = CD(transpose_mode=True)

pc1 = torch.rand(32, 1000, 5).cuda()
pc2 = torch.rand(32, 50, 5).cuda()

cd_dist = cd(pc1, pc2)  # 32
```