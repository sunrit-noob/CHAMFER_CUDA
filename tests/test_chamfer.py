import torch
import numpy as np
from sklearn.neighbors import KDTree
from chamferCuda import CD
import time

def t2n(t):
    return t.detach().cpu().numpy()

def run_kdtree(pc1, pc2, scale=1, offset=0):
    pc1 = pc1 / scale - offset
    B = pc1.shape[0]

    chamfer = [0.0 for _ in range(B)]
    for b in range(B):
        pc2_kd_tree = KDTree(pc2[b])
        one_distances, _ = pc2_kd_tree.query(pc1[b])
        pc1_to_pc2_chamfer = np.mean(np.square(one_distances))

        pc1_kd_tree = KDTree(pc1[b])
        two_distances, _ = pc1_kd_tree.query(pc2[b])
        pc2_to_pc1_chamfer = np.mean(np.square(two_distances))

        chamfer[b] = pc1_to_pc2_chamfer + pc2_to_pc1_chamfer
    
    return np.array(chamfer)

def run_ChamferCuda(pc1, pc2):
    pc1 = torch.from_numpy(pc1).float().cuda()
    pc2 = torch.from_numpy(pc2).float().cuda()
    chamferDist = CD(transpose_mode=True)
    d  = chamferDist(pc1, pc2)
    return t2n(d)

def compare(dim, n1, n2=-1):
    if n2 < 0:
        n2 = n1

    kd_test_times = []
    cuda_test_times = []
    for _ in range(5):
        pc1 = np.random.random((2, n1, dim))
        pc2 = np.random.random((2, n2, dim))

        t0 = time.perf_counter()
        chamfer_dist = run_kdtree(pc1, pc2)
        t1 = time.perf_counter()
        kd_test_times.append(t1 - t0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cuda_dist = run_ChamferCuda(pc1, pc2)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        cuda_test_times.append(t1 - t0)

        np.testing.assert_allclose(chamfer_dist, cuda_dist, rtol=5e-2, atol=5e-3)

    kd_test_times = np.array(kd_test_times)
    cuda_test_times = np.array(cuda_test_times)
    print("KDTree times:", np.mean(kd_test_times))
    print("CUDA times:", np.mean(cuda_test_times))


class TestChamferCuda:

    def test_chamfer_cuda_performance(self, benchmark):
        dim = 5
        pc1 = np.random.random((1, 224, dim))
        pc2 = np.random.random((1, 224, dim))
        benchmark(run_ChamferCuda, pc1, pc2)

    def test_chamfer_cuda_5_1000(self):
        compare(5, 1000)

    def test_chamfer_cuda_5_100(self):
        compare(5, 100)

    def test_chamfer_cuda_5_10(self):
        compare(5, 10)

    def test_chamfer_cuda_5_1001(self):
        compare(5, 1001)

    def test_chamfer_cuda_5_101(self):
        compare(5, 101)

    def test_chamfer_cuda_5_11(self):
        compare(5, 11)

    def test_chamfer_cuda_5_300000_50(self):
        compare(5, 30000, 50)

    def test_chamfer_cuda_5_300001_50(self):
        compare(5, 30001, 50)

    def test_chamfer_cuda_5_10000(self):
        compare(5, 10000)

    def test_chamfer_cuda_5_10001(self):
        compare(5, 10001)