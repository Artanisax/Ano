import numpy as np
import glob, os
from tqdm import tqdm
f0_dir = '/home/wangzq_lab/cse12110524/Projects/NPU-NTU_System_for_Voice_Privacy_2024_Challenge_Implementation/data/processed/f0_abs'  # 替换为实际路径
bad = 0
for f in tqdm(glob.glob(os.path.join(f0_dir, '*.npy'))):
    arr = np.load(f)
    if np.any(arr == 0) or np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        print(f, np.any(arr == 0), np.any(np.isnan(arr)), np.any(np.isinf(arr)))
        bad += 1
print(bad)