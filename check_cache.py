"""
检查预处理缓存目录中的 .npy 文件是否包含 NaN 或 Inf。
用法: python check_cache.py --dir path/to/cache --workers 16
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def check_single_file(file_path: Path):
    """单文件检查函数，返回 (文件路径, 是否异常, NaN数, Inf数, 形状, 错误信息)"""
    try:
        data = np.load(file_path, allow_pickle=False)
        if not isinstance(data, np.ndarray):
            return file_path, True, 0, 0, getattr(data, "shape", "N/A"), "非 NumPy 数组"
        
        # 仅对数值类型进行 NaN/Inf 检查
        if not np.issubdtype(data.dtype, np.number):
            return file_path, False, 0, 0, data.shape, None
            
        n_nan = int(np.sum(np.isnan(data)))
        n_inf = int(np.sum(np.isinf(data)))
        return file_path, (n_nan > 0 or n_inf > 0), n_nan, n_inf, data.shape, None
    except Exception as e:
        return file_path, True, 0, 0, "N/A", str(e)

def main():
    parser = argparse.ArgumentParser(description="扫描缓存目录，检测 NaN/Inf 异常值")
    parser.add_argument("--dir", type=str, required=True, help="包含 .npy 文件的目录路径")
    parser.add_argument("--workers", type=int, default=16, help="并行工作进程数 (默认 16)")
    parser.add_argument("--delete-bad", action="store_true", help="发现异常时直接删除问题文件")
    args = parser.parse_args()

    cache_dir = Path(args.dir)
    if not cache_dir.is_dir():
        print(f"❌ 目录不存在: {cache_dir}")
        sys.exit(1)

    npy_files = sorted(cache_dir.rglob("*.npy"))
    if not npy_files:
        print(f"⚠️  未在 {cache_dir} 中找到任何 .npy 文件")
        return

    print(f"🔍 开始扫描 {len(npy_files)} 个文件 (使用 {args.workers} 进程)...")
    
    bad_files = []
    total_nan = 0
    total_inf = 0
    total_checked = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_file = {executor.submit(check_single_file, f): f for f in npy_files}
        for future in tqdm(as_completed(future_to_file), total=len(npy_files), desc="进度"):
            fpath, is_bad, n_nan, n_inf, shape, err = future.result()
            total_nan += n_nan
            total_inf += n_inf
            total_checked += 1
            
            if is_bad:
                bad_files.append((fpath, shape, n_nan, n_inf, err))
                if args.delete_bad:
                    try:
                        fpath.unlink()
                    except Exception:
                        pass

    # 📊 打印报告
    print("\n" + "="*60)
    print("📈 检测报告")
    print("="*60)
    print(f"📁 扫描目录      : {cache_dir}")
    print(f"✅ 成功检查文件数 : {total_checked}")
    print(f"❌ 异常/损坏文件数 : {len(bad_files)}")
    print(f"🔢 累计 NaN 数量  : {total_nan}")
    print(f"🔢 累计 Inf 数量  : {total_inf}")
    
    if bad_files:
        print("\n🚨 异常文件明细 (前 20 条):")
        for i, (f, shape, n_nan, n_inf, err) in enumerate(bad_files[:20]):
            status = f"NaN={n_nan}, Inf={n_inf}"
            if err: status += f" | 错误: {err}"
            print(f"  [{i+1}] {f.name:<30} | Shape: {str(shape):<15} | {status}")
        
        if len(bad_files) > 20:
            print(f"  ... 还有 {len(bad_files)-20} 个文件未显示 ...")
            
        if args.delete_bad:
            print(f"\n🗑️  已自动删除 {len(bad_files)} 个异常文件。请重新运行预处理脚本补全。")
        else:
            print(f"\n💡 建议: 删除异常文件后重新运行预处理:")
            print(f"   rm {bad_files[0][0].parent}/*.npy")
            print(f"   python preprocess.py --mode f0 --workers {args.workers}")
    else:
        print("\n🎉 恭喜！所有缓存文件均干净无异常。")
    print("="*60)

if __name__ == "__main__":
    main()