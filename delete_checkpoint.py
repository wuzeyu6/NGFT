import os
import shutil
import argparse
from pathlib import Path

def delete_checkpoint_dirs(root_dir: str, dry_run: bool = True, ignore_case: bool = True) -> None:
    """
    删除指定根目录下名称包含 "checkpoint" 的子文件夹
    
    Args:
        root_dir: 要扫描的根目录（绝对路径/相对路径）
        dry_run: 仅预览要删除的目录，不执行实际删除（默认开启，安全第一）
        ignore_case: 是否忽略大小写匹配（如匹配 CheckPoint、CHECKPOINT 等）
    """
    # 校验根目录是否存在
    root_path = Path(root_dir).resolve()
    if not root_path.exists():
        print(f"❌ 错误：根目录不存在 -> {root_path}")
        return
    
    if not root_path.is_dir():
        print(f"❌ 错误：指定路径不是文件夹 -> {root_path}")
        return

    # 遍历根目录下的所有子项
    deleted_count = 0
    for item in root_path.iterdir():
        # 仅处理文件夹，且名称包含 checkpoint（忽略大小写）
        if item.is_dir():
            dir_name = item.name.lower() if ignore_case else item.name
            if "checkpoint" in dir_name:
                print(f"🔍 找到 checkpoint 文件夹：{item}")
                if not dry_run:
                    try:
                        # 递归删除文件夹（强制删除所有内容）
                        shutil.rmtree(item)
                        print(f"✅ 成功删除：{item}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"❌ 删除失败：{item} | 错误信息：{str(e)}")
    
    # 输出总结
    if dry_run:
        print("\n📌 【预览模式】未执行实际删除，以上是待删除的 checkpoint 文件夹")
    else:
        print(f"\n📊 清理完成：共找到并删除 {deleted_count} 个 checkpoint 文件夹")

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="删除指定文件夹下包含 'checkpoint' 的子文件夹")
    parser.add_argument(
        "--root_dir", 
        required=True, 
        help="要扫描的根目录（如：/mnt/shared-storage-user/xxx/model_result/）"
    )
    parser.add_argument(
        "--no-dry-run", 
        action="store_false", 
        dest="dry_run", 
        help="关闭预览模式，执行实际删除（谨慎使用！）"
    )
    parser.add_argument(
        "--case-sensitive", 
        action="store_false", 
        dest="ignore_case", 
        help="开启大小写敏感匹配（默认忽略大小写）"
    )
    
    args = parser.parse_args()
    
    # 执行删除逻辑
    delete_checkpoint_dirs(
        root_dir=args.root_dir,
        dry_run=args.dry_run,
        ignore_case=args.ignore_case
    )