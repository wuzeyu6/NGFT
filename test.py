# filename: reorder_json.py

import json
import argparse
import sys


def sort_from_middle(sorted_by_activation: list) -> list:
    """
    将一个已排序的列表重新排列，使得中间的元素排在最前面，然后依次向两边扩展。

    Args:
        sorted_by_activation (list): 一个已经按某个标准（如激活值）从大到小排序的列表。

    Returns:
        list: 一个重新排序后的新列表。
    """
    if not sorted_by_activation:
        return []

    n = len(sorted_by_activation)
    reordered_list = []

    # 初始化左右指针，使其指向列表的中心
    # 使用 (n - 1) // 2 和 n // 2 可以优雅地处理奇数和偶数长度
    left_ptr = (n - 1) // 2
    right_ptr = n // 2

    # 当新列表的长度还未达到原列表长度时，持续添加元素
    while len(reordered_list) < n:
        # 如果 left 和 right 指针在同一个位置 (奇数长度列表的初始情况)
        if left_ptr == right_ptr:
            reordered_list.append(sorted_by_activation[left_ptr])
        else:
            # 对于偶数长度的列表，或者在后续迭代中，我们成对地添加元素
            # 先添加左边的（激活值稍高一些的中间值）
            if left_ptr >= 0:
                reordered_list.append(sorted_by_activation[left_ptr])
            # 再添加右边的（激活值稍低一些的中间值）
            if right_ptr < n:
                reordered_list.append(sorted_by_activation[right_ptr])

        # 移动指针，向两边扩展
        left_ptr -= 1
        right_ptr += 1

    return reordered_list


def main():
    """
    主函数，负责处理文件读写和调用排序逻辑。
    """
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(
        description="从一个已排序的JSON文件中读取数据，将其从中间向两边重新排序，并写入新的JSON文件。"
    )
    parser.add_argument("input_file", help="输入的JSON文件路径 (数据需已按激活值从大到小排序)")
    parser.add_argument("output_file", help="输出的JSON文件路径")

    args = parser.parse_args()

    # 2. 读取并解析输入的JSON文件
    try:
        print(f"正在从 '{args.input_file}' 读取数据...")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 输入文件 '{args.input_file}' 未找到。", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误: 文件 '{args.input_file}' 不是有效的JSON格式。", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"读取文件时发生未知错误: {e}", file=sys.stderr)
        sys.exit(1)

    # 确保JSON文件的顶层结构是一个列表
    if not isinstance(original_data, list):
        print(f"错误: JSON文件的顶层数据结构必须是一个列表 (list)，但当前是 {type(original_data).__name__}。",
              file=sys.stderr)
        sys.exit(1)

    print(f"成功加载 {len(original_data)} 条数据。")

    # 3. 调用核心排序函数
    print("正在对数据进行从中间向两边的重新排序...")
    reordered_data = sort_from_middle(original_data)
    print("排序完成。")

    # 4. 将重新排序后的数据写入输出文件
    try:
        print(f"正在将结果写入 '{args.output_file}'...")
        with open(args.output_file, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 确保中文字符不会被转义为 \uXXXX
            # indent=4 使输出的JSON文件格式化，易于阅读
            json.dump(reordered_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"写入文件时发生错误: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n处理完成！重新排序的数据已成功保存到 '{args.output_file}'。")


if __name__ == "__main__":
    main()