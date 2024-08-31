import pandas as pd

# 读取 Excel 文件
df = pd.read_excel("测试结果记录2024.04.08.xlsx")

# 对 DataFrame 中的所有数字保留三位小数
def round_numbers(x):
    if isinstance(x, (int, float)):
        return round(x, 3)
    else:
        return x

df_rounded = df.applymap(round_numbers)

# 将处理后的 DataFrame 写入 Excel 文件
df_rounded.to_excel("rounded_file.xlsx", index=False)
