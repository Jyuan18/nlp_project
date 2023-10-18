import pandas as pd
from config import *


# 生成标签表
def generate_label():
    df = pd.read_csv(TRAIN_SAMPLE_PATH, delimiter=' ', usecols=[1], names=['label'])
    label_list = df['label'].value_counts().keys().tolist()
    # seqeval 无法识别下划线_
    label_dict = {v.replace('_', '-'): k for k, v in enumerate(label_list)}
    label = pd.DataFrame(list(label_dict.items()))
    # 保存为 CSV 文件，文件路径由 LABEL_PATH 变量指定，header=None 表示不包含列名，index=None 表示不包含行索引
    label.to_csv(LABEL_PATH, header=None, index=None)


if __name__ == '__main__':
    # 生成标签表
    generate_label()
