
from ChineseTextEDA.eda import EDA

eda = EDA()

# 输入和输出文件路径
input_file = "data.txt"
output_file = "output_aug.txt"

# 打开输入文件并读取所有内容
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# 使用句号分隔句子并存储到列表中
sentences = text.split("。")

# 对每个句子执行数据增强并将结果写入输出文件
with open(output_file, "w", encoding="utf-8") as f:
    for sentence in sentences:
        # 过滤掉空白和非法字符
        if len(sentence.strip()) == 0:
            continue

        # 执行数据增强
        augmented_sentences = eda.eda(sentence)

        # 写入输出文件
        for aug_sentence in augmented_sentences:
            f.write(aug_sentence + "。")

