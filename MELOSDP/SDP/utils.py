# 自定义写入ARFF文件函数
def save_arff(file_path, relation, attributes, data):
    with open(file_path, 'w') as f:
        f.write(f"@relation {relation}\n\n")
        for attr_name, attr_type in attributes:
            if isinstance(attr_type, list):
                f.write(f"@attribute {attr_name} {{{','.join(attr_type)}}}\n")
            else:
                f.write(f"@attribute {attr_name} {attr_type}\n")
        f.write("\n@data\n")
        for row in data:
            f.write(",".join(str(value) for value in row) + "\n")