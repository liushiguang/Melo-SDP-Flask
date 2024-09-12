import requests
from flask import Flask, request, jsonify
from twisted.python.util import println
from MELOSDP import app
from MELOSDP.util import APIResponse
import json
from llms.check_code import check_code
from llms.repair_code import fetch_code_suggestions
from SDP.machine_learning import naive_bayes_classifier,random_forest,supporting_vector_machine
from SDP.kmeans import kmeans_classifier
from SDP.inference_conv import inference_conv
from SDP.inference_linearn import inference_linear

# 测试方法
@app.route('/test', methods=['GET'])
def delete_book():
    response = APIResponse(200, "data", '测试')
    response_dict = response.__dict__
    print(response_dict)  # 打印调试信息
    return jsonify(response_dict)


# 代码纠错
@app.route('/codeTest', methods=['POST'])
def code_test():
    file = request.files.get('file')
    file_extension = request.form.get('fileExtension')
    if file:
        file_content = file.read().decode('utf-8')
        # print(file_content)  # 打印文件内容用于调试
        # print(file_extension)
        code = check_code(file_extension, file_content)
        println("指出错误的代码——————————————————")
        print(code)

        response = {
            'code': 200,
            'message': 'File received',
            'data': code
        }
        return jsonify(response)
    else:
        return jsonify({'code': 400, 'message': 'No file uploaded'})


# 代码修改意见
@app.route('/suggestionTest', methods=['POST'])
def suggestion_test():
    modified_code = request.form.get('modifiedCode')
    suggestions = fetch_code_suggestions(modified_code)
    println("修改意见—————————————————")
    print(suggestions)
    response = {
        'code': 200,
        'message': 'File received',
        'data': suggestions
    }
    return jsonify(response)


# 模型测试数据
@app.route('/dataTest', methods=['POST'])
def data_test():
    # 获取文件
    # file = request.files.get('file')

    # 获取其他表单数据
    dl_models = request.form.get('dlModels')
    ml_models = request.form.get('mlModels')
    metrics = request.form.get('metrics')
    test_type = request.form.get('type')

    # 解析 JSON 数据
    dl_models = json.loads(dl_models)
    ml_models = json.loads(ml_models)
    metrics = json.loads(metrics)

    # file_content = file.read().decode('utf-8')

    # print(file_content)
    print(ml_models)
    print(dl_models)
    print(metrics)
    print(test_type)

    # 生成结果数据，每个模型对应一个数据数组
    data = []

    for model in dl_models:
        if model == "FCFNN":
            data.append({'type': model, 'data': inference_conv()})
        elif model == "CNN":
            data.append({'type': model, 'data': inference_linear()})
        else:
            data.append({'type': model, 'data': [0, 0, 0, 0, 0]})

    for model in ml_models:
        if model == "SVM":
            data.append({'type': model, 'data': supporting_vector_machine()})
        elif model == "RF":
            data.append({'type': model, 'data': random_forest()})
        elif model == "NBC":
            data.append({'type': model, 'data': naive_bayes_classifier()})
        elif model == "KM":
            data.append({'type': model, 'data': kmeans_classifier()})
        else:
            data.append({'type': model, 'data': [0, 0, 0, 0, 0]})

    response_data = {
        'code': 400,
        "message": "Data received successfully",
        'data': data
    }
    return jsonify(response_data)


if __name__ == '__main__':
    app.run()
