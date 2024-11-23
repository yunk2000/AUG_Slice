# -*- coding:utf-8 -*-
import pickle
import re
import os
import json


def make_label(data_path, label_path, _dict):
    for filename in os.listdir(data_path):
        filepath = os.path.join(data_path, filename)
        _labels = {}
        f = open(filepath, 'r')
        slicelists = f.read().split('------------------------------')
        f.close()

        labelpath = os.path.join(label_path, filename[:-4] + '_label.pkl')

        if slicelists[0] == '':
            del slicelists[0]
        if slicelists[-1] == '' or slicelists[-1] == '\n' or slicelists[-1] == '\r\n':
            del slicelists[-1]

        for slice in slicelists:
            sentences = slice.split('\n')
            if sentences[0] == '\r' or sentences[0] == '':
                del sentences[0]
            if sentences == []:
                continue
            if sentences[-1] == '':
                del sentences[-1]
            if sentences[-1] == '\r':
                del sentences[-1]

            slicename = sentences[0]
            label = 0
            key = '/public/home/aclh122csn/yunk/datasets/NVD/' + ('/').join(
                slicename.split(' ')[1].split('/')[-2:])  # key in label_source
            print("*******************************************", key)
            if key not in _dict.keys():
                _labels[slicename] = 0
                continue
            if len(_dict[key]) == 0:
                _labels[slicename] = 0
                continue
            sentences = sentences[1:]
            for sentence in sentences:
                if (is_number(sentence.split(' ')[-1])) is False:
                    continue
                linenum = int(sentence.split(' ')[-1])
                vullines = _dict[key]
                if linenum in vullines:
                    label = 1
                    _labels[slicename] = 1
                    break
            if label == 0:
                _labels[slicename] = 0

        with open(labelpath, 'wb') as f1:
            pickle.dump(_labels, f1)
        f1.close()


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


if __name__ == '__main__':
    with open('../data_sets/NVD_diff/nvd_label.json', 'rb') as f:
        data = json.load(f)
    f.close()

    _dict = {}

    for item in data:
        file_name = item.get("file_name")
        line = item.get("line")
        numbers_list = [int(num.strip()) for num in line.split(",")]
        if file_name not in _dict.keys():
            _dict[file_name] = numbers_list
        else:
            _dict[file_name].append(numbers_list)

    code_path = './C_new_3/test_data/'  # slice code of software
    label_path = './C_new_3/label_data/'  # labels

    make_label(code_path, label_path, _dict)
