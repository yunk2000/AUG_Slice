## coding:utf-8
import os
import pickle


def make_label(path, _dict):
    print(path)
    f = open(path, 'r')
    slicelists = f.read().split('------------------------------')[:-1]
    f.close()

    labels = {}
    if slicelists[0] == '':
        del slicelists[0]
    if slicelists[-1] == '' or slicelists[-1] == '\n' or slicelists[-1] == '\r\n':
        del slicelists[-1]
    
    for slicelist in slicelists:
        sentences = slicelist.split('\n')
            
        if sentences[0] == '\r' or sentences[0] == '':
            del sentences[0]
        if sentences == []:
            continue
        if sentences[-1] == '':
            del sentences[-1]
        if sentences[-1] == '\r':
            del sentences[-1]

        slicename = sentences[0].split(' ')[1].split('/')[-3] + '/' + sentences[0].split(' ')[1].split('/')[-2] + '/' + sentences[0].split(' ')[1].split('/')[-1]
        sens = sentences[1:]
        
        label = 0

        if slicename not in _dict.keys():
            labels[sentences[0]] = label
            continue
        else:
            vulline_nums = _dict[slicename]
            for sentence in sens:
                if (is_number(sentence.split(' ')[-1])) is False:
                    continue
                linenum = int(sentence.split(' ')[-1])
                if linenum not in vulline_nums:
                    continue
                else:
                    label = 1
                    break
        labels[sentences[0]] = label
    
    return labels


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


def main():
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

    code_path = './C/test_data/'
    label_path = './C/label_data/'
    
    path = os.path.join(code_path, 'api_slices.txt')
    list_all_apilabel = make_label(path, _dict)
    dec_path = os.path.join(label_path, 'api_slices_label.pkl')
    f = open(dec_path, 'wb')
    pickle.dump(list_all_apilabel, f, True)
    f.close()
    
    path = os.path.join(code_path, 'arraysuse_slices.txt')
    list_all_arraylabel = make_label(path, _dict)
    dec_path = os.path.join(label_path, 'arraysuse_slices_label.pkl')
    f = open(dec_path, 'wb')
    pickle.dump(list_all_arraylabel, f, True)
    f.close()
    
    path = os.path.join(code_path, 'pointersuse_slices.txt')
    list_all_pointerlabel = make_label(path, _dict)
    dec_path = os.path.join(label_path, 'pointersuse_slices_label.pkl')
    f = open(dec_path, 'wb')
    pickle.dump(list_all_pointerlabel, f, True)
    f.close()
 
    path = os.path.join(code_path, 'integeroverflow_slices.txt')
    list_all_exprlabel = make_label(path, _dict)
    dec_path = os.path.join(label_path, 'integeroverflow_slices_label.pkl')
    f = open(dec_path, 'wb')
    pickle.dump(list_all_exprlabel, f, True)
    f.close()
    

if __name__ == '__main__':
    main()
