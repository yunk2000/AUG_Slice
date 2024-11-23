## coding:utf-8
import pickle
import os
import shutil
import re

def dealfunc_nvd(folder_path,diff_path):
    vulline_dict = {}
    for filename in os.listdir(folder_path):
        if 'PATCHED' in filename:
            continue
        pattern = re.compile(r"(?P<cve_id>CVE[-_][0-9]*[-_][0-9]*)[-_]")
        match = re.search(pattern, filename)
        cve_id = "-".join(match.group("cve_id").split("_"))
        filepath = os.path.join(folder_path,filename)
        f = open(filepath,'r')
        sentences = f.read().split('\n')
        f.close()
        diffpath = os.path.join(diff_path,cve_id,(cve_id + '.txt'))
        f = open(diffpath,'r')
        diffsens = f.read().split('\n')

        f.close()
            
        vul_code = []
        index = -1
        index_start= []
        for sen in diffsens:
            index += 1
            if sen.startswith('@@ ') is True:
                index_start.append(index)
        for i in range(0,len(index_start)):
            if i < len(index_start)-1: 
                diff_sens = diffsens[index_start[i]:index_start[i+1]]
            else:
                diff_sens = diffsens[index_start[i]:]
            startline = diff_sens[0]
            diff_sens = diff_sens[1:]
            index = -1
            for sen in diff_sens:
                index += 1
                if sen.startswith('-') is True and sen.startswith('---') is False:
                    if sen.strip('-').strip() == '' or sen.strip('-').strip()==',' or sen.strip('-').strip() == ';' \
                            or sen.strip('-').strip() == '{' or sen.strip('-').strip() == '}':
                        continue
                    vul_code.append(sen.strip('-').strip())

        for i in range(0,len(sentences)):
            if sentences[i].strip() not in vul_code:
                continue
            else:
                linenum = i + 1
                if filepath not in vulline_dict.keys():
                    vulline_dict[filepath] = [linenum]
                else:
                    vulline_dict[filepath].append(linenum)
    with open('./vul_context_func.pkl','wb') as f:
        pickle.dump(vulline_dict,f)
    f.close()

                                
if __name__ == "__main__":
    data_source1 = '/public/home/aclh122csn/yunk/datasets/NVD/NVD'
    diff_path = '/public/home/aclh122csn/yunk/datasets/NVD/NVD_diff'
    dealfunc_nvd(data_source1, diff_path)

