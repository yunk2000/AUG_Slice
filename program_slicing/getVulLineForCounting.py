# -*- coding:utf-8 -*-

from optparse import OptionParser
import re
import os
import pickle

def autoGetVulLine(rawPath, fileName):
    with open(fileName) as f:
        flag = 0
        path = ''
        txtstring = ''
        vulLineFile = open('vul_context_SARD.pkl', 'wb')
        txtfile = open('vul_context_SARD.txt', 'w')

        flawLineList = set([])
        file2FlawLineDict = {}

        for line in f.readlines():
            filePath = re.findall('<file path=\"(.+)\" language=\"', line)
            # print("filePath"+str(filePath))

            if not filePath:
                fileEndFlag = re.findall('</file>', line)
                flawLine = re.findall('<flaw line=\"(\d+)\" name=\"', line)
                mixLine = re.findall('<mixed line=\"(\d+)\" name=\"', line)

                if flag == 1 and flawLine:
                    flawLineList.add(flawLine[0])
                if flag == 1 and mixLine:
                    flawLineList.add(mixLine[0])

                if flag == 1 and fileEndFlag:
                    flag = 0
                    if flawLineList and os.path.exists(os.path.join(rawPath, path)) and path.find('shared') == -1:
                        file2FlawLineDict[os.path.join(rawPath, path)] = set(flawLineList)

                    path = ''
                    flawLineList.clear()
                continue
            else:
                path = filePath[0]
                flag = 1

        pickle.dump(file2FlawLineDict, vulLineFile)

        for key in file2FlawLineDict.keys():
            for line in file2FlawLineDict[key]:
                txtstring = txtstring + key +' '+line + '\n'
        txtfile.write(txtstring)


if __name__ == '__main__':

    rawPath = '/public/home/aclh122csn/yunk/datasets/SARD/SARD'
    fileName = '/public/home/aclh122csn/yunk/datasets/SARD/SARD_testcaseinfo.xml'
    autoGetVulLine(rawPath, fileName)
