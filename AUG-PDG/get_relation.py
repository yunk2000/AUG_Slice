# -- coding: utf-8 --
import re
from py2neo.packages.httpstream import http
http.socket_timeout = 9999


def getLoopRealtionOfCFG(cfg):
    list_loop_nodes = []

    for node in cfg.vs:
        if node['type'] == 'Condition':
            filepath = node['filepath']
            location_row = int(node['location'].split(':')[0])
            with open(filepath, 'r') as fin:
                content = fin.readlines()
            src_code = content[location_row - 1]

            pattern = re.compile("(?:for|while)")
            result = re.search(pattern, src_code)

            if result:
                list_loop_nodes.append(node)

    _dict = {}

    for loop_node in list_loop_nodes:
        list_body_nodes = []

        for es in cfg.es:
            if cfg.vs[es.tuple[0]] == loop_node and es['var'] == 'True':
                start_node = cfg.vs[es.tuple[1]]
                not_scan_list = [loop_node['name']]

                list_body_nodes, temp = getSubCFGGraph(start_node, list_body_nodes, not_scan_list)

        _dict[loop_node['name']] = [body_node['name'] for body_node in list_body_nodes]

    return _dict


def getSwitchRelationOfCFG(cfg):
    list_switch_nodes = []

    for node in cfg.vs:
        if node['type'] == 'Condition':
            filepath = node['filepath']
            location_row = int(node['location'].split(':')[0])
            with open(filepath, 'r') as fin:
                content = fin.readlines()
            src_code = content[location_row - 1]

            pattern = re.compile("switch")
            result = re.search(pattern, src_code)

            if result:
                list_switch_nodes.append(node)

    _dict = {}

    for switch_node in list_switch_nodes:
        list_true_case_nodes = []
        list_false_case_nodes = []

        for es in cfg.es:
            if cfg.vs[es.tuple[0]] == switch_node and es['var'] == 'True':
                start_node = cfg.vs[es.tuple[1]]
                not_scan_list = [switch_node['name']]

                list_true_case_nodes, temp = getSubCFGGraph(start_node, list_true_case_nodes, not_scan_list)

            elif cfg.vs[es.tuple[0]] == switch_node and es['var'] == 'False':
                start_node = cfg.vs[es.tuple[1]]
                not_scan_list = [switch_node['name']]

                list_false_case_nodes, temp = getSubCFGGraph(start_node, list_false_case_nodes, not_scan_list)

        _share_list = []
        for t_node in list_true_case_nodes:
            if t_node in list_false_case_nodes:
                _share_list.append(t_node)

        if _share_list:
            list_true_case_nodes = [t_node for t_node in list_true_case_nodes if t_node not in _share_list]
            list_false_case_nodes = [f_node for f_node in list_false_case_nodes if f_node not in _share_list]

        _dict[switch_node['name']] = (
        [t_node['name'] for t_node in list_true_case_nodes], [f_node['name'] for f_node in list_false_case_nodes])

    return _dict


def main():
    j = JoernSteps()
    j.connectToDatabase()
    all_func_node = getALLFuncNode(j)
    
    for node in all_func_node:
        testID = getFuncFile(j, node._id).split('/')[-4]
        path = os.path.join("cfg_aug", testID)

        store_file_name = node.properties['name'] + '_' + str(node._id)
        store_path = os.path.join(path, store_file_name)
        fin = open(os.path.join(path, 'cfg'))
        cfg = pickle.load(fin)
        fin.close()

        _dictLoop = getLoopRealtionOfCFG(cfg)
        _dictLoop_node2loop = {}
        for key in _dictLoop.keys():
            _list = _dictLoop[key]
            for v in _list:
                if v not in _dictLoop_node2loop.keys():
                    _dictLoop_node2loop[v] = [key]
                else:
                    _dictLoop_node2loop[v].append(key)
        for key in _dictLoop_node2loop.keys():
            _dictLoop_node2loop[key] = list(set(_dictLoop_node2loop[key]))

        _dictSwitch = getSwitchRelationOfCFG(cfg)
        _dictLoop_node2switch = {}
        for key in _dictSwitch.keys():
            _list = _dictSwitch[key][0] + _dictSwitch[key][1]
            for v in _list:
                if v not in _dictLoop_node2switch.keys():
                    _dictLoop_node2switch[v] = [key]
                else:
                    _dictLoop_node2switch[v].append(key)
        for key in _dictLoop_node2switch.keys():
            _dictLoop_node2switch[key] = list(set(_dictLoop_node2switch[key]))

        if not os.path.exists(path):
            os.mkdir(path)
        if not os.path.exists(store_path):
            os.mkdir(store_path)
        else:
            continue

        filename = 'dict_loop2cfgnode'
        dict_store_path_3 = os.path.join(store_path, filename)
        fout = open(dict_store_path_3, 'wb')
        pickle.dump(_dictLoop, fout, True)
        fout.close()

        filename = 'dict_cfgnode2loop'
        dict_store_path_4 = os.path.join(store_path, filename)
        fout = open(dict_store_path_4, 'wb')
        pickle.dump(_dictLoop_node2loop, fout, True)
        fout.close()

        filename = 'dict_switch2cfgnode'
        dict_store_path_5 = os.path.join(store_path, filename)
        fout = open(dict_store_path_5, 'wb')
        pickle.dump(_dictSwitch, fout, True)
        fout.close()

        filename = 'dict_cfgnode2switch'
        dict_store_path_6 = os.path.join(store_path, filename)
        fout = open(dict_store_path_6, 'wb')
        pickle.dump(_dictLoop_node2switch, fout, True)
        fout.close()


if __name__ == '__main__':
    main()
