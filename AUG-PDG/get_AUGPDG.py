## coding:utf-8
from access_db_operate import *
import copy
from general_op import *
from py2neo.packages.httpstream import http
http.socket_timeout = 9999


def handleIfDependencyDecl(pdg, startnode, endnode, var, dict_if2cfgnode, dict_cfgnode2if):
    list_if = dict_cfgnode2if[startnode['name']]
    list_not_scan = []

    for ifstmt_n in list_if:
        tuple_statements = dict_if2cfgnode[ifstmt_n]
        if startnode['name'] in tuple_statements[0]:
            list_not_scan += tuple_statements[1]
        elif startnode['name'] in tuple_statements[1]:
            list_not_scan += tuple_statements[0]

    if endnode['name'] not in list_not_scan:
        pdg = addDataEdge(pdg, startnode['name'], endnode['name'], var)

    return pdg


def handleLoopDependencyDecl(pdg, startnode, endnode, var, dict_loop2cfgnode, dict_cfgnode2loop):
    loop_list = dict_cfgnode2loop[startnode['name']]
    _not_scan = []
    for loop in loop_list:
        _tuple = dict_loop2cfgnode[loop]
        if startnode['name'] in _tuple:
            _not_scan += _tuple

    if endnode['name'] not in _not_scan:
        pdg = addDataEdge(pdg, startnode['name'], endnode['name'], var)

    return pdg


def handleSwitchDependencyDecl(pdg, startnode, endnode, var, dict_switch2cfgnode, dict_cfgnode2switch):
    switch_list = dict_cfgnode2switch[startnode['name']]
    _not_scan = []
    for switch in switch_list:
        _tuple = dict_switch2cfgnode[switch]
        if startnode['name'] in _tuple:
            _not_scan += _tuple

    if endnode['name'] not in _not_scan:
        pdg = addDataEdge(pdg, startnode['name'], endnode['name'], var)

    return pdg


def completeDeclStmtOfPDG(pdg, dict_use, dict_def, dict_if2cfgnode, dict_cfgnode2if, dict_loop2cfgnode,
                          dict_cfgnode2loop, dict_switch2cfgnode, dict_cfgnode2switch):
    list_sorted_pdgnode = sortedNodesByLoc(pdg.vs)
    dict_declnode2val = {}

    for node in pdg.vs:
        if (node['type'] == 'IdentifierDeclStatement' or node['type'] == 'Parameter' or node['type'] == 'Statement') and \
                node['code'].find(' = ') == -1:
            if node['type'] == 'IdentifierDeclStatement' or node['type'] == 'Parameter':
                list_var = dict_def[node['name']]
            else:
                list_var = getVarOfNode(node['code'])

            if list_var == False:
                continue
            else:
                for var in list_var:
                    results = getInitNodeOfDecl(pdg, list_sorted_pdgnode, node, var, dict_use, dict_def)
                    if results != []:
                        for result in results:
                            if node['name'] not in dict_cfgnode2if.keys() and node[
                                'name'] not in dict_cfgnode2loop.keys() and node[
                                'name'] not in dict_cfgnode2switch.keys():
                                startnode = node['name']
                                endnode = result[0]['name']
                                pdg = addDataEdge(pdg, startnode, endnode, var)
                            else:
                                if node['name'] in dict_cfgnode2if.keys():
                                    pdg = handleIfDependencyDecl(pdg, node, result[0], var, dict_if2cfgnode, dict_cfgnode2if)

                                if node['name'] in dict_cfgnode2loop.keys():
                                    pdg = handleLoopDependencyDecl(pdg, node, result[0], var, dict_loop2cfgnode,
                                                             dict_cfgnode2loop)

                                if node['name'] in dict_cfgnode2switch.keys():
                                    pdg = handleSwitchDependencyDecl(pdg, node, result[0], var, dict_switch2cfgnode,
                                                               dict_cfgnode2switch)

    return pdg


def get_nodes_before_exit(pdg, dict_if2cfgnode, dict_cfgnode2if):
    _dict = {}
    for key in dict_cfgnode2if.keys():
        results = pdg.vs.select(name=key)
        if len(results) != 0 and (results[0]['type'] == 'BreakStatement' or results[0]['type'] == 'ReturnStatement' or results[0]['code'].find('exit ') != -1 or results[0]['type'] == 'GotoStatement'):# if stms have return
            if_name = ''
            if len(dict_cfgnode2if[key]) == 1:
                if_name = dict_cfgnode2if[key][0]
            else:
                if_name = get_ifname(key, dict_if2cfgnode, dict_cfgnode2if)


            _list_name_0 = dict_if2cfgnode[if_name][0]
            _list_name_1 = dict_if2cfgnode[if_name][1]

            if key in _list_name_0:
                ret_index = _list_name_0.index(key)
                del _list_name_0[ret_index] #_list_name are set of nodes which under the same if with return node or exit or goto statement

                for name in _list_name_0:
                    _dict[name] = key

            if key in _list_name_1:
                ret_index = _list_name_1.index(key)
                del _list_name_1[ret_index] #_list_name are set of nodes which under the same if with return node or exit or goto statement

                for name in _list_name_1:
                    _dict[name] = key

        else:
            continue

    return _dict


def handleIfDependencyReverse(pdg, startnode, endnode, use_var, dict_if2cfgnode, dict_cfgnode2if):
    list_if = dict_cfgnode2if[startnode['name']]
    list_not_scan = []

    for ifstmt_n in list_if:
        tuple_statements = dict_if2cfgnode[ifstmt_n]
        if startnode['name'] in tuple_statements[0]:
            list_not_scan += tuple_statements[1]
        elif startnode['name'] in tuple_statements[1]:
            list_not_scan += tuple_statements[0]

    if endnode['name'] not in list_not_scan:
        addDataEdge(pdg, startnode['name'], endnode['name'], use_var)

    return pdg


def handleSwitchDependencyReverse(pdg, startnode, endnode, use_var, dict_switch2cfgnode, dict_cfgnode2switch):
    switch_list = dict_cfgnode2switch[startnode['name']]
    _not_scan = []
    for switch in switch_list:
        _tuple = dict_switch2cfgnode[switch]
        if startnode['name'] in _tuple:
            _not_scan += _tuple

    if endnode['name'] not in _not_scan:
        addDataEdge(pdg, startnode['name'], endnode['name'], use_var)

    return pdg


def handleLoopDependencyReverse(pdg, startnode, endnode, use_var, dict_loop2cfgnode, dict_cfgnode2loop):
    loop_list = dict_cfgnode2loop[startnode['name']]
    _not_scan = []
    for loop in loop_list:
        _tuple = dict_loop2cfgnode[loop]
        if startnode['name'] in _tuple:
            _not_scan += _tuple

    if endnode['name'] not in _not_scan:
        addDataEdge(pdg, startnode['name'], endnode['name'], use_var)

    return pdg


def completeReverseDataEdgeOfPDG(pdg, dict_use, dict_def, dict_if2cfgnode, dict_cfgnode2if, dict_loop2cfgnode, dict_cfgnode2loop, dict_switch2cfgnode, dict_cfgnode2switch):
    list_sorted_pdgnode = sortedNodesByLoc(pdg.vs)
    exit2stmt_dict = get_nodes_before_exit(pdg, dict_if2cfgnode, dict_cfgnode2if)

    for i in range(0, len(list_sorted_pdgnode)):
        if list_sorted_pdgnode[i]['name'] in dict_use.keys():
            list_use_var = dict_use[list_sorted_pdgnode[i]['name']]

            for use_var in list_use_var:
                for j in range(i-1, -1, -1):
                    if list_sorted_pdgnode[j]['name'] in exit2stmt_dict.keys():
                        exit_name = exit2stmt_dict[list_sorted_pdgnode[j]['name']]
                        if list_sorted_pdgnode[i]['name'] == exit_name:
                            break

                    elif list_sorted_pdgnode[j]['name'] in dict_def.keys() and use_var in dict_def[list_sorted_pdgnode[j]['name']]:
                        if list_sorted_pdgnode[i]['name'] not in dict_cfgnode2if.keys() and list_sorted_pdgnode[i]['name'] not in dict_cfgnode2loop.keys() \
                                and list_sorted_pdgnode[i]['name'] not in dict_cfgnode2switch.keys():
                            startnode = list_sorted_pdgnode[i]['name']
                            endnode = list_sorted_pdgnode[j]['name']
                            addDataEdge(pdg, startnode, endnode, use_var)
                            break

                        elif list_sorted_pdgnode[i]['name'] in dict_cfgnode2if.keys():
                            pdg = handleIfDependencyReverse(pdg, list_sorted_pdgnode[i], list_sorted_pdgnode[j], use_var, dict_if2cfgnode, dict_cfgnode2if)

                        elif list_sorted_pdgnode[i]['name'] in dict_cfgnode2loop.keys():
                            pdg = handleLoopDependencyReverse(pdg, list_sorted_pdgnode[i], list_sorted_pdgnode[j], use_var, dict_loop2cfgnode, dict_cfgnode2loop)

                        elif list_sorted_pdgnode[i]['name'] in dict_cfgnode2switch.keys():
                            pdg = handleSwitchDependencyReverse(pdg, list_sorted_pdgnode[i], list_sorted_pdgnode[j], use_var, dict_switch2cfgnode, dict_cfgnode2switch)

    return pdg


def completeBranchEdgeOfPDG(pdg, dict_if2cfgnode, dict_cfgnode2if, dict_loop2cfgnode, dict_cfgnode2loop, dict_switch2cfgnode, dict_cfgnode2switch):
    list_sorted_pdgnode = sortedNodesByLoc(pdg.vs)

    for node in list_sorted_pdgnode:
        node_name = node['name']

        if node_name in dict_cfgnode2if.keys():
            list_if_statements = dict_cfgnode2if[node_name]
            for if_stmt in list_if_statements:
                tuple_statements = dict_if2cfgnode[if_stmt]
                for stmt_node in tuple_statements[0] + tuple_statements[1]:
                    addControlEdge(pdg, if_stmt, stmt_node)

        if node_name in dict_cfgnode2loop.keys():
            list_loops = dict_cfgnode2loop[node_name]
            for loop_stmt in list_loops:
                loop_body_nodes = dict_loop2cfgnode[loop_stmt]
                for body_node in loop_body_nodes:
                    addControlEdge(pdg, loop_stmt, body_node)

        if node_name in dict_cfgnode2switch.keys():
            list_switch_statements = dict_cfgnode2switch[node_name]
            for switch_stmt in list_switch_statements:
                switch_body_nodes = dict_switch2cfgnode[switch_stmt]
                for body_node in switch_body_nodes:
                    addControlEdge(pdg, switch_stmt, body_node)

    return pdg


def findScopeNode(node_name, dict_if2cfgnode, dict_cfgnode2if, dict_loop2cfgnode, dict_cfgnode2loop, dict_switch2cfgnode, dict_cfgnode2switch):
    if node_name in dict_cfgnode2if:
        return dict_cfgnode2if[node_name][0]
    elif node_name in dict_cfgnode2loop:
        return dict_cfgnode2loop[node_name][0]
    elif node_name in dict_cfgnode2switch:
        return dict_cfgnode2switch[node_name][0]
    return None


def findFirstDefNode(var, dict_def):
    for def_node, vars_defined in dict_def.items():
        if var in vars_defined:
            return def_node
    return None


def trackVariableLifetime(pdg, dict_use, dict_def, dict_if2cfgnode, dict_cfgnode2if, dict_loop2cfgnode, dict_cfgnode2loop, dict_switch2cfgnode, dict_cfgnode2switch):
    list_sorted_pdgnode = sortedNodesByLoc(pdg.vs)

    for i, node in enumerate(list_sorted_pdgnode):
        node_name = node['name']

        if node_name in dict_use:
            list_used_vars = dict_use[node_name]

            for var in list_used_vars:
                last_use_node = None

                for j in range(i + 1, len(list_sorted_pdgnode)):
                    if var in dict_use.get(list_sorted_pdgnode[j]['name'], []):
                        last_use_node = list_sorted_pdgnode[j]

                if node_name in dict_cfgnode2if or node_name in dict_cfgnode2loop or node_name in dict_cfgnode2switch:
                    if last_use_node:
                        scope_node = findScopeNode(node_name, dict_if2cfgnode, dict_cfgnode2if, dict_loop2cfgnode, dict_cfgnode2loop, dict_switch2cfgnode, dict_cfgnode2switch)
                        pdg = addDataEdge(pdg, last_use_node['name'], scope_node, var)
                else:
                    if last_use_node:
                        first_def_node = findFirstDefNode(var, dict_def)
                        pdg = addDataEdge(pdg, last_use_node['name'], first_def_node, var)

    return pdg


def areNodesInSameControlStructure(startnode, endnode, dict_if2cfgnode, dict_cfgnode2if, dict_loop2cfgnode, dict_cfgnode2loop, dict_switch2cfgnode, dict_cfgnode2switch):
    if startnode in dict_cfgnode2if and endnode in dict_cfgnode2if:
        if set(dict_cfgnode2if[startnode]) & set(dict_cfgnode2if[endnode]):
            return True

    if startnode in dict_cfgnode2loop and endnode in dict_cfgnode2loop:
        if set(dict_cfgnode2loop[startnode]) & set(dict_cfgnode2loop[endnode]):
            return True

    if startnode in dict_cfgnode2switch and endnode in dict_cfgnode2switch:
        if set(dict_cfgnode2switch[startnode]) & set(dict_cfgnode2switch[endnode]):
            return True

    return False


def main():
    j = JoernSteps()
    j.connectToDatabase()
    all_func_node = getALLFuncNode(j)
    
    for node in all_func_node:
        
        testID = getFuncFile(j, node._id).split('/')[-4]
        
        path = os.path.join("pdg_aug", testID)
        store_file_name = node.properties['name'] + '_' + str(node._id)
        store_path = os.path.join(path, store_file_name)
        
        if os.path.exists(store_path):
            continue

        fin = open(os.path.join("pdg_db", testID, store_file_name))
        pdg = pickle.load(fin)
        fin.close()

        cfg_path = os.path.join("cfg_aug", testID, store_file_name)
        if not os.path.exists(cfg_path):
          continue

        for _file in os.listdir(cfg_path):
            if _file == 'dict_if2cfgnode':
                fin = open(os.path.join(cfg_path, _file))
                dict_if2cfgnode = pickle.load(fin)
                fin.close()

            elif _file == 'dict_cfgnode2if':
                fin = open(os.path.join(cfg_path, _file))
                dict_cfgnode2if = pickle.load(fin)
                fin.close()

            elif _file == 'dict_loop2cfgnode':
                fin = open(os.path.join(cfg_path, _file))
                dict_loop2cfgnode = pickle.load(fin)
                fin.close()

            elif _file == 'dict_cfgnode2loop':
                fin = open(os.path.join(cfg_path, _file))
                dict_cfgnode2loop = pickle.load(fin)
                fin.close()

            elif _file == 'dict_switch2cfgnode':
                fin = open(os.path.join(cfg_path, _file))
                dict_switch2cfgnode = pickle.load(fin)
                fin.close()

            elif _file == 'dict_cfgnode2switch':
                fin = open(os.path.join(cfg_path, _file))
                dict_cfgnode2switch = pickle.load(fin)
                fin.close()

            else:    # 如果文件为cfg
                fin = open(os.path.join(cfg_path, _file))
                cfg = pickle.load(fin)
                fin.close()

        d_use, d_def = getUseDefVarByPDG(j, pdg)

        # 1
        aug_pdg_1 = completeReverseDataEdgeOfPDG(pdg, d_use, d_def, dict_if2cfgnode, dict_cfgnode2if, dict_loop2cfgnode, dict_cfgnode2loop, dict_switch2cfgnode, dict_cfgnode2switch)

        # 2, 3
        aug_pdg_2 = completeBranchEdgeOfPDG(aug_pdg_1, dict_if2cfgnode, dict_cfgnode2if, dict_loop2cfgnode, dict_cfgnode2loop, dict_switch2cfgnode, dict_cfgnode2switch)

        # 4, 5
        aug_pdg_3 = trackVariableLifetime(aug_pdg_2, d_use, d_def, dict_if2cfgnode, dict_cfgnode2if, dict_loop2cfgnode, dict_cfgnode2loop, dict_switch2cfgnode, dict_cfgnode2switch)

        if not os.path.exists(path):
            os.mkdir(path)
           
        f = open(store_path, 'wb')
        pickle.dump(aug_pdg_3, f, True)
        f.close()
    

if __name__ == '__main__':
    main()             



