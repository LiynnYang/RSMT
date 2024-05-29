# arr = [[0,0],[0.25,0.25],[0.51,0.26],[0.75,0.27],[0.26,0.51],[0.52,0.52],[0.76,0.75],[0.8,0.9]]
import math

def read_file(filename):
    edges = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            edge = list(map(int, line.split()))
            edges.append(edge)
    return edges


def findConnectDict(edges, N):
    connectDict = {}
    for i in range(N+2):
        connectDict[i] = []
    for i in range(len(edges)):
        edge = edges[i]
        connectDict[edge[0]].append(edge[1])
        connectDict[edge[1]].append(edge[0])
    return connectDict


def findCircle(addEdge, edges, N):
    connectDict = findConnectDict(edges, N)
    beginNode = addEdge[0]
    endNode = addEdge[1]
    # 开始用DFS的思想来找从一个端点到另一个端点的路径
    visited = [False for i in range(N+2)]
    path = [-1 for i in range(N+2)]
    simuStack = []
    simuStack.append(beginNode)
    while simuStack:
        nowNode = simuStack.pop()
        visited[nowNode] = True
        if nowNode == endNode:
            break
        for tmp_node in connectDict[nowNode]:
            if not visited[tmp_node]:
                path[tmp_node] = nowNode
                simuStack.append(tmp_node)
    # 回溯寻边
    resEdge = []
    temp_node = endNode
    while temp_node != beginNode:
        resEdge.append([temp_node, path[temp_node]])
        temp_node = path[temp_node]
    resEdge.append(addEdge)  # 再加上加的边
    edges.append(addEdge)
    return resEdge


def chooseCase(n, m, arr):
    arr_n = arr[n]
    arr_m = arr[m]
    if arr_n[0] <= arr_m[0] and arr_n[1] <= arr_m[1]:
        return 1
    elif arr_n[0] >= arr_m[0] and arr_n[1] >= arr_m[1]:
        return 2
    elif arr_n[0] <= arr_m[0] and arr_n[1] > arr_m[1]:
        return 3
    elif arr_n[0] >= arr_m[0] and arr_n[1] < arr_m[1]:
        return 4
    return -1


def calCost(edge, edges, arr):  # edge存在在edges里面，所以要将nm去掉
    n = edge[0]  # 指出节点
    m = edge[1]  # 指入节点
    case = chooseCase(n, m, arr)  # 根据点的分布查找是属于纸上推导的哪种情况
    R = [tmpEdge[0] for tmpEdge in edges if tmpEdge[1] == m]  # 其余指向节点m的一阶邻居节点集
    R.remove(n)
    C = [tmpEdge[1] for tmpEdge in edges if tmpEdge[0] == n]  # 从n指出的其他节点的一阶邻居节点集
    C.remove(m)
    overlap_x, overlap_y = 0, 0
    if case == 1:
        a, b = n, m
        tempy = [arr[i][1]-arr[a][1] for i in C]
        tempy.append(0)
        y1 = max(tempy)
        overlap_y = min(y1, arr[b][1] - arr[a][1])
        tempx = [arr[b][0] -arr[i][0] for i in R]
        tempx.append(0)
        x1 = max(tempx)
        overlap_x = min(x1, arr[b][0] - arr[a][0])
    elif case == 2:
        a, b = m, n
        tempy = [arr[b][1] - arr[i][1] for i in C]
        tempy.append(0)
        y1 = max(tempy)
        overlap_y = min(y1, arr[b][1] - arr[a][1])
        tempx = [arr[i][0] - arr[a][0] for i in R]
        tempx.append(0)
        x1 = max(tempx)
        overlap_x = min(x1, arr[b][0] - arr[a][0])
    elif case == 3:
        a, b = n, m
        tempy = [arr[a][1] - arr[i][1] for i in C]
        tempy.append(0)
        y1 = max(tempy)
        overlap_y = min(y1, arr[a][1] - arr[b][1])
        tempx = [arr[b][0] - arr[i][0] for i in R]
        tempx.append(0)
        x1 = max(tempx)
        overlap_x = min(x1, arr[b][0] - arr[a][0])
    elif case == 4:
        a, b = m, n
        tempy = [arr[i][1] - arr[b][1] for i in C]
        tempy.append(0)
        y1 = max(tempy)
        overlap_y = min(y1, arr[a][1] - arr[b][1])
        tempx = [arr[i][0] - arr[a][0] for i in R]
        tempx.append(0)
        x1 = max(tempx)
        overlap_x = min(x1, arr[b][0] - arr[a][0])
    return abs(arr[m][0]-arr[n][0]) + abs(arr[m][1]-arr[n][1]) - (overlap_x + overlap_y)


def breakCircle(resedge, edges, arr):
    cost = [calCost(i, edges, arr) for i in resedge]
    maxCostIndex = cost.index(max(cost))
    distance = max(cost) - cost[-1]
    deleteEdge = resedge[maxCostIndex]
    edges.remove(deleteEdge)
    return edges, deleteEdge, distance


def foundBreak(addedge, edges, N, arr):
    resedge = findCircle(addedge, edges, N)  # 此时返回的边是无向的
    for i in range(len(resedge)):
        edge = resedge[i]
        if edge in edges:
            continue
        elif [edge[1], edge[0]] in edges:
            resedge[i] = [edge[1], edge[0]]

    edges, deleteEdge, distance = breakCircle(resedge, edges, arr)
    return edges, deleteEdge, distance


def calCost_direct(edge, adj_in, adj_out, arr):  # 原本的边就不存在
    n = edge[0]  # 指出节点
    m = edge[1]  # 指入节点
    case = chooseCase(n, m, arr)  # 根据点的分布查找是属于纸上推导的哪种情况
    R = adj_in[m]  # 其余指向节点m的一阶邻居节点集
    # R.remove(n)
    C = adj_out[n]  # 从n指出的其他节点的一阶邻居节点集
    # C.remove(m)
    overlap_x, overlap_y = 0, 0
    if case == 1:
        a, b = n, m
        tempy = [arr[i][1]-arr[a][1] for i in C]
        tempy.append(0)
        y1 = max(tempy)
        overlap_y = min(y1, arr[b][1] - arr[a][1])
        tempx = [arr[b][0] -arr[i][0] for i in R]
        tempx.append(0)
        x1 = max(tempx)
        overlap_x = min(x1, arr[b][0] - arr[a][0])
    elif case == 2:
        a, b = m, n
        tempy = [arr[b][1] - arr[i][1] for i in C]
        tempy.append(0)
        y1 = max(tempy)
        overlap_y = min(y1, arr[b][1] - arr[a][1])
        tempx = [arr[i][0] - arr[a][0] for i in R]
        tempx.append(0)
        x1 = max(tempx)
        overlap_x = min(x1, arr[b][0] - arr[a][0])
    elif case == 3:
        a, b = n, m
        tempy = [arr[a][1] - arr[i][1] for i in C]
        tempy.append(0)
        y1 = max(tempy)
        overlap_y = min(y1, arr[a][1] - arr[b][1])
        tempx = [arr[b][0] - arr[i][0] for i in R]
        tempx.append(0)
        x1 = max(tempx)
        overlap_x = min(x1, arr[b][0] - arr[a][0])
    elif case == 4:
        a, b = m, n
        tempy = [arr[i][1] - arr[b][1] for i in C]
        tempy.append(0)
        y1 = max(tempy)
        overlap_y = min(y1, arr[a][1] - arr[b][1])
        tempx = [arr[i][0] - arr[a][0] for i in R]
        tempx.append(0)
        x1 = max(tempx)
        overlap_x = min(x1, arr[b][0] - arr[a][0])
    return abs(arr[m][0]-arr[n][0]) + abs(arr[m][1]-arr[n][1]) - (overlap_x + overlap_y)


def direction(n, m, adj_in, adj_out, arr):
    edge1 = [n, m]
    edge2 = [m, n]
    length1 = round(calCost_direct(edge1, adj_in, adj_out, arr), 5)
    # print(length1)
    length2 = round(calCost_direct(edge2, adj_in, adj_out, arr), 5)
    # print(length2)
    if length1 > length2:
        return edge2, length2
    else:
        return edge1, length1


# 计算圆 与 线段相交的点
def line_intersect_circle(x0, y0, r, x1, y1, x2, y2):
    '''
    计算圆心与线段的交点
    :param circle_center: 圆心 （x0,y0）
    :param r: 半径
    :param point1: 线段起始点（x1,y1）
    :param point2: 线段终点（x2,y2）
    :return: 线段与圆的交点
    '''
    # x0, y0=circle_center
    # x1, y1=point1
    # x2, y2=point2

    #求直线y=kx+b的斜率及b
    k = (y1 - y2) / (x1 - x2)
    b0 = y1 - k * x1
    #直线与圆的方程化简为一元二次方程ax**2+bx+c=0
    a = k ** 2 + 1
    b = 2 * k * (b0 - y0) - 2 * x0
    c = (b0 - y0) ** 2 + x0 ** 2 - r ** 2
    #判别式判断解，初中知识
    delta = b ** 2 - 4 * a * c
    if delta >= 0:
        p1x = round((-b - delta**(0.5)) / (2 * a), 5)
        p2x = round((-b + delta**(0.5)) / (2 * a), 5)
        p1y = round(k * p1x + b0, 5)
        p2y = round(k * p2x + b0, 5)
        inp = [[p1x, p1y], [p2x, p2y]]
        inp = [p for p in inp if p[0] >= min(x1, x2) and p[0] <= max(x1, x2)]
    else:
        inp = []
    return inp if inp != [] else [[x1, y1]]

# 注意输入时，block1的大小大于block2的大小；先输入block1重心的坐标，再输入block2重心的坐标
def cengravity(x1, y1, x2, y2, block1_number, block2_number):
    x = block1_number/block2_number
    r = ((block1_number - block2_number)/(block1_number + block2_number)) * math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
    # cor是找到的新重心的点的x，y坐标
    cor = line_intersect_circle(x1, y1, r, x1, y1, x2, y2)
    return cor


if __name__ == '__main__':
    # N = 8  # 节点的个数
    # edges = [[0,2],[1,4],[4,5],[2,5],[5,6],[3,6],[6,7]]
    # addedge = [0,1]
    # a = [[0, 0], [0.25, 0.25], [0.51, 0.26], [0.75, 0.27], [0.26, 0.51], [0.52, 0.52], [0.76, 0.75], [0.8, 0.9]]
    # edges.append(addedge)
    # newedges = foundBreak(addedge, edges, N, a)  #  找环并且破环
    # print(newedges)
    # edge = direction(0, 1, edges, a)
    # print(edge)
    # cor = cengravity(0, 0, 4, 0, 3, 1)
    # print(cor)
    arr = [[0.6, 0.2], [0.8, 0.1], [0.7, 0.3]]
    adj_in = [[1], [], []]
    adj_out = [[], [0], []]
    # edges = [[1, 0]]
    print(direction(2, 0, adj_in, adj_out, arr))