#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <utility>
#include <queue>
#include <tuple>
#include <list>
#include <algorithm>
#include <cstdio>
#include <ctime>

namespace py=pybind11;

struct cmp{
	bool operator()(std::pair<int,int> p1, std::pair<int,int> p2){
		//按照第一个元素升序排序
		return p1.first < p2.first;
	}
};

int get_start_point(std::vector<std::vector<float> > arr){
    int start_point = -1;
    float sp_x = 1.;
    float sp_y = 0.;
    for(int i=0; i < (int)arr.size(); i++){
        float x = arr[i][0];
        float y = arr[i][1];
        if (x < sp_x){
            sp_x = x;
            start_point = i;
        }

        else if (x == sp_x && y > sp_y){
            sp_y = y;
            start_point = i;
        }
    }
    return start_point;
}


std::list<int> BFS(int start_point, int block_id, std::vector<int> &belongs, std::vector<int> &start_points, int degree, std::vector<int> de, std::vector<std::vector<int> > mp, std::vector<bool> &visited, int m_block=20){
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int,int> >, cmp> q_pre;
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int,int> >, cmp> q_cur;
    q_pre.push(std::make_pair(de[start_point], start_point));
    std::list<int> block;

    while(1){
        if (!q_pre.empty()){
            int visit_cur = q_pre.top().second;
            q_pre.pop();
            visited[visit_cur] = 1;
            block.push_back(visit_cur);
            belongs[visit_cur] = block_id;

//            for(int i=0; i<degree; i++){
//                if(mp[visit_cur][i] == 1 && visited[i] == 0)
//                    q_cur.push(std::make_pair(de[i], i));
//            }
            for(int p: link_table[visit_cur]){
                if(visited[p] == 0)
                    q_cur.push(std::make_pair(de[p], p));
            }

            if((int)block.size() == m_block)
            {
                // return block, q_pre, q_cur;
                while(!q_pre.empty()){
                    start_points.push_back(q_pre.top().second);
                    q_pre.pop();
                }

                while(!q_cur.empty()){
                    start_points.push_back(q_cur.top().second);
                    q_cur.pop();
                }

                return block;
            }
        }

        else if (!q_cur.empty()){
            while(!q_cur.empty()){
                q_pre.push(q_cur.top());
                q_cur.pop();
            }
        }
        else{
            // 将现有的不足20的块返回
            // return block, q_pre, q_cur;
            while(!q_pre.empty()){
                    start_points.push_back(q_pre.top().second);
                    q_pre.pop();
            }
            while(!q_cur.empty()){
                start_points.push_back(q_cur.top().second);
                q_cur.pop();
            }
            return block;
        }
    }
}

std::tuple<std::vector<std::list<int> >, std::vector<int>> post_process(std::vector<std::vector<int> > link_table, std::vector<std::list<int> > blocks, std::vector<int> belongs, int m_block=20, int block_max=30){
    // 不会有两个小于20的block直接相连，所以只合并小于10的block至20的block
    // print('post processing(merge some small blocks together)')
    //merge_into = np.array(list(range(len(blocks))))  // 初始时每个块的连接是他本身的id号
    // 根据mp得到邻接表
    std::vector<std::vector<int> > link_block((int)blocks.size(), std::vector<int>((int)blocks.size(), 0));
    std::vector<std::list<int> > table_block((int)blocks.size());
    std::list<int> com20_id;
    std::vector<int> count_uncom_link((int)blocks.size(), 0); //统计链接的不完整block的个数
    for(unsigned i=0; i<blocks.size(); i++){
        std::list<int> block = blocks[i];
        if((int)block.size() == m_block)
            com20_id.push_back(i);
        for(int point : block){
            std::vector<int> link_point = link_table[point];
            for(int p : link_point){
                unsigned j = belongs[p];
                if(i != j && link_block[i][j] != 1){
                    if((int)blocks[j].size() < m_block)//还未统计且是不完整块
                        count_uncom_link[i]++;
                    link_block[i][j] = 1;
                    link_block[j][i] = 1;
                    table_block[i].push_back(j);
                    table_block[j].push_back(i);
                }
            }
        }

    }

    // 按连接不完整块的多少升序, 先对链接的少的合并
    com20_id.sort([count_uncom_link] (const int& v1, const int& v2) { return count_uncom_link[v1] < count_uncom_link[v2];});
    std::list<int> blank_block;
    for(int id : com20_id){// 对正常块进行遍历
        std::list<int> linked = table_block[id];
        linked.sort([blocks](const int& a, const int& b){return blocks[a].size() < blocks[b].size();});  // 将其连接的block按从小到大排序
        std::list<int> block1 = blocks[id];
        for(int link_id : linked){
            std::list<int> block2 = blocks[link_id];
            if(blocks[link_id].size() != 0 && blocks[id].size() + blocks[link_id].size() <= (unsigned)block_max){
                blocks[id].splice(blocks[id].begin(), blocks[link_id]);  //blocks[link_id] (empty)
                for(int p : block2)
                    belongs[p] = id;
                blank_block.push_back(link_id);
            }
            else if(blocks[id].size() + blocks[link_id].size() > (unsigned)block_max)
                break;
        }
    }

    blank_block.sort();
    std::vector<int> sub(blocks.size(), 0);

    for(unsigned i : blank_block){
        for(unsigned j=i; j<blocks.size(); j++){
            sub[j]++;
        }
    }
    for(unsigned i=0; i<belongs.size(); i++)
        belongs[i] -= sub[belongs[i]];

    auto it1 = blank_block.rbegin();
    while(it1 !=blank_block.rend()){
        auto it = blocks.begin();
        blocks.erase(it + *it1);
        it1++;
    }
    return std::tuple<std::vector<std::list<int> >, std::vector<int>>{blocks, belongs};
}

std::tuple<std::vector<std::list<int> >, std::vector<int> > generate_blocks(std::vector<std::vector<float> >arr, std::vector<std::vector<int> >link_table, int m_block=20, int block_max=30){
    int degree = arr.size();
    int start_point = get_start_point(arr);
//    printf("start point %d", start_point);
    std::vector<int> start_points;
    start_points.push_back(start_point);
    std::vector<int> de(degree, 0);
    std::vector<int> belongs(degree, 0);
    std::vector<std::list<int> > blocks;
    std::vector<bool> visited(degree, 0);
    int block_id = 0;
//    clock_t  time1 = clock();
    for(int i=0; i<(int)start_points.size(); i++){
        int start = start_points[i];
        if(visited[start])
            continue;
        std::list<int> block = BFS(start, block_id, belongs, start_points, degree, de, link_table, visited, m_block);
        blocks.push_back(block);
        block_id++;
    }
//    clock_t  time2 = clock();
//    std::tuple<std::vector<std::list<int> >, std::vector<int> > ans = post_process(link_table, blocks, belongs, m_block, block_max);
//    double time3 = clock();
//    printf("init blocks cost time:%.5fs", (double)(time2-time1)/CLOCKS_PER_SEC);
//    return ans;
    return std::tuple<std::vector<std::list<int> >, std::vector<int> >{blocks, belongs};
}



//c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) BFS.cpp -o generate_blocks$(python3-config --extension-suffix)
PYBIND11_MODULE(generate_blocks, m) {  //模块名
    m.doc() = "generate init blocks by BFS strategy"; // optional module docstring

    m.def("generate_blocks", &generate_blocks, "generate init blocks",  //方法名
            py::arg("arr"), py::arg("mp"), py::arg("link_table"), py::arg("m_block")=20, py::arg("block_max")=30);
}