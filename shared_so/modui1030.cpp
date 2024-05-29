#include<cstdio>
#include<algorithm>
#include<cstring>
#include<cmath>
#include<fstream>
#include<iostream>
using namespace std;

using namespace std;
const int MAXN = 100010;
const int MAXE = 4 * MAXN;

/*Kruskal  Algorithm*/
int parent[MAXN];

int father(int u)
{
	while (parent[u] != u)
	{
		parent[u] = parent[parent[u]];
		u = parent[u];
	}
	return u;
}

bool connect(int u, int v)
{
	int fu = father(u);
	int fv = father(v);
	if (fv == fu) return false;
	parent[fu] = fv;
	return true;
}

struct Edge
{
	int u, v, w;
	Edge() {}
	Edge(int u, int v, int w) :u(u), v(v), w(w) {}
	bool operator<(const Edge &ee)const
	{
		return w < ee.w;
	}
};

Edge edge[MAXE];
int cnt;

void addEdge(int u, int v, int val)
{
	edge[cnt++] = Edge(u, v, val);
}
/*线段树  */
struct Point
{
	int x, y, id;
	bool operator<(const Point &t) const
	{
		if (x == t.x) return y < t.y;
		return x < t.x;
	}
};

Point point[MAXN];
const int INF = 0x3f3f3f3f;

struct Node
{
	int len, id;
};
Node tree[(MAXN + 20) << 2];

void build(int l, int r, int rt)//init tree
{
	tree[rt].len = INF;
	tree[rt].id = -1;
	if (l == r) return;
	int mid = (l + r) >> 1;
	build(l, mid, rt << 1);
	build(mid + 1, r, rt << 1 | 1);
}

void update(int l, int r, int rt, Point &p, int pos)//push Point into tree
{
	if (l == r)
	{
		if (tree[rt].len > p.x + p.y)
		{
			tree[rt].len = p.x + p.y;
			tree[rt].id = p.id;
		}
		return;
	}
	int mid = (l + r) >> 1;
	if (pos <= mid) update(l, mid, rt << 1, p, pos);
	else update(mid + 1, r, rt << 1 | 1, p, pos);
	if (tree[rt << 1].len < tree[rt << 1 | 1].len)
	{
		tree[rt].len = tree[rt << 1].len;
		tree[rt].id = tree[rt << 1].id;
	}
	else
	{
		tree[rt].len = tree[rt << 1 | 1].len;
		tree[rt].id = tree[rt << 1 | 1].id;
	}
}

Node query(int l, int r, int L, int R, int rt)//从大于point[i].y-point[i].x的节点找最小值
{
	if (l == L && r == R) return tree[rt];
	int mid = (l + r) >> 1;
	if (R <= mid) return query(l, mid, L, R, rt << 1);
	else if (L >= mid + 1) return query(mid + 1, r, L, R, rt << 1 | 1);
	else
	{
		Node t1 = query(l, mid, L, mid, rt << 1);
		Node t2 = query(mid + 1, r, mid + 1, R, rt << 1 | 1);
		if (t1.len < t2.len) return t1;
		else return t2;
	}
}

int cpy[MAXN], arr[MAXN];

void solve(int n)
{
	sort(point + 1, point + 1 + n);
	for (int i = 1; i <= n; i++)
	{
		arr[i] = point[i].y - point[i].x;
		cpy[i] = arr[i];
	}
	sort(cpy + 1, cpy + 1 + n);
	int cc = unique(cpy + 1, cpy + 1 + n) - cpy;//离散化
	for (int i = 1; i <= n; i++)
	{
		arr[i] = lower_bound(cpy + 1, cpy + cc, arr[i]) - cpy;
	}
	build(1, n, 1);
	for (int i = n; i > 0; i--)
	{
		int len = point[i].x + point[i].y;
		Node t = query(1, n, arr[i], n, 1);
		if (t.id != -1) addEdge(point[i].id, t.id, abs(len - t.len));
		update(1, n, 1, point[i], arr[i]);
	}
}

/*Kruskal  Algorithm*/
int kruskal_mst(int n)
{
	int u, v;
	long long sum = 0;
	ofstream outFile("tempMST.txt", ios::out);
	sort(edge, edge + cnt);
	for (int i = 0; i < cnt; i++)
	{
		u = edge[i].u; v = edge[i].v;
		if (connect(u, v))
		{
			outFile << u << " " << v << endl;
			sum += (long long)edge[i].w;
		}
	}
	outFile.close();
	return sum;
}

/*Kruskal  Algorithm*/
void kruskal_mst2(int * edge1, int * edge2, int n)
{
	int u, v;
	long long sum = 0;
	sort(edge, edge + cnt);
	int count = 0;
	for (int i = 0; i < cnt; i++)
	{
		u = edge[i].u; v = edge[i].v;
		if (connect(u, v))
		{
			edge1[count] = u;
			edge2[count] = v;
			count++;
			sum += (long long)edge[i].w;
		}
	}
}


extern "C"
void modui(char * filename) {
	ifstream myfile(filename, ios::in);
	if (!myfile.is_open()) {
		cout << "C++打开文件错误" << endl;
		exit(0);
	}
	cout << "C++文件正确打开" << endl;
	int n;
	myfile >> n;
	cnt = 0;

	for (int i = 1; i <= n; i++)
	{
		myfile >> point[i].x >> point[i].y;
		point[i].id = i;
	}
	myfile.close();
	for (int i = 1; i <= n; i++)
	{
		parent[i] = i;
	}
	solve(n);
	for (int i = 1; i <= n; i++)
		point[i].y = -point[i].y;
	solve(n);
	for (int i = 1; i <= n; i++)
		point[i].y = -point[i].y, swap(point[i].x, point[i].y);
	solve(n);
	for (int i = 1; i <= n; i++)
		point[i].y = -point[i].y;
	solve(n);
	//cout << "Total Weight = " << kruskal_mst(n) << endl;
}

extern "C"
void modui2(int * x, int * y, int *edge1, int *edge2, int size) {
	int n = size;
	cnt = 0;
	for (int i = 1; i <= n; i++)
	{
		point[i].x = x[i-1];
		point[i].y = y[i-1];
		point[i].id = i;
	}
	for (int i = 1; i <= n; i++)
	{
		parent[i] = i;
	}
	solve(n);
	for (int i = 1; i <= n; i++)
		point[i].y = -point[i].y;
	solve(n);
	for (int i = 1; i <= n; i++)
		point[i].y = -point[i].y, swap(point[i].x, point[i].y);
	solve(n);
	for (int i = 1; i <= n; i++)
		point[i].y = -point[i].y;
	solve(n);
	kruskal_mst2(edge1, edge2, n);
}

int main()
{
	char str[] = "tempInputFile.txt";
	modui(str);
	return 0;
}