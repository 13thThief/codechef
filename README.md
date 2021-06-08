# C++ Snippets

```
typedef long long ll;

typedef unsigned long long ull;

typedef pair<ll, ll> pll;

typedef pair<double, double> pdd;

typedef vector<ll> vll;

typedef vector<pll> vpl;

#define one(a,n) fill_n((a), (n), 1)

#define rep(i,a) for(int i=0;i<(a);++i)

#define repp(i,a,b) for(int i=(a);i<(b);++i)

#define fill(a,b) memset((a),(b),sizeof((a)))

#define clr(a) memset((a),0,sizeof((a)))

#define foreach(a,it) for( typeof((a).begin()) it=(a).begin();it!=(a).end();it++ )

#define mp make_pair

#define pb push_back

#define all(s) s.begin(),s.end()

#define sz(a) int(a.size())

#define ft first

#define sd second

#define sf(n) scanf("%d",&n)

#define pf(n) printf("%d\n",n)

#define sll(n) scanf("%lld",&n)

#define pll(n) printf("%lld\n",n)

#define fast_io ios_base::sync_with_stdio(false);cin.tie(NULL)

const double eps = 1e-8;

const ll inf = 2e9+1;

const ll mod = 1e9+7;

const int N = int(2e5)+10;

ll power(ll x, ll y){
ll t=1;
while(y>0){
 if (y%2) y-=1, t=t*x%mod;
 else y/=2, x=x*x%mod;
 }
return t;
}

#define mp make_pair

#define pb push_back

#define wl(n) while(n--)

#define fi first

#define se second

#define all(c) c.begin(),c.end()

typedef vector<int> vi;

typedef pair<int,int> pii;

typedef pair<int,pair<int,int> > piii ;

typedef pair<ll,ll> pll;

typedef pair<ll,int> pli;

#define sz(a) int((a).size())

#define ini(a,v) memset(a,v,sizeof(a))

#define sc(x) scanf("%d",&x)

#define sc2(x,y) scanf("%d%d",&x,&y)

#define sc3(x,y,z) scanf("%d%d%d",&x,&y,&z)

#define scl(x) scanf("%lld",&x)

#define scl2(x,y) scanf("%lld%lld",&x,&y)

#define scl3(x,y,z) scanf("%lld%lld%lld",&x,&y,&z)

#define scs(s) scanf("%s",s);

#define gcd __gcd

#define debug() printf("here\n")

#define chk(a) cerr << endl << #a << " : " << a << endl

#define chk2(a,b) cerr << endl << #a << " : " << a << "\t" << #b << " : " << b << endl

#define tr(container, it) for(typeof(container.begin()) it = container.begin(); it != container.end(); it++)

#define MOD 1000000007

#define inf ((1<<29)-1)

#define linf ((1LL<<60)-1)

const double eps = 1e-9;

#define ms(s, n) memset(s, n, sizeof(s))

#define FOR(i, a, b) for (int i = (a); i < (b); i++)

#define FORd(i, a, b) for (int i = (a) - 1; i >= (b); i--)

#define FORall(it, a) for (__typeof((a).begin()) it = (a).begin(); it != (a).end(); it++)

#define sz(a) int((a).size())

#define present(t, x) (t.find(x) != t.end())

#define all(a) (a).begin(), (a).end()

#define uni(a) (a).erase(unique(all(a)), (a).end())

#define pb push_back

#define pf push_front

#define mp make_pair

#define fi first

#define se second

#define prec(n) fixed<<setprecision(n)

#define bit(n, i) (((n) >> (i)) & 1)

#define bitcount(n) __builtin_popcountll(n)

typedef long long ll;

typedef unsigned long long ull;

typedef long double ld;

typedef pair<int, int> pi;

typedef vector<int> vi;

typedef vector<pi> vii;

const int MOD = (int) 1e9 + 7;

const int INF = (int) 1e9;

const ll LINF = (ll) 1e18;

const ld PI = acos((ld) -1);

const ld EPS = 1e-9;

inline ll gcd(ll a, ll b) {ll r; while (b) {r = a % b; a = b; b = r;} return a;}

inline ll lcm(ll a, ll b) {return a / gcd(a, b) * b;}

inline ll fpow(ll n, ll k, int p = MOD) {ll r = 1; for (; k; k >>= 1) {if (k & 1) r = r * n % p; n = n * n % p;} return r;}

template<class T> inline int chkmin(T& a, const T& val) {return val < a ? a = val, 1 : 0;}

template<class T> inline int chkmax(T& a, const T& val) {return a < val ? a = val, 1 : 0;}

template<class T> inline T isqrt(T k) {T r = sqrt(k) + 1; while (r * r > k) r--; return r;}

template<class T> inline T icbrt(T k) {T r = cbrt(k) + 1; while (r * r * r > k) r--; return r;}

inline void addmod(int& a, int val, int p = MOD) {if ((a = (a + val)) >= p) a -= p;}

inline void submod(int& a, int val, int p = MOD) {if ((a = (a - val)) < 0) a += p;}

inline int mult(int a, int b, int p = MOD) {return (ll) a * b % p;}

inline int inv(int a, int p = MOD) {return fpow(a, p - 2, p);}

printf("\nTime elapsed: %dms", 1000 * clock() / CLOCKS_PER_SEC);
```


BLITZKRIEG
Bit Manipulation:
1.	To multiply by 2^x : 	S = S<<x
2.	To divide by 2^x : 	S = S>>x
3.	To set jth bit : 	S|=(1<<j)
4.	To check jth bit: 	T = S & (1<<j) (If T=0 not set else set)
5.	To turn off jth bit:	S&=~(1<<j)
6.	To flip jth bit:		S^=(1<<j)
7.	To get value of LSB	T = (S & (-S)) (Gives 2^position)
that if on:
8.	To turn on all bits	S = (1<<S) - 1
in a set of size n:

Priority Queue:
priority_queue<int, vector<int>, greater<int>>	q;
(greater for lowest to highest priority)

Segment Tree:
//1-Based Indexing of Seg Tree is used
//Left child 2*p, Right Child 2*p+1, Parent p/2
//For n=50000 use N=132000, n=1e5 use 264000

//Point Update, add z to pth position in the array
void update(int p, int z) {
    int low=0;
    int high=N-1;
    int mid;
    int pos=1;
    while(low<high) {
        mid=(low+high)/2;
        if (p>mid)        {
            pos=(2*pos)+1;
            low=mid+1;
        }
        else        {
            pos=(2*pos);
            high=mid;
        }
    }
    seg[pos]+=z;
    pos/=2;
    while(pos>0)    {
        seg[pos]+=z;
        pos/=2;
    }
}
//Standard sum query from qlow to qhigh (For lazy refer to update)
int query(int low, int high, int p){
    if (qlow<=low && qhigh>=high)
        return seg[p];
    if (qlow>qhigh || qlow>high || qhigh<low)
        return 0;
    int mid=(low+high)/2;
    return query(low, mid, (2*p)) + query(mid+1, high, (2*p)+1);
}

//Range Update With Lazy
inline void update(int low, int high, int pos){
    if (lazy[pos]>0)    {
        seg[pos]+=(((LL)(high-low+1))*((LL)(lazy[pos])));
        if (low!=high)        {
            lazy[(2*pos)]+=lazy[pos];
            lazy[(2*pos)+1]+=lazy[pos];
        }
        lazy[pos]=0;
    }
    if (qlow>high||qhigh<low||qhigh<qlow)
    return;
    if (qlow<=low&&qhigh>=high)    {
        seg[pos]+=(LL)(high-low+1);	//Adding 1, multiply by v for v
        if (low!=high)        {
            lazy[(2*pos)]++;		//Add v for v
            lazy[(2*pos)+1]++;		//Add v for v
        }
        return;
    }
    int mid=(low+high)/2;
    update(low,mid,(2*pos));
    update(mid+1,high,(2*pos)+1); 
    seg[pos]=seg[(2*pos)]+seg[(2*pos)+1];
}
Tips:
1.	For counting problems, try counting number of incorrect ways instead of correct ways. (Filtering vs Generating)
2.	Prune Infeasible/Inferior Search Space Early
3.	Utilize Symmetries
4.	Pre-Computation a.k.a Pre-Calculation
5.	Try solving the problem backwards
Techniques to solve:
1.	Binary Search the answer
2.	Meet in the middle (Solve left half, Solve right half, combine)
3.	Greedy
4.	DP
#define N 150
vi v[(N*N)+1];
int ans[(N*N)+1],p1[(N*N)+1],p2[(N*N)+1],d[(N*N)+1],n,m;
inline bool dfs(int s)
{
    if (s!=m)
    {
        int i;
        for (i=0;i<v[s].size();i++)
        {
            if (d[p2[v[s][i]]]==d[s]+1)
            {
                if (dfs(p2[v[s][i]]))
                {
                    p2[v[s][i]]=s;
                    p1[s]=v[s][i];
                    ans[s]=v[s][i];
                    return true;
                }
            }
        }
        d[s]=inf;
        return false;
    }
    return true;
}
bool bfs()
{
    int i;
    queue<int> q;
    for (i=0;i<m;i++)
    {
        if (p1[i]==m)
        {
            d[i]=0;
            q.push(i);
        }
        else
            d[i]=inf;
    }
    d[m]=inf;
    while(!q.empty())
    {
        int s=q.front();
        q.pop();
        if (d[s]<d[m])
        {
            for (i=0;i<v[s].size();i++)
            {
                if (d[p2[v[s][i]]]==inf)
                {
                    d[p2[v[s][i]]]=d[s]+1;
                    q.push(p2[v[s][i]]);
                }
            }
        }
    }
    return (d[m]!=inf);
}
int hopkarp()
{
    int i,j;
    for (i=0;i<m;i++)
    {
        p1[i]=m;
        p2[i]=m;
    }
    int res=0;
    while(bfs())
    {
        for (i=0;i<m;i++)
            if (p1[i]==m && dfs(i))
                res++;
    }
    return res;
}
Toposort, Articulation Points & Bridges, Strongly Connected Components:
typedef pair<int, int> ii;      // In this chapter, we will frequently use these
typedef vector<ii> vii;      // three data type shortcuts. They may look cryptic
typedef vector<int> vi;   // but shortcuts are useful in competitive programming

#define DFS_WHITE -1 // normal DFS, do not change this with other values (other than 0), because we usually use memset with conjunction with DFS_WHITE
#define DFS_BLACK 1

vector<vii> AdjList;

void printThis(char* message) {
  printf("==================================\n");
  printf("%s\n", message);
  printf("==================================\n");
}

vi dfs_num;     // this variable has to be global, we cannot put it in recursion
int numCC;

void dfs(int u) {          // DFS for normal usage: as graph traversal algorithm
  printf(" %d", u);                                    // this vertex is visited
  dfs_num[u] = DFS_BLACK;      // important step: we mark this vertex as visited
  for (int j = 0; j < (int)AdjList[u].size(); j++) {
    ii v = AdjList[u][j];                      // v is a (neighbor, weight) pair
    if (dfs_num[v.first] == DFS_WHITE)         // important check to avoid cycle
      dfs(v.first);      // recursively visits unvisited neighbors v of vertex u
} }

// note: this is not the version on implicit graph
void floodfill(int u, int color) {
  dfs_num[u] = color;                            // not just a generic DFS_BLACK
  for (int j = 0; j < (int)AdjList[u].size(); j++) {
    ii v = AdjList[u][j];
    if (dfs_num[v.first] == DFS_WHITE)
      floodfill(v.first, color);
} }

vi topoSort;             // global vector to store the toposort in reverse order

void dfs2(int u) {    // change function name to differentiate with original dfs
  dfs_num[u] = DFS_BLACK;
  for (int j = 0; j < (int)AdjList[u].size(); j++) {
    ii v = AdjList[u][j];
    if (dfs_num[v.first] == DFS_WHITE)
      dfs2(v.first);
  }
  topoSort.push_back(u); }                   // that is, this is the only change

#define DFS_GRAY 2              // one more color for graph edges property check
vi dfs_parent;      // to differentiate real back edge versus bidirectional edge

void graphCheck(int u) {               // DFS for checking graph edge properties
  dfs_num[u] = DFS_GRAY;   // color this as DFS_GRAY (temp) instead of DFS_BLACK
  for (int j = 0; j < (int)AdjList[u].size(); j++) {
    ii v = AdjList[u][j];
    if (dfs_num[v.first] == DFS_WHITE) {     // Tree Edge, DFS_GRAY to DFS_WHITE
      dfs_parent[v.first] = u;                  // parent of this children is me
      graphCheck(v.first);
    }
    else if (dfs_num[v.first] == DFS_GRAY) {             // DFS_GRAY to DFS_GRAY
      if (v.first == dfs_parent[u])          // to differentiate these two cases
        printf(" Bidirectional (%d, %d) - (%d, %d)\n", u, v.first, v.first, u);
      else  // the most frequent application: check if the given graph is cyclic
        printf(" Back Edge (%d, %d) (Cycle)\n", u, v.first);
    }
    else if (dfs_num[v.first] == DFS_BLACK)             // DFS_GRAY to DFS_BLACK
      printf(" Forward/Cross Edge (%d, %d)\n", u, v.first);
  }
  dfs_num[u] = DFS_BLACK;     // after recursion, color this as DFS_BLACK (DONE)
}

vi dfs_low;       // additional information for articulation points/bridges/SCCs
vi articulation_vertex;
int dfsNumberCounter, dfsRoot, rootChildren;

void articulationPointAndBridge(int u) {
  dfs_low[u] = dfs_num[u] = dfsNumberCounter++;      // dfs_low[u] <= dfs_num[u]
  for (int j = 0; j < (int)AdjList[u].size(); j++) {
    ii v = AdjList[u][j];
    if (dfs_num[v.first] == DFS_WHITE) {                          // a tree edge
      dfs_parent[v.first] = u;
      if (u == dfsRoot) rootChildren++;  // special case, count children of root

      articulationPointAndBridge(v.first);

      if (dfs_low[v.first] >= dfs_num[u])              // for articulation point
        articulation_vertex[u] = true;           // store this information first
      if (dfs_low[v.first] > dfs_num[u])                           // for bridge
        printf(" Edge (%d, %d) is a bridge\n", u, v.first);
      dfs_low[u] = min(dfs_low[u], dfs_low[v.first]);       // update dfs_low[u]
    }
    else if (v.first != dfs_parent[u])       // a back edge and not direct cycle
      dfs_low[u] = min(dfs_low[u], dfs_num[v.first]);       // update dfs_low[u]
} }

vi S, visited;                                    // additional global variables
int numSCC;

void tarjanSCC(int u) {
  dfs_low[u] = dfs_num[u] = dfsNumberCounter++;      // dfs_low[u] <= dfs_num[u]
  S.push_back(u);           // stores u in a vector based on order of visitation
  visited[u] = 1;
  for (int j = 0; j < (int)AdjList[u].size(); j++) {
    ii v = AdjList[u][j];
    if (dfs_num[v.first] == DFS_WHITE)
      tarjanSCC(v.first);
    if (visited[v.first])                                // condition for update
      dfs_low[u] = min(dfs_low[u], dfs_low[v.first]);
  }

  if (dfs_low[u] == dfs_num[u]) {         // if this is a root (start) of an SCC
    printf("SCC %d:", ++numSCC);            // this part is done after recursion
    while (1) {
      int v = S.back(); S.pop_back(); visited[v] = 0;
      printf(" %d", v);
      if (u == v) break;
    }
    printf("\n");
} }

int main() {
  int V, total_neighbors, id, weight;

  freopen("in_01.txt", "r", stdin);

  scanf("%d", &V);
  AdjList.assign(V, vii()); // assign blank vectors of pair<int, int>s to AdjList
  for (int i = 0; i < V; i++) {
    scanf("%d", &total_neighbors);
    for (int j = 0; j < total_neighbors; j++) {
      scanf("%d %d", &id, &weight);
      AdjList[i].push_back(ii(id, weight));
    }
  }

  printThis("Standard DFS Demo (the input graph must be UNDIRECTED)");
  numCC = 0;
  dfs_num.assign(V, DFS_WHITE);    // this sets all vertices' state to DFS_WHITE
  for (int i = 0; i < V; i++)                   // for each vertex i in [0..V-1]
    if (dfs_num[i] == DFS_WHITE)            // if that vertex is not visited yet
      printf("Component %d:", ++numCC), dfs(i), printf("\n");   // 3 lines here!
  printf("There are %d connected components\n", numCC);

  printThis("Flood Fill Demo (the input graph must be UNDIRECTED)");
  numCC = 0;
  dfs_num.assign(V, DFS_WHITE);
  for (int i = 0; i < V; i++)
    if (dfs_num[i] == DFS_WHITE)
      floodfill(i, ++numCC);
  for (int i = 0; i < V; i++)
    printf("Vertex %d has color %d\n", i, dfs_num[i]);

  // make sure that the given graph is DAG
  printThis("Topological Sort (the input graph must be DAG)");
  topoSort.clear();
  dfs_num.assign(V, DFS_WHITE);
  for (int i = 0; i < V; i++)            // this part is the same as finding CCs
    if (dfs_num[i] == DFS_WHITE)
      dfs2(i);
  reverse(topoSort.begin(), topoSort.end());                 // reverse topoSort
  for (int i = 0; i < (int)topoSort.size(); i++)       // or you can simply read
    printf(" %d", topoSort[i]);           // the content of `topoSort' backwards
  printf("\n");

  printThis("Graph Edges Property Check");
  numCC = 0;
  dfs_num.assign(V, DFS_WHITE); dfs_parent.assign(V, -1);
  for (int i = 0; i < V; i++)
    if (dfs_num[i] == DFS_WHITE)
      printf("Component %d:\n", ++numCC), graphCheck(i);       // 2 lines in one

  printThis("Articulation Points & Bridges (the input graph must be UNDIRECTED)");
  dfsNumberCounter = 0; dfs_num.assign(V, DFS_WHITE); dfs_low.assign(V, 0);
  dfs_parent.assign(V, -1); articulation_vertex.assign(V, 0);
  printf("Bridges:\n");
  for (int i = 0; i < V; i++)
    if (dfs_num[i] == DFS_WHITE) {
      dfsRoot = i; rootChildren = 0;
      articulationPointAndBridge(i);
      articulation_vertex[dfsRoot] = (rootChildren > 1); }       // special case
  printf("Articulation Points:\n");
  for (int i = 0; i < V; i++)
    if (articulation_vertex[i])
      printf(" Vertex %d\n", i);

  printThis("Strongly Connected Components (the input graph must be DIRECTED)");
  dfs_num.assign(V, DFS_WHITE); dfs_low.assign(V, 0); visited.assign(V, 0);
  dfsNumberCounter = numSCC = 0;
  for (int i = 0; i < V; i++)
    if (dfs_num[i] == DFS_WHITE)
      tarjanSCC(i);

  return 0;
}

Kruskal’s, Prim’s:
// Union-Find Disjoint Sets Library written in OOP manner, using both path compression and union by rank heuristics
class UnionFind {                                              // OOP style
private:
  vi p, rank, setSize;                       // remember: vi is vector<int>
  int numSets;
public:
  UnionFind(int N) {
    setSize.assign(N, 1); numSets = N; rank.assign(N, 0);
    p.assign(N, 0); for (int i = 0; i < N; i++) p[i] = i; }
  int findSet(int i) { return (p[i] == i) ? i : (p[i] = findSet(p[i])); }
  bool isSameSet(int i, int j) { return findSet(i) == findSet(j); }
  void unionSet(int i, int j) { 
    if (!isSameSet(i, j)) { numSets--; 
    int x = findSet(i), y = findSet(j);
    // rank is used to keep the tree short
    if (rank[x] > rank[y]) { p[y] = x; setSize[x] += setSize[y]; }
    else                   { p[x] = y; setSize[y] += setSize[x];
                             if (rank[x] == rank[y]) rank[y]++; } } }
  int numDisjointSets() { return numSets; }
  int sizeOfSet(int i) { return setSize[findSet(i)]; }
};

vector<vii> AdjList;
vi taken;                                  // global boolean flag to avoid cycle
priority_queue<ii> pq;            // priority queue to help choose shorter edges

void process(int vtx) {    // so, we use -ve sign to reverse the sort order
  taken[vtx] = 1;
  for (int j = 0; j < (int)AdjList[vtx].size(); j++) {
    ii v = AdjList[vtx][j];
    if (!taken[v.first]) pq.push(ii(-v.second, -v.first));
} }                                // sort by (inc) weight then by (inc) id

int main() {
  int V, E, u, v, w;
  freopen("in_03.txt", "r", stdin);

  scanf("%d %d", &V, &E);
  // Kruskal's algorithm merged with Prim's algorithm
  AdjList.assign(V, vii());
  vector< pair<int, ii>> EdgeList;   // (weight, two vertices) of the edge
  for (int i = 0; i < E; i++) {
    scanf("%d %d %d", &u, &v, &w);            // read the triple: (u, v, w)
    EdgeList.push_back(make_pair(w, ii(u, v)));                // (w, u, v)
    AdjList[u].push_back(ii(v, w));
    AdjList[v].push_back(ii(u, w));
  }
  sort(EdgeList.begin(), EdgeList.end()); // sort by edge weight O(E log E)
                      // note: pair object has built-in comparison function

  int mst_cost = 0;
  UnionFind UF(V);                     // all V are disjoint sets initially
  for (int i = 0; i < E; i++) {                      // for each edge, O(E)
    pair<int, ii> front = EdgeList[i];
    if (!UF.isSameSet(front.second.first, front.second.second)) {  // check
      mst_cost += front.first;                // add the weight of e to MST
      UF.unionSet(front.second.first, front.second.second);    // link them
  } }                       // note: the runtime cost of UFDS is very light

  // note: the number of disjoint sets must eventually be 1 for a valid MST
  printf("MST cost = %d (Kruskal's)\n", mst_cost);



// inside int main() --- assume the graph is stored in AdjList, pq is empty
  taken.assign(V, 0);                // no vertex is taken at the beginning
  process(0);   // take vertex 0 and process all edges incident to vertex 0
  mst_cost = 0;
  while (!pq.empty()) {  // repeat until V vertices (E=V-1 edges) are taken
    ii front = pq.top(); pq.pop();
    u = -front.second, w = -front.first;  // negate the id and weight again
    if (!taken[u])                 // we have not connected this vertex yet
      mst_cost += w, process(u); // take u, process all edges incident to u
  }                                        // each edge is in pq only once!
  printf("MST cost = %d (Prim's)\n", mst_cost);

  return 0;
}

Dijkstra’s:
int main() {
  int V, E, s, u, v, w;
  vector<vii> AdjList;
  freopen("in_05.txt", "r", stdin);

  scanf("%d %d %d", &V, &E, &s);

  AdjList.assign(V, vii()); // assign blank vectors of pair<int, int>s to AdjList
  for (int i = 0; i < E; i++) {
    scanf("%d %d %d", &u, &v, &w);
    AdjList[u].push_back(ii(v, w));                              // directed graph
  }

  // Dijkstra routine
  vi dist(V, INF); dist[s] = 0;                    // INF = 1B to avoid overflow
  priority_queue< ii, vector<ii>, greater<ii>> pq; pq.push(ii(0, s));
                             // ^to sort the pairs by increasing distance from s
  while (!pq.empty()) {                                             // main loop
    ii front = pq.top(); pq.pop();     // greedy: pick shortest unvisited vertex
    int d = front.first, u = front.second;
    if (d > dist[u]) continue;   // this check is important, see the explanation
    for (int j = 0; j < (int)AdjList[u].size(); j++) {
      ii v = AdjList[u][j];                       // all outgoing edges from u
      if (dist[u] + v.second < dist[v.first]) {
        dist[v.first] = dist[u] + v.second;                 // relax operation
        pq.push(ii(dist[v.first], v.first));
  } } }  // note: this variant can cause duplicate items in the priority queue

  for (int i = 0; i < V; i++) // index + 1 for final answer
    printf("SSSP(%d, %d) = %d\n", s, i, dist[i]);

  return 0;
}

Ford Fulkerson's:
Ford-Fulkerson Algorithm
The following is simple idea of Ford-Fulkerson algorithm:
1) Start with initial flow as 0.
2) While there is a augmenting path from source to sink. 
           Add this path-flow to flow.
3) Return flow.

Maximum Matching:
Max Matching in a bipartite graph is equal to max flow from above
Primality Tests:
typedef long long ll;
typedef vector<int> vi;
typedef map<int, int> mii;

ll _sieve_size;
bitset<10000010> bs;   // 10^7 should be enough for most cases
vi primes;   // compact list of primes in form of vector<int>


// first part

void sieve(ll upperbound) {          // create list of primes in [0..upperbound]
  _sieve_size = upperbound + 1;                   // add 1 to include upperbound
  bs.set();                                                 // set all bits to 1
  bs[0] = bs[1] = 0;                                     // except index 0 and 1
  for (ll i = 2; i <= _sieve_size; i++) if (bs[i]) {
    // cross out multiples of i starting from i * i!
    for (ll j = i * i; j <= _sieve_size; j += i) bs[j] = 0;
    primes.push_back((int)i);  // also add this vector containing list of primes
} }                                           // call this method in main method

bool isPrime(ll N) {                 // a good enough deterministic prime tester
  if (N <= _sieve_size) return bs[N];                   // O(1) for small primes
  for (int i = 0; i < (int)primes.size(); i++)
    if (N % primes[i] == 0) return false;
  return true;                    // it takes longer time if N is a large prime!
}                      // note: only work for N <= (last prime in vi "primes")^2


// second part

vi primeFactors(ll N) {   // remember: vi is vector of integers, ll is long long
  vi factors;                    // vi `primes' (generated by sieve) is optional
  ll PF_idx = 0, PF = primes[PF_idx];     // using PF = 2, 3, 4, ..., is also ok
  while (N != 1 && (PF * PF <= N)) {   // stop at sqrt(N), but N can get smaller
    while (N % PF == 0) { N /= PF; factors.push_back(PF); }    // remove this PF
    PF = primes[++PF_idx];                              // only consider primes!
  }
  if (N != 1) factors.push_back(N);     // special case if N is actually a prime
  return factors;         // if pf exceeds 32-bit integer, you have to change vi
}


// third part

ll numPF(ll N) {
  ll PF_idx = 0, PF = primes[PF_idx], ans = 0;
  while (N != 1 && (PF * PF <= N)) {
    while (N % PF == 0) { N /= PF; ans++; }
    PF = primes[++PF_idx];
  }
  if (N != 1) ans++;
  return ans;
}

ll numDiffPF(ll N) {
  ll PF_idx = 0, PF = primes[PF_idx], ans = 0;
  while (N != 1 && (PF * PF <= N)) {
    if (N % PF == 0) ans++;                           // count this pf only once
    while (N % PF == 0) N /= PF;
    PF = primes[++PF_idx];
  }
  if (N != 1) ans++;
  return ans;
}

ll sumPF(ll N) {
  ll PF_idx = 0, PF = primes[PF_idx], ans = 0;
  while (N != 1 && (PF * PF <= N)) {
    while (N % PF == 0) { N /= PF; ans += PF; }
    PF = primes[++PF_idx];
  }
  if (N != 1) ans += N;
  return ans;
}

ll numDiv(ll N) {
  ll PF_idx = 0, PF = primes[PF_idx], ans = 1;             // start from ans = 1
  while (N != 1 && (PF * PF <= N)) {
    ll power = 0;                                             // count the power
    while (N % PF == 0) { N /= PF; power++; }
    ans *= (power + 1);                              // according to the formula
    PF = primes[++PF_idx];
  }
  if (N != 1) ans *= 2;             // (last factor has pow = 1, we add 1 to it)
  return ans;
}

ll sumDiv(ll N) {
  ll PF_idx = 0, PF = primes[PF_idx], ans = 1;             // start from ans = 1
  while (N != 1 && (PF * PF <= N)) {
    ll power = 0;
    while (N % PF == 0) { N /= PF; power++; }
    ans *= ((ll)pow((double)PF, power + 1.0) - 1) / (PF - 1);         // formula
    PF = primes[++PF_idx];
  }
  if (N != 1) ans *= ((ll)pow((double)N, 2.0) - 1) / (N - 1);        // last one
  return ans;
}

ll EulerPhi(ll N) {
  ll PF_idx = 0, PF = primes[PF_idx], ans = N;             // start from ans = N
  while (N != 1 && (PF * PF <= N)) {
    if (N % PF == 0) ans -= ans / PF;                // only count unique factor
    while (N % PF == 0) N /= PF;
    PF = primes[++PF_idx];
  }
  if (N != 1) ans -= ans / N;                                     // last factor
  return ans;
}

int main() {
  // first part: the Sieve of Eratosthenes
  sieve(10000000);                       // can go up to 10^7 (need few seconds)
  printf("%d\n", isPrime(2147483647));                        // 10-digits prime
  printf("%d\n", isPrime(136117223861LL));        // not a prime, 104729*1299709
  // second part: prime factors
  vi res = primeFactors(2147483647);   // slowest, 2147483647 is a prime
  for (vi::iterator i = res.begin(); i != res.end(); i++) printf("> %d\n", *i);

  res = primeFactors(136117223861LL);   // slow, 2 large pfactors 104729*1299709
  for (vi::iterator i = res.begin(); i != res.end(); i++) printf("# %d\n", *i);

  res = primeFactors(142391208960LL);   // faster, 2^10*3^4*5*7^4*11*13
  for (vi::iterator i = res.begin(); i != res.end(); i++) printf("! %d\n", *i);

  //res = primeFactors((ll)(1010189899 * 1010189899)); // "error"
  //for (vi::iterator i = res.begin(); i != res.end(); i++) printf("^ %d\n", *i);

  // third part: prime factors variants
  printf("numPF(%d) = %lld\n", 50, numPF(50)); // 2^1 * 5^2 => 3
  printf("numDiffPF(%d) = %lld\n", 50, numDiffPF(50)); // 2^1 * 5^2 => 2
  printf("sumPF(%d) = %lld\n", 50, sumPF(50)); // 2^1 * 5^2 => 2 + 5 + 5 = 12
  printf("numDiv(%d) = %lld\n", 50, numDiv(50)); // 1, 2, 5, 10, 25, 50, 6 divisors
  printf("sumDiv(%d) = %lld\n", 50, sumDiv(50)); // 1 + 2 + 5 + 10 + 25 + 50 = 93
  printf("EulerPhi(%d) = %lld\n", 50, EulerPhi(50)); // 20 integers < 50 are relatively prime with 50

  return 0;
}

Catalan Numbers dp
catalan[0]=catalan[1]=1;
for(int i=2;i<=n;i++):
    catalan[i]=0;
    for(int j=0;j<i;j++):
        catalan[i]+=catalan[j]*catalan[i-j-1];
// Property
C(n) =(1/(n+1)) * choose(2n, n);
C(n+1) = Summation(i = 0 to n) [C(i) * C(n-i)]

Extended Euclidean:
//store x,y and d as global variables
void extendedEuclid(int a, int b)
{
	if (b==0) {x=1; y=0; d=a; return; }	//base case
	extendedEuclid(b, a%b);
	int x1=y;
	int y1= x- (a/b)*y;
	x=x1;
	y=y1;
}
//Call extendedEuclid(a,b) for ax+by=c
//Multiply lhs and rhs by c/gcd(a,b)
//Multiply x, y by c/gcd(a,b)
//x=(x*c/gcd(a,b)) + (b/gcd(a,b))/n
//y=(y*c/gcd(a,b)) – (a/gcd(a,b))/n
/Find all values of n satisfying
Matrix Exponentiation:
ll ans[2][2];

void mat_expo(ll a[][2], ll n)
{
    ll i,j,k,b[2][2];
    ans[0][0]=1;
    ans[1][0]=0;
    ans[0][1]=0;
    ans[1][1]=1;
    while(n>0)
    {
        if (n%2==1)
        {
            for (i=0;i<2;i++)
            {
                for (j=0;j<2;j++)
                {
                    b[i][j]=0;
                    for (k=0;k<2;k++)
                    {
                        b[i][j]+=(ans[i][k]*a[k][j]);
                        b[i][j]%=mod;
                    }
                }
            }
            for (i=0;i<2;i++)
                for (j=0;j<2;j++)
                    ans[i][j]=b[i][j];
        }
        for (i=0;i<2;i++)
        {
            for (j=0;j<2;j++)
            {
                b[i][j]=0;
                for (k=0;k<2;k++)
                {
                    b[i][j]+=(a[i][k]*a[k][j]);
                    b[i][j]%=mod;
                }
            }
        }
        for (i=0;i<2;i++)
            for (j=0;j<2;j++)
                a[i][j]=b[i][j];
        n/=2;
    }
}


ll fibo_sum(ll n)
{
    ll a[2][2];
    a[0][0]=0;
    a[0][1]=1;
    a[1][0]=1;
    a[1][1]=1;
    mat_expo(a, n+1);
    return ans[1][1]%mod;
}
KMP Prefix Table:
void prefix()
{
    int i=1, j=0;
    f[0]=0;
    while(i<m)
    {
        if (s[i]==s[j])
        {
            f[i]=j+1;
            i++;
            j++;
        }
        else if (j>0)
            j=f[j-1];
        else
        {
            f[i]=0;
            i++;
        }
    }
}

void kmp()
{
    int i=0, j=0;
    prefix();
    while(i<n)
    {
        if (t[i]==s[j])
        {
            if (j==m-1)	//Match Found
            {
                v.pb(i);
                j=f[j];
                i++;
            }
            else
            {
                i++;
                j++;
            }
        }
        else if (j>0)
            j=f[j-1];
        else
            i++;
    }
}


LCA:
#include <cstdio>
#include <vector>
using namespace std;

#define MAX_N 1000

vector< vector<int>> children;

int L[2*MAX_N], E[2*MAX_N], H[MAX_N], idx;

void dfs(int cur, int depth) {
  H[cur] = idx;
  E[idx] = cur;
  L[idx++] = depth;
  for (int i = 0; i < children[cur].size(); i++) {
    dfs(children[cur][i], depth+1);
    E[idx] = cur;                              // backtrack to current node
    L[idx++] = depth;
  }
}

void buildRMQ() {
  idx = 0;
  memset(H, -1, sizeof H);
  dfs(0, 0);                       // we assume that the root is at index 0
}

int main() {
  children.assign(10, vector<int>());
  children[0].push_back(1); children[0].push_back(7);
  children[1].push_back(2); children[1].push_back(3); children[1].push_back(6);
  children[3].push_back(4); children[3].push_back(5);
  children[7].push_back(8); children[7].push_back(9);

  buildRMQ();
  return 0;
}

int main() {
  // same example as in chapter 2: segment tree
  int n = 7, A[] = {18, 17, 13, 19, 15, 11, 20};
  RMQ rmq(n, A);
  for (int i = 0; i < n; i++)
    for (int j = i; j < n; j++)
      printf("RMQ(%d, %d) = %d\n", i, j, rmq.query(i, j));

  return 0;
}

Square Root Decompisiton:
#define N 131072
#define BLOCK 318

struct data
{
    int L, R, I, K;
}d[100000];

bool cmp(data x, data y)
{
    if(x.L/BLOCK != y.L/BLOCK)
		return x.L/BLOCK < y.L/BLOCK;
	return x.R < y.R;
}

sort(d, d+q, cmp);
        int ans[q];
        int currentL=-1, currentR=-1;
        z=0;
        for (i=0;i<q;i++)
        {
            int L=d[i].L, R=d[i].R;
            if (L>currentR || R<currentL)
            {
                for (j=currentL;j<=currentR && j>=0;j++)
                {
                    c[b[j]]--;
                    if (c[b[j]]==0)
                        delet(b[j]);
                }
                for (j=L;j<=R;j++)
                {
                    if (c[b[j]]==0)
                        insert(b[j]);
                    c[b[j]]++;
                }
                currentL=L; currentR=R;
            }
            else
            {
                while(currentL < L) 
                {
                    c[b[currentL]]--;
                    if (c[b[currentL]]==0)
                        delet(b[currentL]);
                    currentL++;
                }
                while(currentR > R) 
                {
                    c[b[currentR]]--;
                    if (c[b[currentR]]==0)
                        delet(b[currentR]);
                    currentR--;
                }
                while(currentL > L)
                {
                    if (c[b[currentL-1]]==0)
                        insert(b[currentL-1]);
                    c[b[currentL-1]]++;
                    currentL--;
                }
                while(currentR < R) 
                {
                    if (c[b[currentR+1]]==0)
                        insert(b[currentR+1]);
                    c[b[currentR+1]]++;
                    currentR++;
                }
            }
            if (z<d[i].K)
                ans[d[i].I]=-1;
            else
                ans[d[i].I]=queryk(d[i].K);
        }
        for (i=0;i<q;i++)
            printf("%d\n",ans[i]);
}


Convex Hull
typedef double coord_t; // coordinate type
typedef double coord2_t;  // must be big enough to hold 2*max(|coordinate|)^2
struct Point {
	coord_t x, y;

	bool operator <(const Point &p) const {
		return x < p.x ||(x == p.x && y < p.y);
	}
};

coord2_t cross(const Point &O, const Point &A, const Point &B){
	return(A.x - O.x) *(B.y - O.y) -(A.y - O.y) *(B.x - O.x);
}

// Returns a list of points on the convex hull in counter-clockwise order.
// Note: the last point in the returned list is the same as the first one.
vector<Point> convex_hull(vector<Point> P)
{
	int n = P.size(), k = 0;
	vector<Point> H(2*n);

	// Sort points lexicographically
	sort(P.begin(), P.end());

	// Build lower hull
	for(int i = 0; i < n; ++i){
		while(k >= 2 && cross(H[k-2], H[k-1], P[i]) <= 0) k--;
		H[k++] = P[i];
	}

	// Build upper hull
	for(int i = n-2, t = k+1; i >= 0; i--){
		while(k >= t && cross(H[k-2], H[k-1], P[i]) <= 0) k--;
		H[k++] = P[i];
	}

	H.resize(k-1);
	return H;
}

Shoelace
double polygonArea(double X[], double Y[], int n){
    double area = 0.0;
    // Calculate value of shoelace formula
    int j = n - 1;
    for(int i = 0; i < n; i++){
        area+=(X[j] + X[i])*(Y[j] - Y[i]);
        j = i;  //j is previous vertex to i
    }
    return abs(area / 2.0);
}

Overlapping Circles / Rectangles area, Area of Union of Circles
//CIRCLE RECTANGLE
long double section(long double h, long double r = 1){
    // returns the positive root of intersection of line y = h with circle centered at the origin and radius r
    assert(r >= 0); // assume r is positive, leads to some simplifications in the formula below(can factor out r from the square root)
    return(h < r)?sqrt(r * r - h * h) : 0; 
} 
long double g(long double x, long double h, long double r = 1){
    // indefinite integral of circle segment
    return .5f *(sqrt(1 - x * x /(r * r)) * x * r + r * r * asin(x / r) - 2 * h * x); 
}
long double area(long double x0, long double x1, long double h, long double r){
    // area of intersection of an infinitely tall box with left edge at x0, right edge at x1, bottom edge at h and top edge at infinity, with circle centered at the origin with radius r
    if(x0 > x1)
        std::swap(x0, x1); // this must be sorted otherwise we get negative area
    long double s = section(h, r);
    return g(max(-s, min(s, x1)), h, r) - g(max(-s, min(s, x0)), h, r); // integrate the area
}
long double area(long double x0, long double x1, long double y0, long double y1, long double r){
    // area of the intersection of a finite box with a circle centered at the origin with radius r
    if(y0 > y1)
        std::swap(y0, y1); // this will simplify the reasoning
    if(y0 < 0){
        if(y1 < 0)
            return area(x0, x1, -y0, -y1, r); // the box is completely under, just flip it above and try again
        else
            return area(x0, x1, 0, -y0, r) + area(x0, x1, 0, y1, r); // the box is both above and below, divide it to two boxes and go again
    } else {
        assert(y1 >= 0); // y0 >= 0, which means that y1 >= 0 also(y1 >= y0) because of the swap at the beginning
        return area(x0, x1, y0, r) - area(x0, x1, y1, r); // area of the lower box minus area of the higher box
    }
}
long double area(long double x0, long double x1, long double y0, long double y1, long double cx, long double cy, long double r){
    // area of the intersection of a general box with a general circle
    x0 -= cx; x1 -= cx;
    y0 -= cy; y1 -= cy;
    // get rid of the circle center
    return area(x0, x1, y0, y1, r);
}

//CIRCLE CIRCLE
function areaOfIntersection(x0, y0, r0, x1, y1, r1){
  var rr0 = r0*r0;
  var rr1 = r1*r1;
  var c = Math.sqrt((x1-x0)*(x1-x0) +(y1-y0)*(y1-y0));
  var phi =(Math.acos((rr0+(c*c)-rr1) /(2*r0*c)))*2;
  var theta =(Math.acos((rr1+(c*c)-rr0) /(2*r1*c)))*2;
  var area1 = 0.5*theta*rr1 - 0.5*rr1*Math.sin(theta);
  var area2 = 0.5*phi*rr0 - 0.5*rr0*Math.sin(phi);
  return area1 + area2;
}

Wilson's Theorem:
	A number n>1 is prime if and only if 
		(n-1)! mod(p) = -1


#include <bits/stdc++.h>
using namespace std;
 
// define character size
// currently Trie supports lowercase English characters (a - z)
#define CHAR_SIZE 26
 
// A Trie node
struct Trie
{
    bool isLeaf;    // true when node is a leaf node
    Trie* character[CHAR_SIZE];
};
 
// Function that returns a new Trie node
Trie* getNewTrieNode()
{
    Trie* node = new Trie;
    node->isLeaf = false;
 
    for (int i = 0; i < CHAR_SIZE; i++)
        node->character[i] = NULL;
 
    return node;
}
// Iterative function to insert a string in Trie.
void insert(Trie*& head, char* str)
{
    // start from root node
    Trie* curr = head;
    while (*str)
    {
        // create a new node if path doesn't exists
        if (curr->character[*str - 'a'] == NULL)
            curr->character[*str - 'a'] = getNewTrieNode();
 
        // go to next node
        curr = curr->character[*str - 'a'];
 
        // move to next character
        str++;
    }
    // mark current node as leaf
    curr->isLeaf = true;
}
// Iterative function to search a string in Trie. It returns true
// if the string is found in the Trie, else it returns false
bool search(Trie* head, char* str)
{
    // return false if Trie is empty
    if (head == NULL)
        return false;
    Trie* curr = head;
    while (*str)
    {
        // go to next node
        curr = curr->character[*str - 'a'];
        // if string is invalid (reached end of path in Trie)
        if (curr == NULL)
            return false;
        // move to next character
        str++;
    }
    // if current node is a leaf and we have reached the
    // end of the string, return true
    return curr->isLeaf;
}
// returns true if given node has any children
bool haveChildren(Trie const* curr)
{
    for (int i = 0; i < CHAR_SIZE; i++)
        if (curr->character[i])
            return true;    // child found
 
    return false;
}
// Recursive function to delete a string in Trie.
bool deletion(Trie*& curr, char* str)
{
    // return if Trie is empty
    if (curr == NULL)
        return false;
    // if we have not reached the end of the string
    if (*str)
    {
        // recurse for the node corresponding to next character in 
        // the string and if it returns true, delete current node 
        // (if it is non-leaf)
        if (curr != NULL && curr->character[*str - 'a'] != NULL &&
            deletion(curr->character[*str - 'a'], str + 1) &&
            curr->isLeaf == false)
        {
            free(curr);
            curr = NULL;
            return true;
        }
    }
    // if we have reached the end of the string
    if (*str == '\0' && curr->isLeaf)
    {
        // if current node is a leaf node and don't have any children
        if (!haveChildren(curr))
        {
            free(curr); // delete current node
            curr = NULL;
            return true; // delete non-leaf parent nodes
        }
        // if current node is a leaf node and have children
        else
        {
            // mark current node as non-leaf node (DON'T DELETE IT)
            curr->isLeaf = false;
            return false;       // don't delete its parent nodes
        }
    }
    return false;
}
#define fast_io ios_base::sync_with_stdio(false);cin.tie(NULL)
