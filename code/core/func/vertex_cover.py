class Find_h4(object):

    def __init__(self):
        self.maxn = 120  # bigger than agent total count
        self.gr = [[0] * self.maxn for i in range(self.maxn)]


    def isCover(self, V, k, E):

        Set = (1 << k) - 1  #1*2^k - 1
        limit = (1 << V)    #1*2^v

        vis = [[None] * self.maxn for i in range(self.maxn)]

        while (Set < limit):

            vis = [[0] * self.maxn for i in range(self.maxn)]
            cnt = 0

            j = 1
            v = 1
            while (j < limit):
                if (Set & j):
                    for k in range(1, V + 1):
                        if (self.gr[v][k] and not vis[v][k]):
                            vis[v][k] = 1
                            vis[k][v] = 1
                            cnt += 1
                j = j << 1
                v += 1

            if (cnt == E):
                return True

            c = Set & -Set
            r = Set + c
            Set = (((r ^ Set) >> 2) // c) | r
        return False


    # Returns answer to graph stored in gr[][]
    def findMinCover(self, n, m):
        # Binary search the answer
        left = 1
        right = n #vertecies size
        while (right > left):
            mid = (left + right) >> 1
            if (self.isCover(n, mid, m) == False):
                left = mid + 1
            else:
                right = mid

        # at the end of while loop both left and
        # right will be equal,/ as when they are
        # not, the while loop won't exit the
        # minimum size vertex cover = left = right
        return left


    # Inserts an edge in the graph
    def insertEdge(self, u):

        self.gr[u[0]][u[1]] = 1
        self.gr[u[1]][u[0]] = 1

        #self.gr[u][v] = 1
        #self.gr[v][u] = 1  # Undirected graph


    def insert_all_edges(self, edges):
        for i in range(len(edges)):
            self.insertEdge(edges[i])

    def find_mincover(self, v, e):
        self.gr = [[0] * self.maxn for i in range(self.maxn)]
        self.insert_all_edges(e)
        h = self.findMinCover(len(v), len(e))
        return h

    '''
    # Let us create another graph
    gr = [[0] * maxn for i in range(maxn)]
    
    #
    #     2 ---- 4 ---- 6
    #     /|     |
    # 1 |     | vertex cover = {2, 3, 4}
    #     \ |     |
    #     3 ---- 5
    V = 6
    E = 7
    insertEdge(1, 2)
    insertEdge(1, 3)
    insertEdge(2, 3)
    insertEdge(2, 4)
    insertEdge(3, 5)
    insertEdge(4, 5)
    insertEdge(4, 6)
    
    print("Minimum size of a vertex cover = ", findMinCover(V, E))'''