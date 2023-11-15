MAX=1000
class Node:
    def __init__(self, index,cost,visited=False,solved=False,and_map=False):
        self.index=index
        self.cost=cost
        self.visited=visited
        self.solved=solved
        self.and_map= and_map;
        self.children=None
        self.path = False

    def __str__(self):
        return f'{self.index}: {self.cost}'

    def set_children(self,ch):
        self.children=ch


#Use case 1
adj=[]
EDGE=1 #g cost of edge,
n_nodes = 10
#heuristic costs
cost=[None,0,6,12,9,5,7,1,4,4,1]
children = {1:[2,3,4], 2:[5,6], 3:[7], 4:[8,9], 5:[10]}

adj = [Node(i, cost[i]) for i in range(n_nodes+1)]
for p,ch in children.items():
    ch_nodes = [adj[c] for c in ch]
    adj[p].set_children(ch_nodes)

and_edges={}

and_edges[adj[1]]=(adj[2],adj[3])
and_edges[adj[4]]=(adj[8],adj[9])

for a in and_edges.values():
    for node in a: node.and_map=True

for a in adj:
    if not a.children: a.solved=True
    #print(f'{a.index} and {a.and_map} solved {a.solved}')


def Cost(c,head):
    if c.and_map: #if child is and edge
        ae=and_edges[head]
        cc=sum(aek.cost+EDGE for aek in ae)
        solved = all([aek.solved for aek in ae])
        return ae, cc, solved
    else: #or edge
        cc = c.cost+EDGE
        return (c,), cc, c.solved


def aostarUtil(head):

    ch={}
    #check if head has any and edges
    for c in head.children:
        nn, cc, solved = Cost(c,head)
        if solved: 
            head.solved=solved
        print(head.index, 'child', c.index, cc, head.solved)
        c.path=False
        ch[nn]=cc

   #head is explored now update the best value of head
    head.cost= min(ch.values())
    best=min(ch,key=ch.get) #set best move to min cost
    if head.solved:
        for b in best:
            b.path=True

    if not head.visited:
        head.visited=True; head.path=True;
        return

    print(f'Explore Head: {head.index} updated cost {head.cost} solved {head.solved}')
    print(type(best))
    for b in best:
        b.path=True
        if not b.solved: aostarUtil(b);



def aostar(head):
    iter = 0
    while not head.solved and iter<MAX:
        print(f'\n  **Iteration {iter}')
        aostarUtil(head)
        iter+=1


aostar(adj[1])

print(f'Cost of Solution : {adj[1].cost}')
print('Path')
for a in adj:
    if a.path: print(f'{a.index} cost: {a.cost}')
