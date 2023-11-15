class Node:
    def __init__(self, value,coord):
        self.value = value
        self.coord = coord
        self.g=0; self.h=0 #initialization
        self.parent=None
    
    #This function may be used for debugging 
    def __str__(self):
        s = f'{self.coord} f= {self.g+self.h:0.2f}, g={self.g:0.2f}, h= {self.h:0.2f}' 
        return s
    
    def move_cost(self, other):
        return 1 


def children(current_node,grid):
    x,y = current_node.coord
    links = [(x-1, y),(x,y-1),(x,y+1),(x+1,y),(x+1,y+1),(x-1,y-1),(x+1,y-1),(x-1,y+1)]
             #(x+1,y+1),(x-1,y-1),(x+1,y-1),(x-1,y+1)#diagonal moves, add in later.
    
    valid_links=[link for row in grid for link in row if link.value!=0]
    valid_children = [link for link in valid_links if link.coord in links]
        
    return valid_children


# for 4 moves
def manhattan(node, goal):
    #manhattan distance 
    xN,yN = node.coord
    xG,yG = goal.coord
    h = abs(xN-xG) + abs(yN-yG)
    return h

# for 8 moves
def diagonal(node, goal):
    xN,yN = node.coord
    xG,yG  = goal.coord
    dx = abs(xN - xG)
    dy = abs(yN- yG)
    return (dx + dy) - min(dx, dy)


def aStar(start, goal, grid):
    #The open and closed lists
    OPEN = list(); CLOSED=list()
    #Set current node to start node
    current = start
    #Add start node to the OPEN list
    OPEN.append(current)
    i=0 # for tracing purpose
    
    #While the open list is not empty
    while OPEN:
        print('Iteration ',i) # for tracing purpose
        i+=1 # for tracing purpose
        
        #Find the item in the open set with the lowest g + h score
        current = min(OPEN, key=lambda o:o.g + o.h)       
        # print statements for tracing purpose
        print('Current Node', current)
        
        
        #print('Contents in OPEN: ')
        #for n in OPEN: print(n)
        # ***************** #
        
        #If it is the item we want, retrace the path and return it
        if current == goal: # trace path by using parent link
            path = []
            while current.parent:
                path.append(current)
                current = current.parent
            path.append(current)
            return path[::-1] 
        
        
        #Move item from OPEN to CLOSED
        OPEN.remove(current); CLOSED.append(current)
        
        #Loop through the node's children/siblings
        for node in children(current,grid):
            #If it is already in the closed list and updated cost is lower, move to OPEN list
            if node in CLOSED:
                new_cost = current.g + current.move_cost(node)
                if new_cost<=node.g: 
                    OPEN.append(node);CLOSED.remove(node)

            #Otherwise if it is already in the open set
            elif node in OPEN:
                #Check if we beat the G score 
                new_cost = current.g + current.move_cost(node)
                if new_cost<=node.g:
                    #If so, update the node to have a new parent
                    node.g = new_cost
                    node.parent = current
            else:
                #If it isn't in the open set, calculate the G and H score for the node      
                node.g = current.g + current.move_cost(node)
                node.h = diagonal(node, goal) 
                #Set the parent to our current item
                node.parent = current
                #Add it to the list
                OPEN.append(node)
    #If no path found
    return None


#use case 1
grid = [[1,1,1,1], #1-not blocked, 0 -  not blocked
        [1,1,1,1],
        [1,1,1,1],
        [1,1,0,0],
        [1,1,0,1]]

#Convert all the points to instances of node
for x in range(len(grid)):
    for y in range(len(grid[x])):
        grid[x][y] = Node(grid[x][y],(x,y))

start = grid[4][0]
goal = grid [0][3]


#Driver Code
path = aStar(start,goal ,grid)
if path:
    print("** Path ** ")
    for p in path:
        print(p.coord, end=" ")
else:
    print("No path found")


