from collections import Counter
import math
import pandas as pd
df = pd.read_csv('playTennis.csv')
n = len(df)


def entropy(ls):
    counts = Counter(x for x in ls)
    total = len(ls)
    probs = [x/total for x in counts.values()]
    E = sum(-p*math.log(p,2) for p in probs)
    return E

def inf_gain(df,a,target):
    df_split = df.groupby(a)
    df_aggr = df_split.agg({target:[entropy, lambda x: len(x)/n]})[target]
    df_aggr.columns = ['Entropy','Proportions']
    #print(df_aggr)
    new_E = sum(df_aggr['Entropy']*df_aggr['Proportions'])
    old_E = entropy(df[target])
    return old_E - new_E



def id3(df,target,attr,def_class=None,def_attr='S'):
    pn = Counter(x for x in df[target])
    print(f'\n** {pn} **')
    if len(pn) == 1:
        return next(iter(pn))
    elif df.empty or (not attr):
        return def_class
    else:
        def_class = max(pn,key = pn.get)
        gains={}
        for a in attr:
            gains[a] = inf_gain(df,a,target)
            print(f'Inf Gains on {a}:{gains[a]:0.03f}')
            
        best = max(gains, key=gains.get)
        print(f'best {best}')
            
        tree = {best:{}}
        attr.remove(best)
        
        for av,data in df.groupby(best):
            #print(av)
            #print(data)
            subtree = id3(data,target,attr,def_class,best)
            tree[best][av]=subtree
            print(f'best {best} {av}')
            print(tree)
            
    return tree



attr = list(df.columns)
attr.remove('PlayTennis')
print('Predicting Attributes',attr)
tree=id3(df,'PlayTennis',attr)



from pprint import pprint
print('Tree')
pprint(tree)
ba = next(iter(tree))
print(f'Best Attribute: {ba}')
print(f'Tree keys: \n {tree[ba].keys()}')
