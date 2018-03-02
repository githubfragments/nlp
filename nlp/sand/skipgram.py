import numpy as np

def _skip_table(n, k):
    items = []
    
    if n==1:
        for s in xrange(k+1):
            item = (s, [s])
            items.append(item)
        return items
        
    subitems = _skip_table(n-1, k)
    
    for subitem in subitems:
        for s in xrange(k+1):
            item = (s+subitem[0], [s] + subitem[1])
            items.append(item)
    
    return items
            
def skip_table(n, k):
    return _skip_table(n-1,k)
    
st = skip_table(2,2)
