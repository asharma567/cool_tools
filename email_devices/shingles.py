
str1 = "ABRACADABRA"
str2 = "BRICABRAC"

def shingle(str):
    s = set()
    for i,c in enumerate(str):
        if i < len(str) - 1:
            shing = c+str[i+1]
            s.add(shing)
    return s

shingle1 = shingle(str1)
shingle2 = shingle(str2)

print('Number of 2-shinlges in ABRACADABRA = '+str(len(shingle1)));
print('Number of 2-shinlges in BRICABRAC = '+str(len(shingle2)));

intersect = shingle1.intersection(shingle2)
union = shingle1.union(shingle2)
print intersect
print union
jaccard = len(intersect)/float(len(union))
print('Jaccard index = '+str(jaccard))