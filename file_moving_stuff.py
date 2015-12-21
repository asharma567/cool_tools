'''
display a file on 
'''
from IPython.display import Image
Image(filename='test.png')

import os

'''
usage
'''
find('*.txt', '/path/to/dir')


#find exact match
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

#And this will find all matches:
def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result

#And this will match a pattern:
import os, fnmatch
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result




shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
