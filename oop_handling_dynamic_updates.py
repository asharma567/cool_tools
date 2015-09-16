'''
Have you ever encountered a problem with dynamically assigning 
properties to an object? 

This is a great way to keep track of those new properties where 
dict_of_names is a hash-map of property_name : value
'''

import types
import inspect

class MyClass(object):
    
    def __init__(self):
        self.a = 'a'
        self.b = 'b'
        self.update_dict()

    # you can also write this as a classmethod or staticmethod 
    # to update all instances with this variable 
    def add_property(self, attr_name, attr_val):
        setattr(self, attr_name, types.StringType(attr_val))
        self.update_dict()
        pass
    
    def update_dict(self):
        attributes = inspect.getmembers(self, lambda a: not(inspect.isroutine(a)))
        current_properties_of_obj = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__') )]
        self.dict_of_names = dict(current_properties_of_obj)
        
        #hack, need to fix
        self.dict_of_names.pop('dict_of_names', None)
        pass

if __name__ == '__main__':
    instantiated_obj1 = MyClass()
    instantiated_obj1.add_property('name_of_obj', 'path')

    #difference here is it won't update
    instantiated_obj1.name_of_obj_alternative = 'path'

    print instantiated_obj1.dict_of_names
    instantiated_obj1.update_dict()
    print instantiated_obj1.dict_of_names