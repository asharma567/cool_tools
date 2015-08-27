from futures import ThreadPoolExecutor
import cPickle


def multithread_map(work_func, work_list, num_workers):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(work_func, work_list))

def memoize(fn):
    try: 
        with open('memoized_stored_results_' + str(fn).split()[1] + '.pkl', 'rb') as f:
            stored_results = cPickle.load(f)
    except IOError:
        stored_results = {}
        
    def memoized(*args):
        if args in stored_results:
            result = stored_results[args]
        else:
            result = stored_results[args] = fn(*args)
            with open('memoized_stored_results_' + str(fn).split()[1] + '.pkl', 'wb') as f:
	            cPickle.dump(stored_results, f)
        return result
    return memoized

def make_verbose(fn):
    '''
    used for debugging purposes 
    to take note of args being 
    passed the function each call
    '''

    def verbose(*args):
        print '%s(%s)' % (fn.__name__, ', '.join(repr(arg) for arg in args))
        return fn(*args)
    return verbose
