from futures import ThreadPoolExecutor
import cPickle


def multithread_map(fn, work_list, num_workers=50):
    '''
    spawns a threadpool and assigns num_workers to some 
    list, array, or any other container. Motivation behind 
    this was for functions that involve scraping.
    '''
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(fn, work_list))

def memoize(fn):
    '''
    memoization for any function i.e. checks a hash-map to 
    see if the same work's already been done avoid unnecessary 
    computation
    '''
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


class TimedOutExc(Exception):
    pass

def deadline(timeout, *args):
    '''
    give a deadline to subject function, usage:
    @deadline(900)
    function will raise error TimedOutExc after 900 seconds
    '''
    def decorate(f):
        def handler(signum, frame):
            raise TimedOutExc()

        def new_f(*args):
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)
            return f(*args)
            signa.alarm(0)

        new_f.__name__ = f.__name__
        return new_f
    return decorate
