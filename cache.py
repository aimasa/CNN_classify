# -*- coding: utf-8 -*-
import hashlib
import os
import pickle

import normal_param

if not os.path.exists(normal_param.cache_root_dir):
    os.makedirs(normal_param.cache_root_dir)


def md5(s):
    m = hashlib.md5()
    m.update(s.encode("utf-8"))
    return m.hexdigest()


def cache_key(f, *args, **kwargs):
    s = '%s-%s-%s' % (f.__name__, str(args), str(kwargs))
    return os.path.join(normal_param.cache_root_dir, '%s.dump' % md5(s))


def cache(f):
    def wrap(*args, **kwargs):
        fn = cache_key(f, *args, **kwargs)
        if os.path.exists(fn):
            print('loading cache')
            with open(fn, 'rb') as fr:
                return pickle.load(fr)

        obj = f(*args, **kwargs)

        with open(fn, 'wb') as fw:
            pickle.dump(obj, fw)
        return obj
    return wrap