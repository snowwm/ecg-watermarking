import itertools

import util
from wm_lsb import LSBEmbedder
from wm_de import DEEmbedder

params = [
    [
        ("key", util.random_bytes(5)),
    ],
    [
        ("carr", util.random_bytes(2400)),
        ("carr", util.random_bytes(100000)),
    ],
    [
        ("wm", util.random_bytes(10)),
        ("wm", util.random_bytes(100)),
    ],
    [
        ("shuffle", False),
        ("shuffle", True),
    ],
    [
        ("contiguous", False),
        ("contiguous", True),
    ],
    [
        ("redundancy", 1),
        ("redundancy", 2),
        ("redundancy", 3),
    ],
    [
        ("block_len", 1),
        ("block_len", 2),
        ("block_len", 3),
    ],
]

de_params = [
    [
        ("de_rand_shift", False),
        ("de_rand_shift", True),
    ],
    [
        ("de_shift", 1),
        ("de_shift", 3),
        ("de_shift", 5),
    ],
]
    
cases = itertools.product(*params)
de_cases = itertools.product(*(params + de_params))

def test(cases, cls):
    for c in cases:
        c = dict(c)
        carr = util.np.frombuffer(c.pop("carr"), dtype=util.np.int8)
        wm = util.to_bits(c.pop("wm"))
        
        worker = cls(**c, debug=False)
        worker.set_container(carr)
        worker.set_watermark(wm)
        
        c.pop("key")
        print(c)
        
        try:
            worker.embed()
            w1 = worker.extract(len(wm))
            assert (w1 == wm).all()
            
            if cls is DEEmbedder:
                max_err = worker.de_n * 2
                assert abs(worker.restored - carr).max() <= max_err
              
            if not worker.shuffle:
                w2 = worker.extract()
                assert (w2[:len(wm)] == wm).all()
                
            print("OK")
        except Exception as e:
            if e.args != ("Insufficient container length or range for this watermark",):
                raise
            print("Skipped")

def test_manual():
    wm = util.to_bits("a")
    e = DEEmbedder(key="ab", contiguous=False, block_len=1, redundancy=1, shuffle=False, debug=True)
    e.set_container(range(20))
    e.set_watermark(wm)
    e.embed()
    print()
    print()
    e.extract(len(wm))

    assert (e.extracted == wm).all()
    
    
#test_manual()
test(cases, LSBEmbedder)
test(de_cases, DEEmbedder)
