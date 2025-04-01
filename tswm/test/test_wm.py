import itertools

import numpy as np

import errors
import util
import wm


def test_algo(cls, verbose=False, only_skipped=False, force=False):
    cls_name = cls.__name__
    matrix = cls.get_test_matrix()
    matrix = [[(k, v) for v in vs] for k, vs in matrix.items()]
    if not matrix:
        print(f"Skipped {cls_name}")
        return

    matrix = itertools.product(*matrix)
    rng = util.Random()
    key = rng.bytes(8)
    num_passed = 0
    num_failed = 0
    num_skipped = 0
    error = None

    for c in matrix:
        c = dict(c)
        c_str = str(c)
        if not only_skipped:
            print(cls_name, c_str)

        wm_len, cont_len = c.pop("wm_cont_len")
        c["wm_len"] = wm_len
        wm = rng.bits(wm_len)
        cont = rng.signal(cont_len, noise_var=5, dtype=np.int8)

        try:
            worker = cls.new(**c, key=key, verbose=verbose)
            worker.set_container(cont)
            worker.set_watermark(wm)

            worker.embed()
            w = worker.extract()
            # print(repr(wm), wm)
            # print(repr(w), w)
            assert np.array_equal(w, wm)

            if worker.max_restore_error is not None:
                assert abs(worker.restored - cont).max() <= worker.max_restore_error

            num_passed += 1
            if not only_skipped:
                print("\33[1;32mOK\33[0m")
        except (
            errors.StaticError,
            errors.CantEmbed,
        ) as e:
            num_skipped += 1
            if only_skipped:
                print(cls_name, c_str)
            print(f"\33[1;33mSkipped\33[0m {repr(e)}")
        except Exception as e:
            num_failed += 1
            if only_skipped:
                print(cls_name, c_str)
            print("\33[1;31mFailed\33[0m")
            if not force:
                error = e
                break

    print()
    print(cls_name)
    print(f"{num_passed=}")
    print(f"{num_failed=}")
    print(f"{num_skipped=}")
    print()

    if error is not None:
        raise error


if __name__ == "__main__":
    test_algo(wm.RCMEmbedder)
    test_algo(wm.ITBEmbedder)
    test_algo(wm.LCBEmbedder)
    test_algo(wm.LSBEmbedder)
    test_algo(wm.NeighorsPEE)
    test_algo(wm.SiblingChannelPEE)
