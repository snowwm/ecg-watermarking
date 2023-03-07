import itertools

import numpy as np

import errors
import util
import wm


def test_algo(cls, debug=False, only_skipped=False, force=False):
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
        wm = util.to_bits(rng.bytes((wm_len + 7) // 8))[:wm_len]
        # We leave some space between the signal extrema and dtype limits.
        # This space is needed for expansion methods.
        cont = rng.signal(cont_len, noise_var=5, dtype=np.int8)

        try:
            worker = cls(**c, key=key, debug=debug)
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
                print("OK")
        except (
            errors.InvalidConfig,
            errors.InsufficientContainerRange,
            errors.CantEmbed,
        ) as e:
            num_skipped += 1
            if only_skipped:
                print(cls_name, c_str)
            print(f"Skipped {repr(e)}")
        except Exception as e:
            num_failed += 1
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
    test_algo(wm.DEEmbedder)
    test_algo(wm.ITBEmbedder)
    test_algo(wm.LCBEmbedder)
    test_algo(wm.LSBEmbedder)
    test_algo(wm.NeighorsPEE)
    test_algo(wm.SiblingChannelPEE)
