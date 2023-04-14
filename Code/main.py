from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from algo_base import AlgoBase
from db import Database
import errors
from records import load_record_file
import util
import wm


def make_parser():
    parser = ArgumentParser(description="Medical timeseries watermarking tool.")
    parser.add_argument("-v", "--verbose", action="store_true", dest="_verbose")
    parser.add_argument("-d", "--data-file", type=Path)
    parser.add_argument("--data-all", action="store_true")
    parser.add_argument("--seed")
    subp = parser.add_subparsers(required=True)

    p = subp.add_parser("test", help="Run tests")
    p.add_argument("-a", "--algo")
    p.add_argument("-s", "--only-skipped", action="store_true")
    p.set_defaults(func=test)

    p = subp.add_parser("rand-bytes", help="Generate random bytes")
    p.add_argument("num_bytes", type=int)
    p.add_argument("out_file", type=Path)
    p.set_defaults(func=rand_bytes)

    p = subp.add_parser("info", help="Print some info about EDF(+) file(s)")
    p.add_argument("rec_files", type=Path, nargs="+")
    p.add_argument("-c", "--channel", type=int, default=-1)
    p.add_argument("-r", "--reconstruct", action="store_true")
    p.set_defaults(func=file_info)

    p = subp.add_parser("add-noise")
    p.add_argument("var", type=float)
    p.add_argument("rec_in", type=Path)
    p.add_argument("rec_out", type=Path, nargs="?")
    p.set_defaults(func=add_noise)

    p = subp.add_parser("embed")
    p.add_argument("rec_in", type=Path)
    p.add_argument("wm_in", type=Path)
    p.add_argument("rec_out", type=Path, nargs="?")
    add_common_args(p)
    p.set_defaults(func=do_wm, action="embed")

    p = subp.add_parser("extract")
    p.add_argument("rec_in", type=Path)
    p.add_argument("wm_out", type=Path)
    p.add_argument("rec_out", type=Path, nargs="?")
    p.add_argument("-l", "--wm-len", type=int, required=True)
    add_common_args(p)
    p.set_defaults(func=do_wm, action="extract")

    p = subp.add_parser("check")
    p.add_argument("rec_in", type=Path)
    p.add_argument("wm_in", type=Path)
    add_common_args(p)
    p.set_defaults(func=do_wm, action="check")

    p = subp.add_parser("research")
    p.add_argument("rec_files", type=Path, nargs="+")
    p.add_argument("-w", "--wm-file", type=Path)
    p.add_argument("-l", "--wm-len", type=int)
    p.add_argument("-n", "--noise-var", type=float)
    add_common_args(p)
    p.set_defaults(func=do_wm, action="research")

    return parser


def add_common_args(p):
    # Common WM params.
    p.add_argument("-c", "--channel", type=int, default=-1)
    p.add_argument(
        "-a",
        "--algo",
        choices=("de", "itb", "lcb", "lcbp", "lsb", "pee"),
        default="lsb",
    )

    # Common embedder params.
    p.add_argument("-k", "--key", dest="_key")
    p.add_argument("-s", "--shuffle", action="store_true", dest="_shuffle")
    p.add_argument("-C", "--non-contiguous", action="store_false", dest="_contiguous")
    p.add_argument("-r", "--redundancy", type=int, dest="_redundancy")
    p.add_argument("-b", "--block-len", type=int, dest="_block_len")

    # DE params.
    p.add_argument("--de-shift", type=int, dest="_de_shift")
    # Here we need to explicitly specify default=None, otherwise store_true sets it to False.
    p.add_argument(
        "--de-rand-shift", action="store_true", default=None, dest="_de_rand_shift"
    )
    p.add_argument("--de-skip", action="store_true", default=None, dest="_de_skip")

    # LCB params.
    p.add_argument("--coder", choices=["rle", "huff", "mock"], dest="_coder")

    # LSB params.
    p.add_argument("--lsb-lowest-bit", type=int, dest="_lsb_lowest_bit")

    # PEE params.
    p.add_argument("--predictor", choices=["neigh", "chan"], dest="_predictor")

    # Predictor params.
    p.add_argument("--left-neighbors", type=int, dest="_left_neighbors")
    p.add_argument("--right-neighbors", type=int, dest="_right_neighbors")

    # Coder params.
    p.add_argument("--coder-transform", choices=["dct", "dwt"], dest="_coder_transform")
    p.add_argument("--rle-bitness", type=int, dest="_rle_bitness")
    p.add_argument("--huff-sym-size", type=int, dest="_huff_bitness")


def test(args, db):
    from test import test_wm

    if args.algo is None:
        algos = wm.WMBase.get_subclasses()
    else:
        algos = [wm.WMBase.find_subclass(algo) for algo in args.algo.split(",")]

    for algo in algos:
        test_wm.test_algo(algo, verbose=args._verbose, only_skipped=args.only_skipped)


def file_info(args, db):
    with db.new_ctx(aggregs=["mean", "worst"]) as db:
        for rec_in in args.rec_files:
            with db.new_ctx() as dbc:
                rec = load_record_file(rec_in, dbc, args.channel)
                rec.print_file_info()

                for c in range(rec.signal_count):
                    dbc.set(**rec.signal_info(c))

                if args.reconstruct:
                    rec.reconstruct_channels(dbc)


def rand_bytes(args, db):
    args.out_file.write_bytes(util.Random().bytes(args.num_bytes))


def add_noise(args, db):
    rec = load_record_file(args.rec_in, db, args.channel)
    rec.add_noise(args.var)
    rec.save(args.rec_out)


def do_wm(args, db):
    wm_params = {
        k[1:]: v for k, v in vars(args).items() if k.startswith("_") and v is not None
    }
    db.set(**wm_params, algo=args.algo)
    worker = AlgoBase.find_subclass(args.algo).new(**wm_params)

    if args.action == "research":
        if args.wm_file is not None:
            watermark = args.wm_file.read_bytes()
            watermark = util.to_bits(watermark)
        elif args.wm_len is not None:
            watermark = util.Random().bits(args.wm_len)
        else:
            parser.error("You must specify one of --wm-len and --wm-file")

        worker.set_watermark(watermark)
        worker.wm_len = len(watermark)
        db.set(noise_var=args.noise_var)

        for rec_in in args.rec_files:
            try:
                with db.new_ctx() as dbc:
                    rec = load_record_file(rec_in, dbc, args.channel)
                    orig_cont = rec.signals.copy()
                    orig_wm = [watermark] * rec.signal_count

                    rec.embed_watermark(worker)

                    if args.noise_var:
                        rec.add_noise(args.noise_var)

                    rec.extract_watermark(worker, orig_wm=orig_wm, orig_carr=orig_cont)
            except errors.DynamicError as e:
                print(f"Skipped ({repr(e)})")

            print()
            print()
    else:
        if args.action in ("embed", "extract"):
            if args.rec_out is None:
                args.rec_out = args.rec_in.parent / (
                    args.rec_in.stem + "_" + args.action + args.rec_in.suffix
                )

        if args.action in ("embed", "check"):
            if not args.wm_in:
                watermark = rec.wm_str
                print("Derived watermark from metadata:")
                print(f"  {watermark!r}")
            else:
                watermark = args.wm_in.read_bytes()
            watermark = util.to_bits(watermark)

        rec = load_record_file(args.rec_in, db, args.channel)

        if args.action == "embed":
            worker.set_watermark(watermark)
            with db.new_ctx(aggregs=["mean", "worst"]) as dbc:
                rec.embed_watermark(worker, dbc)
            rec.save(args.rec_out)
        elif args.action == "extract":
            worker.wm_len = args.wm_len
            with db.new_ctx(aggregs=["mean", "worst"]) as dbc:
                wms = rec.extract_watermark(worker, dbc)
            for wm in wms:
                if not np.array_equal(wm, wms[0]):
                    print(
                        "Warning: different watermarks in different channels. Only the first one will be written."
                    )
                    break
            args.wm_out.write_bytes(wms[0])
            rec.save(args.rec_out)
        elif args.action == "check":
            orig_wm = [watermark] * rec.signal_count
            worker.wm_len = len(watermark)
            with db.new_ctx(aggregs=["mean", "worst"]) as dbc:
                rec.extract_watermark(worker, dbc, orig_wm=orig_wm)


parser = make_parser()
args = parser.parse_args()

util.Random.default_seed = args.seed
db = Database(args.data_file, dump_all=args.data_all)
db.load()
dbc = db.new_ctx(aggregs=["mean", "worst"], aggreg_psnr=True)

with dbc:
    args.func(args, dbc)

print(f"All done! Elapsed time: {dbc.get('elapsed'):.2f} s")
db.dump()
