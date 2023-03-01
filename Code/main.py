from argparse import ArgumentParser
from pathlib import Path
import time

import numpy as np

from db import Database
from edf import EDF
from wm_lsb import LSBEmbedder
from wm_lcb import LCBEmbedder
from wm_de import DEEmbedder
from wm_pee import NeighorsPEE, SiblingChannelPEE
from wm_itb import ITBEmbedder
import util

DEFAULT_KEY = "ВечностьПахнетНефтью"
        

def make_parser():        
    parser = ArgumentParser(description="Medical timeseries watermarking tool.")
    parser.add_argument("-v", "--verbose", action="store_true", dest="_debug")
    parser.add_argument("-d", "--data-file", type=Path)
    subp = parser.add_subparsers(required=True)

    p = subp.add_parser("rand-bytes", help="Generate random bytes")
    p.add_argument("num_bytes", type=int)
    p.add_argument("out_file", type=Path)
    p.add_argument("--seed")
    p.set_defaults(func=rand_bytes)

    p = subp.add_parser("info", help="Print some info about an EDF(+) file")
    p.add_argument("edf_in", type=Path)
    p.set_defaults(func=file_info)

    p = subp.add_parser("add-noise")
    p.add_argument("var", type=float)
    p.add_argument("edf_in", type=Path)
    p.add_argument("edf_out", type=Path, nargs="?")
    p.set_defaults(func=add_noise)

    p = subp.add_parser("embed")
    p.add_argument("edf_in", type=Path)
    p.add_argument("wm_in", type=Path)
    p.add_argument("edf_out", type=Path, nargs="?")
    add_common_args(p)
    p.set_defaults(func=wm, action="embed")

    p = subp.add_parser("extract")
    p.add_argument("edf_in", type=Path)
    p.add_argument("wm_out", type=Path)
    p.add_argument("edf_out", type=Path, nargs="?")
    p.add_argument("-l", "--wm-len", type=int, required=True)
    add_common_args(p)
    p.set_defaults(func=wm, action="extract")

    p = subp.add_parser("check")
    p.add_argument("edf_in", type=Path)
    p.add_argument("wm_in", type=Path)
    add_common_args(p)
    p.set_defaults(func=wm, action="check")

    p = subp.add_parser("research")
    p.add_argument("edf_files", type=Path, nargs="+")
    p.add_argument("-w", "--wm-file", type=Path)
    p.add_argument("-l", "--wm-len", type=int)
    p.add_argument("--seed")
    p.add_argument("-n", "--noise-var", type=float)
    add_common_args(p)
    p.set_defaults(func=wm, action="research")
    
    return parser


def add_common_args(p):
    # Common WM params.
    p.add_argument("-d", "--data-file", type=Path)
    p.add_argument("-c", "--channel", type=int, default=-1)
    p.add_argument("-a", "--algo", choices=("lsb", "lcb", "de", "pee-n", "pee-c", "itb"), default="lsb")

    # Common embedder params.
    p.add_argument("-k", "--key", default=DEFAULT_KEY, dest="_key")
    p.add_argument("-s", "--shuffle", action="store_true", dest="_shuffle")
    p.add_argument("-C", "--non-contiguous", action="store_false", dest="_contiguous")
    p.add_argument("-r", "--redundancy", type=int, dest="_redundancy")
    p.add_argument("-b", "--block-len", type=int, dest="_block_len")

    # LSB params.
    p.add_argument("--lsb-lowest-bit", type=int, dest="_lsb_lowest_bit")

    # LCB params.
    p.add_argument("--lcb-coder", choices=["rle"], dest="_lcb_coder")
    p.add_argument("--lcb-transform", choices=["dct", "dwt"], dest="_lcb_transform")
    p.add_argument("--rle-bitness", type=int, dest="_rle_bitness")

    # DE params.
    p.add_argument("--de-shift", type=int, dest="_de_shift")
    # Here we need to explicitly specify default=None, otherwise store_true sets it to False.
    p.add_argument("--de-rand-shift", action="store_true", default=None, dest="_de_rand_shift")
    p.add_argument("--de-skip", action="store_true", default=None, dest="_de_skip")

    # PEE params.
    p.add_argument("--pee-neigbors", type=int, dest="_pee_neigbors")
    p.add_argument("--pee-ref-channel", type=int, dest="_pee_ref_channel")


def rand_bytes(args, db):
    args.out_file.write_bytes(util.random_bytes(args.num_bytes, args.seed))


def file_info(args, db):
    with db.new_ctx(aggregs=["mean", "worst"]) as db:
        for edf_in in args.edf_files:
            with db.new_ctx() as dbc:
                edf = read_edf(edf_in, dbc, args.channel)
    edf.print_file_info()

                for c in range(edf.signal_count):
                    dbc.set(**edf.signal_info(c))

def add_noise(args, db):
    edf = EDF()
    edf.load(str(args.edf_in))
    edf.add_noise(args.var)
    edf.save(str(args.edf_out))


def wm(args, db):
    if args.algo == "lsb":
        wm_class = LSBEmbedder
    elif args.algo == "lcb":
        wm_class = LCBEmbedder
    elif args.algo == "de":
        wm_class = DEEmbedder
    elif args.algo == "pee-n":
        wm_class = NeighorsPEE
    elif args.algo == "pee-c":
        wm_class = SiblingChannelPEE
    elif args.algo == "itb":
        wm_class = ITBEmbedder

    wm_params = {k[1:]: v for k, v in vars(args).items()
        if k.startswith("_") and v is not None}
    worker = wm_class(**wm_params)
    db.set(**wm_params, algo=args.algo)
    start_time = time.perf_counter()
        
    if args.action == "research":
        if args.wm_file is not None:
            watermark = args.wm_file.read_bytes()
        elif args.wm_len is not None:
            watermark = util.random_bytes(args.wm_len, args.seed)
        else:
            parser.error("You must specify one of --wm-len and --wm-file")

        watermark = util.to_bits(watermark)
        worker.set_watermark(watermark)
        worker.wm_len = len(watermark)
        db.set(noise_var=args.noise_var)
        
        for edf_in in args.edf_files:
            edf = read_edf(edf_in, db, args.channel)
            orig_carr = edf.signals.copy()
            orig_wm = [watermark] * edf.signal_count

            with db.new_ctx(aggregs=["mean", "worst"]) as dbc:
                edf.embed_watermark(worker, dbc)
            
            if args.noise_var:
                edf.add_noise(args.noise_var)
            
            with db.new_ctx(aggregs=["mean", "worst"]) as dbc:
                edf.extract_watermark(worker, dbc, orig_wm=orig_wm, orig_carr=orig_carr)
            print()
            print()
    else:
        if args.action in ("embed", "extract"):
            if args.edf_out is None:
                args.edf_out = args.edf_in.parent / (
                    args.edf_in.stem + "_" + args.action + args.edf_in.suffix)
            
        if args.action in ("embed", "check"):
            if not args.wm_in:
                watermark = edf.wm_str
                print("Derived watermark from metadata:")
                print(f"  {watermark!r}")
            else:
                watermark = args.wm_in.read_bytes()
            watermark = util.to_bits(watermark)
            
        edf = read_edf(args.edf_in, db, args.channel)
        
        if args.action == "embed":
            worker.set_watermark(watermark)
            with db.new_ctx(aggregs=["mean", "worst"]) as dbc:
                edf.embed_watermark(worker, dbc)
            edf.save(str(args.edf_out))
        elif args.action == "extract":
            worker.wm_len = args.wm_len
            with db.new_ctx(aggregs=["mean", "worst"]) as dbc:
                wms = edf.extract_watermark(worker, dbc)
            for wm in wms:
                if not np.array_equal(wm, wms[0]):
                    print("Warning: different watermarks in different channels. Only the first one will be written.")
                    break
            args.wm_out.write_bytes(wms[0])
            edf.save(str(args.edf_out))
        elif args.action == "check":
            orig_wm = [watermark] * edf.signal_count
            worker.wm_len = len(watermark)
            with db.new_ctx(aggregs=["mean", "worst"]) as dbc:
                edf.extract_watermark(worker, dbc, orig_wm=orig_wm)
            
    elapsed = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed:.2} s")
    db.set(elapsed_time=elapsed)
        
        
def read_edf(path, db, chan):
    edf = EDF()
    edf.load(str(path))
    edf.use_channel(chan)
    db.set_props(filename=path.name, filepath=str(path))
    db.set_props(**edf.file_info())
    return edf


parser = make_parser()
args = parser.parse_args()
args.func(args)
