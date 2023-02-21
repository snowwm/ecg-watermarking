from argparse import ArgumentParser
from pathlib import Path
import time

import numpy as np

from db import Database
from edf import EDF
from wm_lsb import LSBEmbedder
from wm_de import DEEmbedder
from wm_pee import PEEEmbedder
from wm_itb import ITBEmbedder
import util

DEFAULT_KEY = "ВечностьПахнетНефтью"
        

def make_parser():        
    parser = ArgumentParser(description="Medical timeseries watermarking tool.")
    parser.add_argument("-v", "--verbose", action="store_true")
    subp = parser.add_subparsers(required=True)

    p = subp.add_parser("rand-bytes", help="Generate random bytes")
    p.add_argument("num_bytes", type=int)
    p.add_argument("out_file", type=Path)
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
    p.add_argument("-l", "--wm-len", dest="wm_len", type=int, required=True)
    add_common_args(p)
    p.set_defaults(func=wm, action="extract")

    p = subp.add_parser("check")
    p.add_argument("edf_in", type=Path)
    p.add_argument("wm_in", type=Path)
    add_common_args(p)
    p.set_defaults(func=wm, action="check")

    p = subp.add_parser("research")
    p.add_argument("wm_in", type=Path)
    p.add_argument("edf_files", type=Path, nargs="+")
    p.add_argument("-n", "--noise-var", dest="noise_var", type=float)
    add_common_args(p)
    p.set_defaults(func=wm, action="research")
    
    return parser


def add_common_args(p):
    p.add_argument("-d", "--data-file", dest="data_file", type=Path)
    p.add_argument("-k", "--key", default=DEFAULT_KEY)
    p.add_argument("-c", "--channel", type=int, default=-1)
    p.add_argument("-a", "--algo", choices=("lsb", "de", "pee", "itb"), default="lsb")
    p.add_argument("-s", "--shuffle", action="store_true")
    p.add_argument("-C", "--contiguous", action="store_false")
    p.add_argument("-r", "--redundancy", type=int, default=1)
    p.add_argument("-b", "--block-len", dest="block_len", type=int, default=1)
    p.add_argument("--de-shift", dest="de_shift", type=int, default=1)
    p.add_argument("--de-rand-shift", dest="de_rand_shift", action="store_true")
    p.add_argument("--de-skip", dest="de_skip", action="store_true")


def rand_bytes(args):
    args.out_file.write_bytes(util.random_bytes(args.num_bytes))


def file_info(args):
    edf = EDF()
    edf.load(str(args.edf_in))
    edf.print_file_info()


def add_noise(args):
    edf = EDF()
    edf.load(str(args.edf_in))
    edf.add_noise(args.var)
    edf.save(str(args.edf_out))


def wm(args):
    wm_params = {
        "key": args.key,
        "shuffle": args.shuffle,
        "contiguous": args.contiguous,
        "redundancy": args.redundancy,
        "block_len": args.block_len,
        "debug": args.verbose,
    }
        
    if args.algo == "lsb":
        wm_class = LSBEmbedder
    elif args.algo == "de":
        wm_class = DEEmbedder
        wm_params["de_shift"] = args.de_shift
        wm_params["de_rand_shift"] = args.de_rand_shift
        wm_params["de_skip"] = args.de_skip
    elif args.algo == "pee":
        wm_class = PEEEmbedder
    elif args.algo == "itb":
        wm_class = ITBEmbedder
        
    worker = wm_class(**wm_params)
    db = Database()
    if args.data_file:
        db.load(args.data_file)
    db.set_props(**wm_params, algo=args.algo)
    start_time = time.perf_counter()
        
    if args.action == "research":
        watermark = args.wm_in.read_bytes()
        watermark = util.to_bits(watermark)
        worker.set_watermark(watermark)
        worker.wm_len = len(watermark)
        db.set_props(noise_var=args.noise_var)
        
        for edf_in in args.edf_files:
            edf = read_edf(edf_in, db, args.channel)
            orig_carr = edf.signals.copy()
            orig_wm = [watermark] * edf.signal_count
            edf.embed_watermark(worker, db.new_ctx())
            
            if args.noise_var:
                edf.add_noise(args.noise_var)
            
            edf.extract_watermark(worker, db.new_ctx(), orig_wm=orig_wm, orig_carr=orig_carr)
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
            edf.embed_watermark(worker, db.new_ctx())
            edf.save(str(args.edf_out))
        elif args.action == "extract":
            worker.wm_len = args.wm_len
            wms = edf.extract_watermark(worker, db.new_ctx())
            for wm in wms:
                if not np.array_equal(wm, wms[0]):
                    print("Warning: different watermarks in different channels. Only the first one will be written.")
                    break
            args.wm_out.write_bytes(wms[0])
            edf.save(str(args.edf_out))
        elif args.action == "check":
            orig_wm = [watermark] * edf.signal_count
            worker.wm_len = len(watermark)
            edf.extract_watermark(worker, db.new_ctx(), orig_wm=orig_wm)
            
    elapsed = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed:.2} s")
    db.set_props(elapsed_time=elapsed)
    if args.data_file:
        db.save(args.data_file)
        
        
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
