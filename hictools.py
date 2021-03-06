#!/usr/bin/env python

from __future__ import print_function
import sys
import os
import errno
import re
import gzip
import numpy as np
import numpy.ma as ma
from binnedData import BinnedData


def sam2mat_main(args):
    region_pattern = r'^[^:]+(?::\d+-\d+)?(?:,[^:]+(?::\d+-\d+)?)?$'
    if args.region is not None and re.search(region_pattern, args.region):
        regions = args.region
    elif args.reglist is not None:
        with open(args.reglist) as f:
            regions = [line.rstrip() for line in f]
    else:
        regions = None

    if args.insam is None:
        sam_fh = sys.stdin
    else:
        sam_fh = open(args.insam, 'r')

    bdata = BinnedData(args.fai, regions=regions, resolution=args.resolution)
    bdata.read_sam(sam_fh)
    sam_fh.close()

    if args.clean:
        bdata.clean()
    if args.ice:
        bdata.iterative_correction()

    margins = bdata.dat.sum(axis=0)
    #print(margins)
    #sys.exit()

    try:
        os.makedirs(args.outdir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise(e)
    bin_outfile = os.path.join(args.outdir, 'bins.txt.gz')
    contact_outfile = os.path.join(args.outdir, 'contacts.txt.gz')
    matrix_outfile = os.path.join(args.outdir, 'matrix.txt.gz')
    bin_f = gzip.open(bin_outfile, 'wb')
    contact_f = gzip.open(contact_outfile, 'wb')
    matrix_f = gzip.open(matrix_outfile, 'wb')

    for i,chrom1,b1 in bdata.iter_bins():
        bin_mid1 = (b1[0]+b1[1])/2
        if ma.is_masked(margins[i]):
            margin = 0
        else:
            margin = int(margins[i])
        print('{}\t{}\t{}\t{}\t{}'.format(chrom1,0,bin_mid1,margin,int(margin>0)), file=bin_f)
        if bdata.cleaned:
            print('\t'.join(bdata.dat.data[i].astype(str)), file=matrix_f)
        else:
            print('\t'.join(bdata.dat[i].astype(str)), file=matrix_f)
        for j,chrom2,b2 in bdata.iter_bins():
            bin_mid2 = (b2[0]+b2[1])/2
            contact = bdata.dat[i,j]
            if j>i and not ma.is_masked(contact) and contact > 0:
                print('{}\t{}\t{}\t{}\t{}'.format(chrom1,bin_mid1,chrom2,bin_mid2,int(contact)), file=contact_f)

    bin_f.close()
    contact_f.close()
    matrix_f.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    subp = parser.add_subparsers(metavar='<command>', help='sub-commands')

    parser_sam2mat = subp.add_parser('sam2mat', help='extract contact matrix from SAM')
    parser_sam2mat.add_argument('-g', '--genome', metavar='FAI', dest='fai', type=str, required=True, help='genome.fa.fai')
    parser_sam2mat.add_argument('-r', '--region', metavar='chr[:start-end]', dest='region', type=str, default=None, help='region(s) as a comma-separated string')
    parser_sam2mat.add_argument('-R', '--reglist', metavar='FILE', dest='reglist', type=str, default=None, help='a list of regions (chr[:start-end])')
    parser_sam2mat.add_argument('-e', '--res', metavar='INT', dest='resolution', type=int, default=1e4, help='resolution in bp, default 10000')
    parser_sam2mat.add_argument('-i', '--input', metavar='SAM', dest='insam', type=str, default=None, help='input SAM, default from stdin')
    parser_sam2mat.add_argument('-o', '--outdir', metavar='DIR', dest='outdir', type=str, default='.', help='output dir, default to "."')
    parser_sam2mat.add_argument('-c', '--clean', metavar='', dest='clean', nargs='?', type=bool, default=False, help='clean contact matrix to remove outliers')
    parser_sam2mat.add_argument('-C', '--ice', metavar='', dest='ice', nargs='?', type=bool, default=False, help='perform interactive correction')
    parser_sam2mat.set_defaults(func=sam2mat_main)

    try:
        args = parser.parse_args()
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(1)

