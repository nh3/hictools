'''
implement a matrix-like class representing interacting bins
'''

from __future__ import print_function
from warnings import warn
from collections import OrderedDict as ordict
import sys
import numpy as np


class BinnedData(object):
    def __init__(self, genome_fai, regions=None, resolution=1e5):
        self.genome_fai = genome_fai
        self.chrom_size = self.read_fai(genome_fai)
        self.resolution = resolution
        self.genomewide = False
        if regions is None and resolution is not None:
            self.genomewide = True
            regions = ','.join(self.chrom_size.keys())
            self.parse_region(regions)
        elif type(regions) is str:
            self.parse_region(regions)
        else:
            try:
                self.regions = np.array(regions, dtype=[('chrom','a32'),('start','i4'),('end','i4')])
            except TypeError:
                raise TypeError('[regions] expect a list of tuples: [("chrI",1,200), ("chrI",251,300), ... ]')
        self.make_bins()


    @staticmethod
    def read_fai(filename):
        '''extract an OrderedDict of {chrom:size} from a genome.fa.fai'''
        chrom_size = ordict()
        with open(filename) as f:
            for line in f:
                chrom,size,offset,line_size,line_byte = line.rstrip().split()
                size = int(size)
                chrom_size[chrom] = size
        return chrom_size


    def parse_region(self, regions):
        parsed_regions = []
        for region in regions.rstrip(',').split(','):
            reg_chrom,colon,reg_coords = region.partition(':')
            if reg_chrom not in self.chrom_size:
                raise ValueError('{} not present in {}'.format(reg_chrom, self.genome_fai))
            if reg_coords == '':
                reg_start = 0
                reg_end = self.chrom_size[reg_chrom]
            else:
                s,dash,e = reg_coords.partition('-')
                if dash == '':
                    raise ValueError('incorrect region format in "{}"'.format(region))
                if e == '':
                    e = self.chrom_size[reg_chrom]
                if s == '':
                    s = 0
                reg_start = int(s)
                reg_end = int(e)
                if reg_end > self.chrom_size[reg_chrom]:
                    warn('specified region end {} exceeds chromosome size {}, replacing with the latter'.format(reg_end, self.chrom_size[reg_chrom]))
                    reg_end = self.chrom_size[reg_chrom]
            parsed_regions.append((reg_chrom,reg_start,reg_end))
        self.regions = np.array(parsed_regions, dtype=[('chrom','a32'),('start','i4'),('end','i4')])


    def make_bins(self):
        '''generate bins of a given resolution, with the constraint of region/chromosome sizes'''
        if self.regions is None:
            raise Exception('regions is None, this should not happen')
        bins = ordict()
        offsets = ordict()
        nbin = 0
        for region in self.regions:
            reg = '{}:{}-{}'.format(*region)
            if self.resolution is not None:
                bin_starts = np.arange(region[1], region[2], self.resolution)
                bin_ends = np.append(bin_starts[1:], region[2])
            else:
                bin_starts = np.array([region[1]])
                bin_ends = np.array([region[2]])
            offsets[reg] = nbin
            bins[reg] = np.vstack((bin_starts,bin_ends,bin_ends-bin_starts)).transpose()
            nbin += len(bins[reg])
        self.bins = bins
        self.offsets = offsets
        self.nbin = nbin


    def fill(self, filename):
        print('start reading {}'.format(filename), end=' ', file=sys.stderr)
        mat = np.loadtxt(filename, dtype=float, delimiter='\t')
        print('done', file=sys.stderr)
        if mat.shape == (self.nbin, self.nbin):
            self.dat = mat
        else:
            raise ValueError('unmatched dimensions')


    def fill_zero(self):
        self.dat = np.zeros((self.nbin,self.nbin), dtype=float)


    def read_sam(self, samfh):
        for line in samfh:
            if line[0] == '@':
                continue
            rname,flag,chrom,pos,mapq,cigar,chrom2,pos2,dist,other = line.rstrip().split('\t', maxsplit=9)
            if chrom2 == '*':
                continue
            if self.genomewide:
                idx1 = self.offsets[chrom] + int(pos)/resolution
                idx2 = self.offsets[chrom2] + int(pos2)/resolution
            else:
                idx1 = self.find_overlap_bin(chrom, pos)
                idx2 = self.find_overlap_bin(chrom2, pos2)
            if idx1 is not None and idx2 is not None
                self.dat[idx1,idx2] += 1
                self.dat[idx2,idx1] += 1


    def find_overlap_bin(self, chrom, pos):
        try:
            region = self.regions[(chrom==self.regions['chrom']) & (pos>self.regions['start']) & (pos<=self.regions['end'])][0]
            reg = '{}:{}-{}'.format(*region)
            k = np.where((pos > self.bins[reg][...,0]) & (pos <= self.bins[reg][...,1]))[0][0]
        except IndexError:
            idx = None
        else:
            idx = self.offsets[reg] + k
        finally:
            return idx


    def write_matrix(self, outfile):
        np.savetxt(outfile, self.dat, fmt='%.2e', delimiter='\t')


    def write_bins(self, outfile):
        if type(outfile) is file:
            f = outfile
        elif type(outfile) is str:
            f = open(outfile, 'w')
        for reg in self.bins:
            chrom,colon,coords = reg.partition(':')
            for b in self.bins[reg]:
                print('{}\t{}\t{}'.format(chrom, *reg)
