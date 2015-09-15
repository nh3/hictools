'''
implement a matrix-like class representing interacting bins
'''

from __future__ import print_function
from warnings import warn
from collections import OrderedDict as ordict
import sys
import numpy as np
import numpy.ma as ma
from scipy.ndimage.filters import gaussian_filter


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


    def iter_bins(self):
        i = -1
        for reg in self.bins:
            chrom,colon,coords = reg.partition(':')
            for b in self.bins[reg]:
                i += 1
                yield i,chrom,b


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
                bin_starts = np.arange(region[1], region[2], self.resolution, dtype=int)
                bin_ends = np.append(bin_starts[1:], region[2])
            else:
                bin_starts = np.array([region[1]], dtype=int)
                bin_ends = np.array([region[2]], dtype=int)
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
        self.cleaned = False


    def fill_zero(self):
        self.dat = np.zeros((self.nbin,self.nbin), dtype=float)


    def read_sam(self, samfh):
        self.fill_zero()
        for line in samfh:
            if line[0] == '@':
                continue
            rname,flag,chrom,pos,mapq,cigar,chrom2,pos2,dist,other = line.rstrip().split('\t', 9)
            if chrom2 == '*':
                continue
            elif chrom2 == '=':
                chrom2 = chrom
            pos = int(pos)
            pos2 = int(pos2)
            if self.genomewide:
                reg = '{}:{}-{}'.format(chrom,0,self.chrom_size[chrom])
                reg2 = '{}:{}-{}'.format(chrom2,0,self.chrom_size[chrom2])
                idx1 = self.offsets[reg] + pos/self.resolution
                idx2 = self.offsets[reg2] + pos2/self.resolution
            else:
                idx1 = self.find_overlap_bin(chrom, pos)
                idx2 = self.find_overlap_bin(chrom2, pos2)
            if idx1 is not None and idx2 is not None:
                self.dat[idx1,idx2] += 1
                self.dat[idx2,idx1] += 1
        self.cleaned = False
        self.corrected = 0x0


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


    def write_circos_links(self, outfile=sys.stdout):
        if type(outfile) is file:
            f = outfile
        elif type(outfile) is str:
            f = open(outfile, 'w')
        for i,chrom1,b1 in self.iter_bins():
            for j,chrom2,b2 in self.iter_bins():
                if j>i:
                    count = self.dat[i,j]
                    print('{}\t{:d}\t{:d}\t{}\t{:d}\t{:d}\tvalue={}'.format(chrom1,b1[0],b1[1],chrom2,b2[0],b2[1],count), file=f)


    def write_bins(self, outfile):
        if type(outfile) is file:
            f = outfile
        elif type(outfile) is str:
            f = open(outfile, 'w')
        for i,chrom,b in self.iter_bins():
            print('{}\t{}\t{}'.format(chrom, *b))


    def clean(self, bottom=1e-2, top=5e-3):
        nbin = len(self.dat)
        bottom_n = int(nbin*bottom)
        top_n = int(nbin*(1-top))                                                                                                                                                                                                     
        rowSum = self.dat.sum(axis=1)
        k = np.sort(rowSum.argsort()[bottom_n:top_n])
        k_remove = np.array(tuple(set(range(nbin))-set(k)))
        mask = np.zeros((nbin,nbin))
        mask[k_remove,] = 1
        mask[...,k_remove] = 1
        self.raw_dat = self.dat.copy()
        self.dat = ma.array(self.dat, mask=mask)
        self.cleaned = True
        return k_remove


    def gaussian_smooth(self, stdev):
        self.smoothed_dat = gaussian_filter(self.dat, stdev)


    def iteractive_correction(self, max_iter=100, tolerance=1e-5):
        totalBias = ma.ones(self.nbin, float)
        mat = self.dat.copy()
        for r in xrange(max_iter):
            print('.', end='', file=sys.stderr)
            binSum = mat.sum(axis=1)
            mask = binSum==0
            bias = binSum/binSum[~mask].mean()
            bias[mask] = 1
            bias -= 1
            bias *= 0.8
            bias += 1
            totalBias *= bias
            biasMat = bias.reshape(1,len(bias)) * bias.reshape(len(bias),1)
            mat = mat / biasMat
            if ma.abs(bias-1).max() < tolerance:
                break
        self.dat = mat
        self.corr = totalBias[~mask].mean()
        self.corrected |= 0x1


    @staticmethod
    def calc_insulation(dat, dist_range, delta_size):
        nbin = len(dat)
        min_d,max_d = dist_range
        if min_d < 0 or max_d < min_d:
            raise ValueError('calc_insulation() requires 0 <= min_d <= max_d')
        insulation = ma.zeros(nbin)
        for i in xrange(nbin):
            if i < max_d or i >= nbin-max_d:
                insulation[i] = -1
            else:
                insulation[i] = dat[i,(i-max_d):(i-min_d)].sum() + dat[i,(i+min_d):(i+max_d)].sum()
        k = insulation > 0
        insulation[k] = ma.log2(insulation[k]/insulation[k].mean())
        insulation[~k] = 0
        delta = ma.zeros(nbin)
        for i in xrange(nbin):
            if i < delta_size:
                delta[i] = insulation[0] - insulation[i+delta_size]
            elif i >= nbin - delta_size:
                delta[i] = insulation[i-delta_size] - insulation[nbin-1] 
            else:
                delta[i] = insulation[i-delta_size]-insulation[i+delta_size]
        return insulation,delta


    @staticmethod
    def calc_directionality(dat, dist_range):
        nbin = len(dat)
        min_d,max_d = dist_range
        if min_d < 0 or max_d < min_d:
            raise ValueError('calc_insulation() requires 0 <= min_d <= max_d')
        directionality = ma.zeros(nbin)
        for i in xrange(nbin):
            if i < max_d or i >= nbin-max_d:
                directionality[i] = 0.0
            else:
                up = dat[i,(i-max_d):(i-min_d)].sum()
                down = dat[i,(i+min_d):(i+max_d)].sum()
                avg = (up+down)/2.0
                directionality[i] = (up-down)/ma.abs(up-down)*((up-avg)**2/avg + (down-avg)**2/avg)
        return directionality


    def correct_by_distance(self):
        nbin = len(self.dat)
        r = c = np.arange(nbin)
        for k in range(1,n):
            rr = r[:-k]
            cc = c[k:]
            kth_diag = self.dat.diagonal(k)
            avg = kth_diag.sum()/(nbin-k)
            if avg == 0:
                ratio = np.ones(len(rr))
            else:
                ratio = kth_diag/avg
            self.dat[rr,cc] = ratio
            self.dat[cc,rr] = ratio
        self.corrected |= 0x2
