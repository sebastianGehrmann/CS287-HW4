#!/usr/bin/env python

"""Sentence Segmentation Preprocessing
"""

import os
import sys
import argparse
import numpy
import h5py
import itertools


class Indexer:
    def __init__(self):
        self.counter = 1
        self.d = {}
        self.rev = {}
        self._lock = False
        
    def convert(self, w):
        if w not in self.d:
            assert(not self._lock)
            self.d[w] = self.counter
            self.rev[self.counter] = w
            self.counter += 1
        return self.d[w]

    def lock(self):
        self._lock = True

    def write(self, outfile):
        out = open(outfile, "w")
        items = [(v, k) for k, v in self.d.iteritems()]
        items.sort()
        for v, k in items:
            print >>out, k, v
        out.close()

def get_data(train, valid):
    target_indexer = Indexer()
    #add special words to indices in the target_indexer
    target_indexer.convert("<space>")
    target_indexer.convert("</s>")
    
    def convert(targetfile, batchsize, seqlength, outfile):
        words = []
        targets = []
        with open(targetfile, 'r') as f:
            for targ_orig in f:
                targ = targ_orig.strip("\n").split(" ") + ["</s>"]
                target_sent = [target_indexer.convert(w) for w in targ] 
                words += target_sent
        
        #only for char RNN
        #arg_output = numpy.array(words[1:] + [target_indexer.convert("</s>")])
        #only for space prediction
        targ_output = [1 if w!=target_indexer.convert("<space>") else 2 for w in words[1:]]
        targ_output = numpy.array(targ_output + [1])
        #print targ_output
        words = numpy.array(words)

        print (words.shape, "shape of the word array before preprocessing")

        # Write output.
        f = h5py.File(outfile, "w")
        size = int(words.shape[0] / (batchsize * seqlength))
        print (size, "number of blocks after conversion")

        original_index = numpy.array([i+1 for i, v in enumerate(words)])
        
        f["target"] = numpy.zeros((size, batchsize, seqlength), dtype=int)
        f["indices"] = numpy.zeros((size, batchsize, seqlength), dtype=int) 
        f["target_output"] = numpy.zeros((size, batchsize, seqlength), dtype=int) 
        pos = 0
        for row in range(batchsize):
            for batch in range(size):
                f["target"][batch, row] = words[pos:pos+seqlength]
                f["indices"][batch, row] = original_index[pos:pos+seqlength]
                f["target_output"][batch, row] = targ_output[pos:pos+seqlength]
                pos = pos + seqlength
        f["target_size"] = numpy.array([target_indexer.counter])
        f["words"] = words
        f["set_size"] = words.shape[0]

    convert(train, args.batchsize, args.seqlength, args.outputfile + ".hdf5")
    target_indexer.lock()
    convert(valid, args.batchsize, args.seqlength, args.outputfile + "val" + ".hdf5")
    target_indexer.write(args.outputfile + ".targ.dict")




FILE_PATHS = {"PTB": ("data/large_train.txt",
                      "data/valid_chars.txt",
                      "data/test_chars.txt",
                      "data/valid_chars_kaggle.txt")}
args = {}


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)

    parser.add_argument('batchsize', help="Batchsize", 
                        type=int)
    parser.add_argument('seqlength', help="Sequence length", 
                        type=int)
    parser.add_argument('outputfile', help="HDF5 output file", 
                        type=str)


    args = parser.parse_args(arguments)
    dataset = args.dataset
    train, valid, test, valid_kaggle = FILE_PATHS[dataset]
    get_data(train, valid)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
