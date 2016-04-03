import os
import sys
import argparse
import numpy
import h5py
import itertools



def construct_word2idx(filename):
	word2id = {}
	with open(filename, "r") as f:
			for line in f:
				k,v = line.split()
				word2id[k] = int(v)
	return word2id

def get_max_line(filename):
	max_linesize = 0
	with open(filename, "r") as f:
		for line in f:
			max_linesize = max(max_linesize, len(line.split()))
	return max_linesize

def get_data(filename, max_linesize, word2idx):
	inputs = []
	with open(filename, "r") as f:
		for line in f:
			current_input = []
			for item in line.split():
				current_input.append(word2idx[item])
			while len(current_input) < max_linesize:
				current_input.append(99)

			inputs.append(current_input)
	return numpy.array(inputs, dtype=int)


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('testset', help="Data set",
                        type=str)
    parser.add_argument('word2idx', help="Data set",
                        type=str)
    parser.add_argument('outputfile', help="HDF5 output file", 
                        type=str)



    args = parser.parse_args(arguments)
    testset = args.testset

    word2idx = construct_word2idx(args.word2idx)
    max_linesize = get_max_line(testset)
    inputs = get_data(testset, max_linesize, word2idx)

    # filename = args.dataset + '.hdf5'
    with h5py.File(args.outputfile, "w") as f:
        f['test_input'] = inputs



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
