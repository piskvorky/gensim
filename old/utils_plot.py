#!/usr/bin/env python

import numpy
import common
import docsim
import matutils
import mscs
from PIL import Image
import math
import sys

import logging

import ipyutils
ipyutils.loadDicts(prefix = 'gensim_' + 'eng')

logging.root.level = 10

LUM_SCALE = 30
LUM_LOG = numpy.log(LUM_SCALE + 1)

SIM_TYPE = "LSI"
#SIM_TYPE = "TFIDF"
lineWidth = 5


def getLum(sim):
	return 255 * numpy.log(1 + LUM_SCALE * sim) / LUM_LOG

def removeDup(lst):
	"""remove duplicates, but keep original item order"""
	seen = set()
	result = []
	for item in lst:
		if item not in seen:
			seen.add(item)
			result.append(item)
	return result	

def processMsc(mat, id):
	logging.info("avg similarity within %s: %f" % (id, numpy.mean(mat)))
	logging.info("stddev similarity within %s: %f" % (id, numpy.std(mat)))
	logging.debug("creating msc=%s luminescence matrix" % (id))
	print mat
	matLum = getLum(mat).astype(numpy.uint8)
	fname = "%s_sim_lum%i_%s.png" % (SIM_TYPE, LUM_SCALE, id)
	logging.debug("saving msc=%s similarity luminescence matrix to %s" % (id, fname))
	logging.info("saving to %s" % fname)
	i = Image.fromarray(matLum, 'L')
	i.save(fname)


logging.info("loading articles")
#arts = docsim.getArts(common.dbFile('serial_eng', '1msc'))
arts = docsim.getArts('/home/radim/workspace/data/dml/results/serial_msc.pdl')

for art in arts:
	art.fullmsc = art.msc[:]
	art.msc = tuple(removeDup([mscs.niceMSC(msc)[0] for msc in art.msc]))
arts = [art for art in arts if art.id_int in ipyutils.rdocids and art.language == "eng" and len(art.msc) == 1]

art2msc = [(art.fullmsc, art.id_int, art.msc) for art in arts]
art2msc.sort()

del arts

logging.info("len(art2msc)=%i" % len(art2msc))
print "first ten art2msc:", art2msc[:10]
print "last ten art2msc:", art2msc[-10:]

art2mscOld = art2msc[:]
art2msc = [(msc, id_int) for fullmsc, id_int, msc in art2msc]

oldMsc = None
breaks = []
for i, (msc, id) in enumerate(art2msc):
	if msc != oldMsc:
		breaks.append((i, msc))
		oldMsc = msc
logging.info("%i breaks" % len(breaks))
print "breaks:", breaks

# print individual category matrices
breaks.append((i, msc)) # append the last category
print "breaks with end:", breaks

new2old = {}
for i, (msc, id) in enumerate(art2msc):
    new2old[i] = ipyutils.rdocids[id]
print '==first 100 new2old:', new2old.items()[:100]
	

logging.info("loading cossim matrix")
mat = matutils.loadMatrix(common.matrixFile("gensim_eng%ssim.mm" % (SIM_TYPE)))
for i in xrange(len(breaks) - 1):
	numArts = breaks[i + 1][0] - breaks[i][0]
	logging.debug("%i articles in category %s" % (numArts, breaks[i][1]))
	matId = numpy.zeros((numArts, numArts), numpy.float)
	for i1 in xrange(numArts):
		print breaks[i][0] + i1, i1, art2mscOld[breaks[i][0] + i1]
		for i2 in xrange(numArts):
			pos1, pos2 = breaks[i][0] + i1, breaks[i][0] + i2
			matId[i1, i2] = mat[new2old[pos1], new2old[pos2]]
	processMsc(matId, breaks[i][1][0])

sys.exit(0)

new2old = {}
for i, (msc, id) in enumerate(art2msc):
    new2old[i] = ipyutils.rdocids[id]


oldpos2newpos = {}
for i in xrange(1, len(breaks)):
	oldpos2newpos.update(zip(range(breaks[i - 1][0], breaks[i][0]), range(lineWidth * i - 1 + breaks[i - 1][0], lineWidth * i - 1 + breaks[i][0])))
oldpos2newpos.update(zip(range(breaks[-1][0], len(art2msc)), range(lineWidth * len(breaks) - 1 + breaks[-1][0], lineWidth * len(breaks) - 1 + len(art2msc))))
assert(len(oldpos2newpos) == len(art2msc))

logging.info("loading cossim matrix")
mat = matutils.loadMatrix(common.matrixFile("gensim_eng%ssim.mm" % (SIM_TYPE)))
#mat = matutils.loadMatrix(common.matrixFile("gensim_engLSIsim.mm"))

matLum = numpy.zeros((len(art2msc) + lineWidth * len(breaks), len(art2msc) + lineWidth * len(breaks)), numpy.uint8) + 255
logging.info("creating %ix%i luminescence matrix" % matLum.shape)
for i in xrange(len(art2msc)):
	for j in xrange(len(art2msc)):
		matLum[oldpos2newpos[i], oldpos2newpos[j]] = getLum(mat[new2old[i], new2old[j]])

fname = "%s_sim_lum%i_fullsort_msc1_line%i.png" % (SIM_TYPE, LUM_SCALE, lineWidth)
logging.info("saving to %s" % fname)
i = Image.fromarray(matLum, 'L')
i.save(fname)
