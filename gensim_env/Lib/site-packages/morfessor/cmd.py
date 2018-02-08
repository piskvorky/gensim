import locale
import logging
import math
import random
import os.path
import sys
import time
import string

from . import get_version
from .baseline import BaselineModel, AnnotationCorpusWeight, \
    MorphLengthCorpusWeight, NumMorphCorpusWeight, FixedCorpusWeight
from .exception import ArgumentException
from .io import MorfessorIO
from .evaluation import MorfessorEvaluation, EvaluationConfig, \
    WilcoxonSignedRank, FORMAT_STRINGS

PY3 = sys.version_info[0] == 3

# _str is used to convert command line arguments to the right type (str for PY3, unicode for PY2
if PY3:
    _str = str
else:
    _str = lambda x: unicode(x, encoding=locale.getpreferredencoding())

_logger = logging.getLogger(__name__)



def get_default_argparser():
    import argparse

    parser = argparse.ArgumentParser(
        prog='morfessor.py',
        description="""
Morfessor %s

Copyright (c) 2012-2014, Sami Virpioja and Peter Smit
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1.  Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

2.  Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following
    disclaimer in the documentation and/or other materials provided
    with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Command-line arguments:
""" % get_version(),
        epilog="""
Simple usage examples (training and testing):

  %(prog)s -t training_corpus.txt -s model.pickled
  %(prog)s -l model.pickled -T test_corpus.txt -o test_corpus.segmented

Interactive use (read corpus from user):

  %(prog)s -m online -v 2 -t -

""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False)

    # Options for input data files
    add_arg = parser.add_argument_group('input data files').add_argument
    add_arg('-l', '--load', dest="loadfile", default=None, metavar='<file>',
            help="load existing model from file (pickled model object)")
    add_arg('-L', '--load-segmentation', dest="loadsegfile", default=None,
            metavar='<file>',
            help="load existing model from segmentation "
                 "file (Morfessor 1.0 format)")
    add_arg('-t', '--traindata', dest='trainfiles', action='append',
            default=[], metavar='<file>',
            help="input corpus file(s) for training (text or bz2/gzipped text;"
                 " use '-' for standard input; add several times in order to "
                 "append multiple files)")
    add_arg('-T', '--testdata', dest='testfiles', action='append',
            default=[], metavar='<file>',
            help="input corpus file(s) to analyze (text or bz2/gzipped text;  "
                 "use '-' for standard input; add several times in order to "
                 "append multiple files)")

    # Options for output data files
    add_arg = parser.add_argument_group('output data files').add_argument
    add_arg('-o', '--output', dest="outfile", default='-', metavar='<file>',
            help="output file for test data results (for standard output, "
                 "use '-'; default '%(default)s')")
    add_arg('-s', '--save', dest="savefile", default=None, metavar='<file>',
            help="save final model to file (pickled model object)")
    add_arg('-S', '--save-segmentation', dest="savesegfile", default=None,
            metavar='<file>',
            help="save model segmentations to file (Morfessor 1.0 format)")
    add_arg('--save-reduced', dest="savereduced", default=None,
            metavar='<file>',
            help="save final model to file in reduced form (pickled model "
            "object). A model in reduced form can only be used for "
            "segmentation of new words.")
    add_arg('-x', '--lexicon', dest="lexfile", default=None, metavar='<file>',
            help="output final lexicon to given file")
    add_arg('--nbest', dest="nbest", default=1, type=int, metavar='<int>',
            help="output n-best viterbi results")

    # Options for data formats
    add_arg = parser.add_argument_group(
        'data format options').add_argument
    add_arg('-e', '--encoding', dest='encoding', metavar='<encoding>',
            help="encoding of input and output files (if none is given, "
                 "both the local encoding and UTF-8 are tried)")
    add_arg('--lowercase', dest="lowercase", default=False,
            action='store_true',
            help="lowercase input data")
    add_arg('--traindata-list', dest="list", default=False,
            action='store_true',
            help="input file(s) for batch training are lists "
                 "(one compound per line, optionally count as a prefix)")
    add_arg('--atom-separator', dest="separator", type=_str, default=None,
            metavar='<regexp>',
            help="atom separator regexp (default %(default)s)")
    add_arg('--compound-separator', dest="cseparator", type=_str, default='\s+',
            metavar='<regexp>',
            help="compound separator regexp (default '%(default)s')")
    add_arg('--analysis-separator', dest='analysisseparator', type=_str,
            default=',', metavar='<str>',
            help="separator for different analyses in an annotation file. Use"
                 "  NONE for only allowing one analysis per line")
    add_arg('--output-format', dest='outputformat', type=_str,
            default=r'{analysis}\n', metavar='<format>',
            help="format string for --output file (default: '%(default)s'). "
            "Valid keywords are: "
            "{analysis} = constructions of the compound, "
            "{compound} = compound string, "
            "{count} = count of the compound (currently always 1), "
            "{logprob} = log-probability of the analysis, and "
            "{clogprob} = log-probability of the compound. Valid escape "
            "sequences are '\\n' (newline) and '\\t' (tabular)")
    add_arg('--output-format-separator', dest='outputformatseparator',
            type=_str, default=' ', metavar='<str>',
            help="construction separator for analysis in --output file "
            "(default: '%(default)s')")
    add_arg('--output-newlines', dest='outputnewlines', default=False,
            action='store_true',
            help="for each newline in input, print newline in --output file "
            "(default: '%(default)s')")

    # Options for model training
    add_arg = parser.add_argument_group(
        'training and segmentation options').add_argument
    add_arg('-m', '--mode', dest="trainmode", default='init+batch',
            metavar='<mode>',
            choices=['none', 'batch', 'init', 'init+batch', 'online',
                     'online+batch'],
            help="training mode ('none', 'init', 'batch', 'init+batch', "
                 "'online', or 'online+batch'; default '%(default)s')")
    add_arg('-a', '--algorithm', dest="algorithm", default='recursive',
            metavar='<algorithm>', choices=['recursive', 'viterbi'],
            help="algorithm type ('recursive', 'viterbi'; default "
                 "'%(default)s')")
    add_arg('-d', '--dampening', dest="dampening", type=_str, default='ones',
            metavar='<type>', choices=['none', 'log', 'ones'],
            help="frequency dampening for training data ('none', 'log', or "
                 "'ones'; default '%(default)s')")
    add_arg('-f', '--forcesplit', dest="forcesplit", type=list, default=['-'],
            metavar='<list>',
            help="force split on given atoms (default '-'). The list argument "
                 "is a string of characthers, use '' for no forced splits.")
    add_arg('-F', '--finish-threshold', dest='finish_threshold', type=float,
            default=0.005, metavar='<float>',
            help="Stopping threshold. Training stops when "
                 "the improvement of the last iteration is"
                 "smaller then finish_threshold * #boundaries; "
                 "(default '%(default)s')")
    add_arg('-r', '--randseed', dest="randseed", default=None,
            metavar='<seed>',
            help="seed for random number generator")
    add_arg('-R', '--randsplit', dest="splitprob", default=None, type=float,
            metavar='<float>',
            help="initialize new words by random splitting using the given "
                 "split probability (default no splitting)")
    add_arg('--skips', dest="skips", default=False, action='store_true',
            help="use random skips for frequently seen compounds to speed up "
                 "training")
    add_arg('--batch-minfreq', dest="freqthreshold", type=int, default=1,
            metavar='<int>',
            help="compound frequency threshold for batch training (default "
                 "%(default)s)")
    add_arg('--max-epochs', dest='maxepochs', type=int, default=None,
            metavar='<int>',
            help='hard maximum of epochs in training')
    add_arg('--nosplit-re', dest="nosplit", type=_str, default=None,
            metavar='<regexp>',
            help="if the expression matches the two surrounding characters, "
                 "do not allow splitting (default %(default)s)")
    add_arg('--online-epochint', dest="epochinterval", type=int,
            default=10000, metavar='<int>',
            help="epoch interval for online training (default %(default)s)")
    add_arg('--viterbi-smoothing', dest="viterbismooth", default=0,
            type=float, metavar='<float>',
            help="additive smoothing parameter for Viterbi training "
                 "and segmentation (default %(default)s)")
    add_arg('--viterbi-maxlen', dest="viterbimaxlen", default=30,
            type=int, metavar='<int>',
            help="maximum construction length in Viterbi training "
                 "and segmentation (default %(default)s)")

    # Options for corpusweight tuning
    add_arg = parser.add_mutually_exclusive_group().add_argument
    add_arg('-D', '--develset', dest="develfile", default=None,
            metavar='<file>',
            help="load annotated data for tuning the corpus weight parameter")
    add_arg('--morph-length', dest='morphlength', default=None, type=float,
            metavar='<float>',
            help="tune the corpusweight to obtain the desired average morph "
                 "length")
    add_arg('--num-morph-types', dest='morphtypes', default=None, type=float,
            metavar='<float>',
            help="tune the corpusweight to obtain the desired number of morph "
                 "types")

    # Options for semi-supervised model training
    add_arg = parser.add_argument_group(
        'semi-supervised training options').add_argument
    add_arg('-w', '--corpusweight', dest="corpusweight", type=float,
            default=1.0, metavar='<float>',
            help="corpus weight parameter (default %(default)s); "
                 "sets the initial value if other tuning options are used")
    add_arg('--weight-threshold', dest='threshold', default=0.01,
            metavar='<float>', type=float,
            help='percentual stopping threshold for corpusweight updaters')
    add_arg('--full-retrain', dest='fullretrain', action='store_true',
            default=False,
            help='do a full retrain after any weights have converged')
    add_arg('-A', '--annotations', dest="annofile", default=None,
            metavar='<file>',
            help="load annotated data for semi-supervised learning")
    add_arg('-W', '--annotationweight', dest="annotationweight",
            type=float, default=None, metavar='<float>',
            help="corpus weight parameter for annotated data (if unset, the "
                 "weight is set to balance the number of tokens in annotated "
                 "and unannotated data sets)")

    # Options for evaluation
    add_arg = parser.add_argument_group('Evaluation options').add_argument
    add_arg('-G', '--goldstandard', dest='goldstandard', default=None,
            metavar='<file>',
            help='If provided, evaluate the model against the gold standard')

    # Options for logging
    add_arg = parser.add_argument_group('logging options').add_argument
    add_arg('-v', '--verbose', dest="verbose", type=int, default=1,
            metavar='<int>',
            help="verbose level; controls what is written to the standard "
                 "error stream or log file (default %(default)s)")
    add_arg('--logfile', dest='log_file', metavar='<file>',
            help="write log messages to file in addition to standard "
                 "error stream")
    add_arg('--progressbar', dest='progress', default=False,
            action='store_true',
            help="Force the progressbar to be displayed (possibly lowers the "
                 "log level for the standard error stream)")

    add_arg = parser.add_argument_group('other options').add_argument
    add_arg('-h', '--help', action='help',
            help="show this help message and exit")
    add_arg('--version', action='version',
            version='%(prog)s ' + get_version(),
            help="show version number and exit")

    return parser


def main(args):
    if args.verbose >= 2:
        loglevel = logging.DEBUG
    elif args.verbose >= 1:
        loglevel = logging.INFO
    else:
        loglevel = logging.WARNING

    logging_format = '%(asctime)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    default_formatter = logging.Formatter(logging_format, date_format)
    plain_formatter = logging.Formatter('%(message)s')
    logging.basicConfig(level=loglevel)
    _logger.propagate = False  # do not forward messages to the root logger

    # Basic settings for logging to the error stream
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    ch.setFormatter(plain_formatter)
    _logger.addHandler(ch)

    # Settings for when log_file is present
    if args.log_file is not None:
        fh = logging.FileHandler(args.log_file, 'w')
        fh.setLevel(loglevel)
        fh.setFormatter(default_formatter)
        _logger.addHandler(fh)
        # If logging to a file, make INFO the highest level for the
        # error stream
        ch.setLevel(max(loglevel, logging.INFO))

    # If debug messages are printed to screen or if stderr is not a tty (but
    # a pipe or a file), don't show the progressbar
    global show_progress_bar
    if (ch.level > logging.INFO or
            (hasattr(sys.stderr, 'isatty') and not sys.stderr.isatty())):
        show_progress_bar = False

    if args.progress:
        show_progress_bar = True
        ch.setLevel(min(ch.level, logging.INFO))

    if (args.loadfile is None and
            args.loadsegfile is None and
            len(args.trainfiles) == 0):
        raise ArgumentException("either model file or training data should "
                                "be defined")

    if args.randseed is not None:
        random.seed(args.randseed)

    io = MorfessorIO(encoding=args.encoding,
                     compound_separator=args.cseparator,
                     atom_separator=args.separator,
                     lowercase=args.lowercase)

    # Load exisiting model or create a new one
    if args.loadfile is not None:
        model = io.read_binary_model_file(args.loadfile)

    else:
        model = BaselineModel(forcesplit_list=args.forcesplit,
                              corpusweight=args.corpusweight,
                              use_skips=args.skips,
                              nosplit_re=args.nosplit)

    if args.loadsegfile is not None:
        model.load_segmentations(io.read_segmentation_file(args.loadsegfile))

    analysis_sep = (args.analysisseparator
                    if args.analysisseparator != 'NONE' else None)

    if args.annofile is not None:
        annotations = io.read_annotations_file(args.annofile,
                                               analysis_sep=analysis_sep)
        model.set_annotations(annotations, args.annotationweight)

    if args.develfile is not None:
        develannots = io.read_annotations_file(args.develfile,
                                               analysis_sep=analysis_sep)
        updater = AnnotationCorpusWeight(develannots, args.threshold)
        model.set_corpus_weight_updater(updater)

    if args.morphlength is not None:
        updater = MorphLengthCorpusWeight(args.morphlength, args.threshold)
        model.set_corpus_weight_updater(updater)

    if args.morphtypes is not None:
        updater = NumMorphCorpusWeight(args.morphtypes, args.threshold)
        model.set_corpus_weight_updater(updater)

    start_corpus_weight = model.get_corpus_coding_weight()

    # Set frequency dampening function
    if args.dampening == 'none':
        dampfunc = None
    elif args.dampening == 'log':
        dampfunc = lambda x: int(round(math.log(x + 1, 2)))
    elif args.dampening == 'ones':
        dampfunc = lambda x: 1
    else:
        raise ArgumentException("unknown dampening type '%s'" % args.dampening)

    # Set algorithm parameters
    if args.algorithm == 'viterbi':
        algparams = (args.viterbismooth, args.viterbimaxlen)
    else:
        algparams = ()

    # Train model
    if args.trainmode == 'none':
        pass
    elif args.trainmode == 'batch':
        if len(model.get_compounds()) == 0:
            _logger.warning("Model contains no compounds for batch training."
                            " Use 'init+batch' mode to add new data.")
        else:
            if len(args.trainfiles) > 0:
                _logger.warning("Training mode 'batch' ignores new data "
                                "files. Use 'init+batch' or 'online' to "
                                "add new compounds.")
            ts = time.time()
            e, c = model.train_batch(args.algorithm, algparams,
                                     args.finish_threshold, args.maxepochs)
            te = time.time()
            _logger.info("Epochs: %s" % e)
            _logger.info("Final cost: %s" % c)
            _logger.info("Training time: %.3fs" % (te - ts))
    elif len(args.trainfiles) > 0:
        ts = time.time()
        if args.trainmode == 'init':
            if args.list:
                data = io.read_corpus_list_files(args.trainfiles)
            else:
                data = io.read_corpus_files(args.trainfiles)
            c = model.load_data(data, args.freqthreshold, dampfunc,
                                args.splitprob)
        elif args.trainmode == 'init+batch':
            if args.list:
                data = io.read_corpus_list_files(args.trainfiles)
            else:
                data = io.read_corpus_files(args.trainfiles)
            c = model.load_data(data, args.freqthreshold, dampfunc,
                                args.splitprob)
            e, c = model.train_batch(args.algorithm, algparams,
                                     args.finish_threshold, args.maxepochs)
            _logger.info("Epochs: %s" % e)
            if args.fullretrain:
                if abs(model.get_corpus_coding_weight() - start_corpus_weight) > 0.1:
                    model.set_corpus_weight_updater(
                        FixedCorpusWeight(model.get_corpus_coding_weight()))
                    model.clear_segmentation()
                    e, c = model.train_batch(args.algorithm, algparams,
                                             args.finish_threshold,
                                             args.maxepochs)
                    _logger.info("Retrain Epochs: %s" % e)
        elif args.trainmode == 'online':
            data = io.read_corpus_files(args.trainfiles)
            e, c = model.train_online(data, dampfunc, args.epochinterval,
                                      args.algorithm, algparams,
                                      args.splitprob, args.maxepochs)
            _logger.info("Epochs: %s" % e)
        elif args.trainmode == 'online+batch':
            data = io.read_corpus_files(args.trainfiles)
            e, c = model.train_online(data, dampfunc, args.epochinterval,
                                      args.algorithm, algparams,
                                      args.splitprob, args.maxepochs)
            e, c = model.train_batch(args.algorithm, algparams,
                                     args.finish_threshold, args.maxepochs - e)
            _logger.info("Epochs: %s" % e)
            if args.fullretrain:
                if abs(model.get_corpus_coding_weight() -
                        start_corpus_weight) > 0.1:
                    model.clear_segmentation()
                    e, c = model.train_batch(args.algorithm, algparams,
                                             args.finish_threshold,
                                             args.maxepochs)
                    _logger.info("Retrain Epochs: %s" % e)
        else:
            raise ArgumentException("unknown training mode '%s'"
                                    % args.trainmode)
        te = time.time()
        _logger.info("Final cost: %s" % c)
        _logger.info("Training time: %.3fs" % (te - ts))
    else:
        _logger.warning("No training data files specified.")

    # Save model
    if args.savefile is not None:
        io.write_binary_model_file(args.savefile, model)

    if args.savesegfile is not None:
        io.write_segmentation_file(args.savesegfile, model.get_segmentations())

    # Output lexicon
    if args.lexfile is not None:
        io.write_lexicon_file(args.lexfile, model.get_constructions())

    if args.savereduced is not None:
        model.make_segment_only()
        io.write_binary_model_file(args.savereduced, model)

    # Segment test data
    if len(args.testfiles) > 0:
        _logger.info("Segmenting test data...")
        outformat = args.outputformat
        csep = args.outputformatseparator
        outformat = outformat.replace(r"\n", "\n")
        outformat = outformat.replace(r"\t", "\t")
        keywords = [x[1] for x in string.Formatter().parse(outformat)]
        with io._open_text_file_write(args.outfile) as fobj:
            testdata = io.read_corpus_files(args.testfiles)
            i = 0
            for count, atoms in testdata:
                if io.atom_separator is None:
                    compound = "".join(atoms)
                else:
                    compound = io.atom_separator.join(atoms)
                if len(atoms) == 0:
                    # Newline in corpus
                    if args.outputnewlines:
                        fobj.write("\n")
                    continue
                if "clogprob" in keywords:
                    clogprob = model.forward_logprob(atoms)
                else:
                    clogprob = 0
                if args.nbest > 1:
                    nbestlist = model.viterbi_nbest(atoms, args.nbest,
                                                    args.viterbismooth,
                                                    args.viterbimaxlen)
                    for constructions, logp in nbestlist:
                        analysis = io.format_constructions(constructions,
                                                           csep=csep)
                        fobj.write(outformat.format(analysis=analysis,
                                                    compound=compound,
                                                    count=count, logprob=logp,
                                                    clogprob=clogprob))
                else:
                    constructions, logp = model.viterbi_segment(
                        atoms, args.viterbismooth, args.viterbimaxlen)
                    analysis = io.format_constructions(constructions, csep=csep)
                    fobj.write(outformat.format(analysis=analysis,
                                                compound=compound,
                                                count=count, logprob=logp,
                                                clogprob=clogprob))
                i += 1
                if i % 10000 == 0:
                    sys.stderr.write(".")
            sys.stderr.write("\n")
        _logger.info("Done.")

    if args.goldstandard is not None:
        _logger.info("Evaluating Model")
        e = MorfessorEvaluation(io.read_annotations_file(args.goldstandard))
        result = e.evaluate_model(model, meta_data={'name': 'MODEL'})
        print(result.format(FORMAT_STRINGS['default']))
        _logger.info("Done")


def get_evaluation_argparser():
    import argparse
    #TODO factor out redundancies with get_default_argparser()
    standard_parser = get_default_argparser()
    parser = argparse.ArgumentParser(
        prog="morfessor-evaluate",
        epilog="""Simple usage example:

  %(prog)s gold_standard model1 model2
""",
        description=standard_parser.description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
    )

    add_arg = parser.add_argument_group('evaluation options').add_argument
    add_arg('--num-samples', dest='numsamples', type=int, metavar='<int>',
            default=10, help='number of samples to take for testing')
    add_arg('--sample-size', dest='samplesize', type=int, metavar='<int>',
            default=1000, help='size of each testing samples')

    add_arg = parser.add_argument_group('formatting options').add_argument
    add_arg('--format-string', dest='formatstring', metavar='<format>',
            help='Python new style format string used to report evaluation '
                 'results. The following variables are a value and and action '
                 'separated with and underscore. E.g. fscore_avg for the '
                 'average f-score. The available values are "precision", '
                 '"recall", "fscore", "samplesize" and the available actions: '
                 '"avg", "max", "min", "values", "count". A last meta-data '
                 'variable (without action) is "name", the filename of the '
                 'model See also the format-template option for predefined '
                 'strings')
    add_arg('--format-template', dest='template', metavar='<template>',
            default='default',
            help='Uses a template string for the format-string options. '
                 'Available templates are: default, table and latex. '
                 'If format-string is defined this option is ignored')

    add_arg = parser.add_argument_group('file options').add_argument
    add_arg('--construction-separator', dest="cseparator", type=_str,
            default=' ', metavar='<regexp>',
            help="construction separator for test segmentation files"
                 " (default '%(default)s')")
    add_arg('-e', '--encoding', dest='encoding', metavar='<encoding>',
            help="encoding of input and output files (if none is given, "
                 "both the local encoding and UTF-8 are tried)")

    add_arg = parser.add_argument_group('logging options').add_argument
    add_arg('-v', '--verbose', dest="verbose", type=int, default=1,
            metavar='<int>',
            help="verbose level; controls what is written to the standard "
                 "error stream or log file (default %(default)s)")
    add_arg('--logfile', dest='log_file', metavar='<file>',
            help="write log messages to file in addition to standard "
                 "error stream")

    add_arg = parser.add_argument_group('other options').add_argument
    add_arg('-h', '--help', action='help',
            help="show this help message and exit")
    add_arg('--version', action='version',
            version='%(prog)s ' + get_version(),
            help="show version number and exit")

    add_arg = parser.add_argument
    add_arg('goldstandard', metavar='<goldstandard>', nargs=1,
            help='gold standard file in standard annotation format')
    add_arg('models', metavar='<model>', nargs='+',
            help='model files to segment (either binary or Morfessor 1.0 style'
                 ' segmentation models).')
    add_arg('-t', '--testsegmentation', dest='test_segmentations', default=[],
            action='append',
            help='Segmentation of the test set. Note that all words in the '
                 'gold-standard must be segmented')

    return parser


def main_evaluation(args):
    """ Separate main for running evaluation and statistical significance
    testing. Takes as argument the results of an get_evaluation_argparser()
    """
    #TODO refactor out redundancies with main()
    if args.verbose >= 2:
        loglevel = logging.DEBUG
    elif args.verbose >= 1:
        loglevel = logging.INFO
    else:
        loglevel = logging.WARNING

    logging_format = '%(asctime)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    default_formatter = logging.Formatter(logging_format, date_format)
    plain_formatter = logging.Formatter('%(message)s')
    logging.basicConfig(level=loglevel)
    _logger.propagate = False  # do not forward messages to the root logger

    # Basic settings for logging to the error stream
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    ch.setFormatter(plain_formatter)
    _logger.addHandler(ch)

    # Settings for when log_file is present
    if args.log_file is not None:
        fh = logging.FileHandler(args.log_file, 'w')
        fh.setLevel(loglevel)
        fh.setFormatter(default_formatter)
        _logger.addHandler(fh)
        # If logging to a file, make INFO the highest level for the
        # error stream
        ch.setLevel(max(loglevel, logging.INFO))

    io = MorfessorIO(encoding=args.encoding)

    ev = MorfessorEvaluation(io.read_annotations_file(args.goldstandard[0]))

    results = []

    sample_size = args.samplesize
    num_samples = args.numsamples

    f_string = args.formatstring
    if f_string is None:
        f_string = FORMAT_STRINGS[args.template]

    for f in args.models:
        result = ev.evaluate_model(io.read_any_model(f),
                                   configuration=EvaluationConfig(num_samples,
                                                                  sample_size),
                                   meta_data={'name': os.path.basename(f)})
        results.append(result)
        print(result.format(f_string))

    io.construction_separator = args.cseparator
    for f in args.test_segmentations:
        segmentation = io.read_segmentation_file(f, False)
        result = ev.evaluate_segmentation(segmentation,
                                          configuration=
                                          EvaluationConfig(num_samples,
                                                           sample_size),
                                          meta_data={'name':
                                                     os.path.basename(f)})
        results.append(result)
        print(result.format(f_string))

    if len(results) > 1 and num_samples > 1:
        wsr = WilcoxonSignedRank()
        r = wsr.significance_test(results)
        WilcoxonSignedRank.print_table(r)
