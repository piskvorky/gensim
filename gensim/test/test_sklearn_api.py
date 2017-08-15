import unittest
import numpy
import os
import codecs
import pickle

from scipy import sparse
try:
    from sklearn.pipeline import Pipeline
    from sklearn import linear_model, cluster
    from sklearn.exceptions import NotFittedError
except ImportError:
    raise unittest.SkipTest("Test requires scikit-learn to be installed, which is not available")

from gensim.sklearn_api.rpmodel import RpTransformer
from gensim.sklearn_api.ldamodel import LdaTransformer
from gensim.sklearn_api.lsimodel import LsiTransformer
from gensim.sklearn_api.ldaseqmodel import LdaSeqTransformer
from gensim.sklearn_api.w2vmodel import W2VTransformer
from gensim.sklearn_api.atmodel import AuthorTopicTransformer
from gensim.sklearn_api.d2vmodel import D2VTransformer
from gensim.sklearn_api.text2bow import Text2BowTransformer
from gensim.sklearn_api.tfidf import TfIdfTransformer
from gensim.sklearn_api.hdp import HdpTransformer
from gensim.sklearn_api.phrases import PhrasesTransformer
from gensim.corpora import mmcorpus, Dictionary
from gensim import matutils, models

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)
datapath_ldaseq = lambda fname: os.path.join(module_path, 'test_data/DTM', fname)

texts = [
    ['complier', 'system', 'computer'],
    ['eulerian', 'node', 'cycle', 'graph', 'tree', 'path'],
    ['graph', 'flow', 'network', 'graph'],
    ['loading', 'computer', 'system'],
    ['user', 'server', 'system'],
    ['tree', 'hamiltonian'],
    ['graph', 'trees'],
    ['computer', 'kernel', 'malfunction', 'computer'],
    ['server', 'system', 'computer'],
]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
author2doc = {'john': [0, 1, 2, 3, 4, 5, 6], 'jane': [2, 3, 4, 5, 6, 7, 8], 'jack': [0, 2, 4, 6, 8], 'jill': [1, 3, 5, 7]}

texts_new = texts[0:3]
author2doc_new = {'jill': [0], 'bob': [0, 1], 'sally': [1, 2]}
dictionary_new = Dictionary(texts_new)
corpus_new = [dictionary_new.doc2bow(text) for text in texts_new]

texts_ldaseq = [
    [u'senior', u'studios', u'studios', u'studios', u'creators', u'award', u'mobile', u'currently', u'challenges', u'senior', u'summary', u'senior', u'motivated', u'creative', u'senior'],
    [u'performs', u'engineering', u'tasks', u'infrastructure', u'focusing', u'primarily', u'programming', u'interaction', u'designers', u'engineers', u'leadership', u'teams', u'teams', u'crews', u'responsibilities', u'engineering', u'quality', u'functional', u'functional', u'teams', u'organizing', u'prioritizing', u'technical', u'decisions', u'engineering', u'participates', u'participates', u'reviews', u'participates', u'hiring', u'conducting', u'interviews'],
    [u'feedback', u'departments', u'define', u'focusing', u'engineering', u'teams', u'crews', u'facilitate', u'engineering', u'departments', u'deadlines', u'milestones', u'typically', u'spends', u'designing', u'developing', u'updating', u'bugs', u'mentoring', u'engineers', u'define', u'schedules', u'milestones', u'participating'],
    [u'reviews', u'interviews', u'sized', u'teams', u'interacts', u'disciplines', u'knowledge', u'skills', u'knowledge', u'knowledge', u'xcode', u'scripting', u'debugging', u'skills', u'skills', u'knowledge', u'disciplines', u'animation', u'networking', u'expertise', u'competencies', u'oral', u'skills', u'management', u'skills', u'proven', u'effectively', u'teams', u'deadline', u'environment', u'bachelor', u'minimum', u'shipped', u'leadership', u'teams', u'location', u'resumes', u'jobs', u'candidates', u'openings', u'jobs'],
    [u'maryland', u'client', u'producers', u'electricity', u'operates', u'storage', u'utility', u'retail', u'customers', u'engineering', u'consultant', u'maryland', u'summary', u'technical', u'technology', u'departments', u'expertise', u'maximizing', u'output', u'reduces', u'operating', u'participates', u'areas', u'engineering', u'conducts', u'testing', u'solve', u'supports', u'environmental', u'understands', u'objectives', u'operates', u'responsibilities', u'handles', u'complex', u'engineering', u'aspects', u'monitors', u'quality', u'proficiency', u'optimization', u'recommendations', u'supports', u'personnel', u'troubleshooting', u'commissioning', u'startup', u'shutdown', u'supports', u'procedure', u'operating', u'units', u'develops', u'simulations', u'troubleshooting', u'tests', u'enhancing', u'solving', u'develops', u'estimates', u'schedules', u'scopes', u'understands', u'technical', u'management', u'utilize', u'routine', u'conducts', u'hazards', u'utilizing', u'hazard', u'operability', u'methodologies', u'participates', u'startup', u'reviews', u'pssr', u'participate', u'teams', u'participate', u'regulatory', u'audits', u'define', u'scopes', u'budgets', u'schedules', u'technical', u'management', u'environmental', u'awareness', u'interfacing', u'personnel', u'interacts', u'regulatory', u'departments', u'input', u'objectives', u'identifying', u'introducing', u'concepts', u'solutions', u'peers', u'customers', u'coworkers', u'knowledge', u'skills', u'engineering', u'quality', u'engineering'],
    [u'commissioning', u'startup', u'knowledge', u'simulators', u'technologies', u'knowledge', u'engineering', u'techniques', u'disciplines', u'leadership', u'skills', u'proven', u'engineers', u'oral', u'skills', u'technical', u'skills', u'analytically', u'solve', u'complex', u'interpret', u'proficiency', u'simulation', u'knowledge', u'applications', u'manipulate', u'applications', u'engineering'],
    [u'calculations', u'programs', u'matlab', u'excel', u'independently', u'environment', u'proven', u'skills', u'effectively', u'multiple', u'tasks', u'planning', u'organizational', u'management', u'skills', u'rigzone', u'jobs', u'developer', u'exceptional', u'strategies', u'junction', u'exceptional', u'strategies', u'solutions', u'solutions', u'biggest', u'insurers', u'operates', u'investment'],
    [u'vegas', u'tasks', u'electrical', u'contracting', u'expertise', u'virtually', u'electrical', u'developments', u'institutional', u'utilities', u'technical', u'experts', u'relationships', u'credibility', u'contractors', u'utility', u'customers', u'customer', u'relationships', u'consistently', u'innovations', u'profile', u'construct', u'envision', u'dynamic', u'complex', u'electrical', u'management', u'grad', u'internship', u'electrical', u'engineering', u'infrastructures', u'engineers', u'documented', u'management', u'engineering', u'quality', u'engineering', u'electrical', u'engineers', u'complex', u'distribution', u'grounding', u'estimation', u'testing', u'procedures', u'voltage', u'engineering'],
    [u'troubleshooting', u'installation', u'documentation', u'bsee', u'certification', u'electrical', u'voltage', u'cabling', u'electrical', u'engineering', u'candidates', u'electrical', u'internships', u'oral', u'skills', u'organizational', u'prioritization', u'skills', u'skills', u'excel', u'cadd', u'calculation', u'autocad', u'mathcad', u'skills', u'skills', u'customer', u'relationships', u'solving', u'ethic', u'motivation', u'tasks', u'budget', u'affirmative', u'diversity', u'workforce', u'gender', u'orientation', u'disability', u'disabled', u'veteran', u'vietnam', u'veteran', u'qualifying', u'veteran', u'diverse', u'candidates', u'respond', u'developing', u'workplace', u'reflects', u'diversity', u'communities', u'reviews', u'electrical', u'contracting', u'southwest', u'electrical', u'contractors'],
    [u'intern', u'electrical', u'engineering', u'idexx', u'laboratories', u'validating', u'idexx', u'integrated', u'hardware', u'entails', u'planning', u'debug', u'validation', u'engineers', u'validation', u'methodologies', u'healthcare', u'platforms', u'brightest', u'solve', u'challenges', u'innovation', u'technology', u'idexx', u'intern', u'idexx', u'interns', u'supplement', u'interns', u'teams', u'roles', u'competitive', u'interns', u'idexx', u'interns', u'participate', u'internships', u'mentors', u'seminars', u'topics', u'leadership', u'workshops', u'relevant', u'planning', u'topics', u'intern', u'presentations', u'mixers', u'applicants', u'ineligible', u'laboratory', u'compliant', u'idexx', u'laboratories', u'healthcare', u'innovation', u'practicing', u'veterinarians', u'diagnostic', u'technology', u'idexx', u'enhance', u'veterinarians', u'efficiency', u'economically', u'idexx', u'worldwide', u'diagnostic', u'tests', u'tests', u'quality', u'headquartered', u'idexx', u'laboratories', u'employs', u'customers', u'qualifications', u'applicants', u'idexx', u'interns', u'potential', u'demonstrated', u'portfolio', u'recommendation', u'resumes', u'marketing', u'location', u'americas', u'verification', u'validation', u'schedule', u'overtime', u'idexx', u'laboratories', u'reviews', u'idexx', u'laboratories', u'nasdaq', u'healthcare', u'innovation', u'practicing', u'veterinarians'],
    [u'location', u'duration', u'temp', u'verification', u'validation', u'tester', u'verification', u'validation', u'middleware', u'specifically', u'testing', u'applications', u'clinical', u'laboratory', u'regulated', u'environment', u'responsibilities', u'complex', u'hardware', u'testing', u'clinical', u'analyzers', u'laboratory', u'graphical', u'interfaces', u'complex', u'sample', u'sequencing', u'protocols', u'developers', u'correction', u'tracking', u'tool', u'timely', u'troubleshoot', u'testing', u'functional', u'manual', u'automated', u'participate', u'ongoing'],
    [u'testing', u'coverage', u'planning', u'documentation', u'testing', u'validation', u'corrections', u'monitor', u'implementation', u'recurrence', u'operating', u'statistical', u'quality', u'testing', u'global', u'multi', u'teams', u'travel', u'skills', u'concepts', u'waterfall', u'agile', u'methodologies', u'debugging', u'skills', u'complex', u'automated', u'instrumentation', u'environment', u'hardware', u'mechanical', u'components', u'tracking', u'lifecycle', u'management', u'quality', u'organize', u'define', u'priorities', u'organize', u'supervision', u'aggressive', u'deadlines', u'ambiguity', u'analyze', u'complex', u'situations', u'concepts', u'technologies', u'verbal', u'skills', u'effectively', u'technical', u'clinical', u'diverse', u'strategy', u'clinical', u'chemistry', u'analyzer', u'laboratory', u'middleware', u'basic', u'automated', u'testing', u'biomedical', u'engineering', u'technologists', u'laboratory', u'technology', u'availability', u'click', u'attach'],
    [u'scientist', u'linux', u'asrc', u'scientist', u'linux', u'asrc', u'technology', u'solutions', u'subsidiary', u'asrc', u'engineering', u'technology', u'contracts'],
    [u'multiple', u'agencies', u'scientists', u'engineers', u'management', u'personnel', u'allows', u'solutions', u'complex', u'aeronautics', u'aviation', u'management', u'aviation', u'engineering', u'hughes', u'technical', u'technical', u'aviation', u'evaluation', u'engineering', u'management', u'technical', u'terminal', u'surveillance', u'programs', u'currently', u'scientist', u'travel', u'responsibilities', u'develops', u'technology', u'modifies', u'technical', u'complex', u'reviews', u'draft', u'conformity', u'completeness', u'testing', u'interface', u'hardware', u'regression', u'impact', u'reliability', u'maintainability', u'factors', u'standardization', u'skills', u'travel', u'programming', u'linux', u'environment', u'cisco', u'knowledge', u'terminal', u'environment', u'clearance', u'clearance', u'input', u'output', u'digital', u'automatic', u'terminal', u'management', u'controller', u'termination', u'testing', u'evaluating', u'policies', u'procedure', u'interface', u'installation', u'verification', u'certification', u'core', u'avionic', u'programs', u'knowledge', u'procedural', u'testing', u'interfacing', u'hardware', u'regression', u'impact', u'reliability', u'maintainability', u'factors', u'standardization', u'missions', u'asrc', u'subsidiaries', u'affirmative', u'employers', u'applicants', u'disability', u'veteran', u'technology', u'location', u'airport', u'bachelor', u'schedule', u'travel', u'contributor', u'management', u'asrc', u'reviews'],
    [u'technical', u'solarcity', u'niche', u'vegas', u'overview', u'resolving', u'customer', u'clients', u'expanding', u'engineers', u'developers', u'responsibilities', u'knowledge', u'planning', u'adapt', u'dynamic', u'environment', u'inventive', u'creative', u'solarcity', u'lifecycle', u'responsibilities', u'technical', u'analyzing', u'diagnosing', u'troubleshooting', u'customers', u'ticketing', u'console', u'escalate', u'knowledge', u'engineering', u'timely', u'basic', u'phone', u'functionality', u'customer', u'tracking', u'knowledgebase', u'rotation', u'configure', u'deployment', u'sccm', u'technical', u'deployment', u'deploy', u'hardware', u'solarcity', u'bachelor', u'knowledge', u'dell', u'laptops', u'analytical', u'troubleshooting', u'solving', u'skills', u'knowledge', u'databases', u'preferably', u'server', u'preferably', u'monitoring', u'suites', u'documentation', u'procedures', u'knowledge', u'entries', u'verbal', u'skills', u'customer', u'skills', u'competitive', u'solar', u'package', u'insurance', u'vacation', u'savings', u'referral', u'eligibility', u'equity', u'performers', u'solarcity', u'affirmative', u'diversity', u'workplace', u'applicants', u'orientation', u'disability', u'veteran', u'careerrookie'],
    [u'embedded', u'exelis', u'junction', u'exelis', u'embedded', u'acquisition', u'networking', u'capabilities', u'classified', u'customer', u'motivated', u'develops', u'tests', u'innovative', u'solutions', u'minimal', u'supervision', u'paced', u'environment', u'enjoys', u'assignments', u'interact', u'multi', u'disciplined', u'challenging', u'focused', u'embedded', u'developments', u'spanning', u'engineering', u'lifecycle', u'specification', u'enhancement', u'applications', u'embedded', u'freescale', u'applications', u'android', u'platforms', u'interface', u'customers', u'developers', u'refine', u'specifications', u'architectures'],
    [u'java', u'programming', u'scripts', u'python', u'debug', u'debugging', u'emulators', u'regression', u'revisions', u'specialized', u'setups', u'capabilities', u'subversion', u'technical', u'documentation', u'multiple', u'engineering', u'techexpousa', u'reviews'],
    [u'modeler', u'semantic', u'modeling', u'models', u'skills', u'ontology', u'resource', u'framework', u'schema', u'technologies', u'hadoop', u'warehouse', u'oracle', u'relational', u'artifacts', u'models', u'dictionaries', u'models', u'interface', u'specifications', u'documentation', u'harmonization', u'mappings', u'aligned', u'coordinate', u'technical', u'peer', u'reviews', u'stakeholder', u'communities', u'impact', u'domains', u'relationships', u'interdependencies', u'models', u'define', u'analyze', u'legacy', u'models', u'corporate', u'databases', u'architectural', u'alignment', u'customer', u'expertise', u'harmonization', u'modeling', u'modeling', u'consulting', u'stakeholders', u'quality', u'models', u'storage', u'agile', u'specifically', u'focus', u'modeling', u'qualifications', u'bachelors', u'accredited', u'modeler', u'encompass', u'evaluation', u'skills', u'knowledge', u'modeling', u'techniques', u'resource', u'framework', u'schema', u'technologies', u'unified', u'modeling', u'technologies', u'schemas', u'ontologies', u'sybase', u'knowledge', u'skills', u'interpersonal', u'skills', u'customers', u'clearance', u'applicants', u'eligibility', u'classified', u'clearance', u'polygraph', u'techexpousa', u'solutions', u'partnership', u'solutions', u'integration'],
    [u'technologies', u'junction', u'develops', u'maintains', u'enhances', u'complex', u'diverse', u'intensive', u'analytics', u'algorithm', u'manipulation', u'management', u'documented', u'individually', u'reviews', u'tests', u'components', u'adherence', u'resolves', u'utilizes', u'methodologies', u'environment', u'input', u'components', u'hardware', u'offs', u'reuse', u'cots', u'gots', u'synthesis', u'components', u'tasks', u'individually', u'analyzes', u'modifies', u'debugs', u'corrects', u'integrates', u'operating', u'environments', u'develops', u'queries', u'databases', u'repositories', u'recommendations', u'improving', u'documentation', u'develops', u'implements', u'algorithms', u'functional', u'assists', u'developing', u'executing', u'procedures', u'components', u'reviews', u'documentation', u'solutions', u'analyzing', u'conferring', u'users', u'engineers', u'analyzing', u'investigating', u'areas', u'adapt', u'hardware', u'mathematical', u'models', u'predict', u'outcome', u'implement', u'complex', u'database', u'repository', u'interfaces', u'queries', u'bachelors', u'accredited', u'substituted', u'bachelors', u'firewalls', u'ipsec', u'vpns', u'technology', u'administering', u'servers', u'apache', u'jboss', u'tomcat', u'developing', u'interfaces', u'firefox', u'internet', u'explorer', u'operating', u'mainframe', u'linux', u'solaris', u'virtual', u'scripting', u'programming', u'oriented', u'programming', u'ajax', u'script', u'procedures', u'cobol', u'cognos', u'fusion', u'focus', u'html', u'java', u'java', u'script', u'jquery', u'perl', u'visual', u'basic', u'powershell', u'cots', u'cots', u'oracle', u'apex', u'integration', u'competitive', u'package', u'bonus', u'corporate', u'equity', u'tuition', u'reimbursement', u'referral', u'bonus', u'holidays', u'insurance', u'flexible', u'disability', u'insurance'],
    [u'technologies', u'disability', u'accommodation', u'recruiter', u'techexpousa'],
    ['bank', 'river', 'shore', 'water'],
    ['river', 'water', 'flow', 'fast', 'tree'],
    ['bank', 'water', 'fall', 'flow'],
    ['bank', 'bank', 'water', 'rain', 'river'],
    ['river', 'water', 'mud', 'tree'],
    ['money', 'transaction', 'bank', 'finance'],
    ['bank', 'borrow', 'money'],
    ['bank', 'finance'],
    ['finance', 'money', 'sell', 'bank'],
    ['borrow', 'sell'],
    ['bank', 'loan', 'sell']
]
dictionary_ldaseq = Dictionary(texts_ldaseq)
corpus_ldaseq = [dictionary_ldaseq.doc2bow(text) for text in texts_ldaseq]

w2v_texts = [
    ['calculus', 'is', 'the', 'mathematical', 'study', 'of', 'continuous', 'change'],
    ['geometry', 'is', 'the', 'study', 'of', 'shape'],
    ['algebra', 'is', 'the', 'study', 'of', 'generalizations', 'of', 'arithmetic', 'operations'],
    ['differential', 'calculus', 'is', 'related', 'to', 'rates', 'of', 'change', 'and', 'slopes', 'of', 'curves'],
    ['integral', 'calculus', 'is', 'realted', 'to', 'accumulation', 'of', 'quantities', 'and', 'the', 'areas', 'under', 'and', 'between', 'curves'],
    ['physics', 'is', 'the', 'natural', 'science', 'that', 'involves', 'the', 'study', 'of', 'matter', 'and', 'its', 'motion', 'and', 'behavior', 'through', 'space', 'and', 'time'],
    ['the', 'main', 'goal', 'of', 'physics', 'is', 'to', 'understand', 'how', 'the', 'universe', 'behaves'],
    ['physics', 'also', 'makes', 'significant', 'contributions', 'through', 'advances', 'in', 'new', 'technologies', 'that', 'arise', 'from', 'theoretical', 'breakthroughs'],
    ['advances', 'in', 'the', 'understanding', 'of', 'electromagnetism', 'or', 'nuclear', 'physics', 'led', 'directly', 'to', 'the', 'development', 'of', 'new', 'products', 'that', 'have', 'dramatically', 'transformed', 'modern', 'day', 'society']
]

d2v_sentences = [models.doc2vec.TaggedDocument(words, [i]) for i, words in enumerate(w2v_texts)]

dict_texts = [
    'human interface computer',
    'survey user computer system response time',
    'eps user interface system',
    'system human system eps',
    'user response time',
    'trees',
    'graph trees',
    'graph minors trees',
    'graph minors survey'
]

phrases_sentences = [
    ['human', 'interface', 'computer'],
    ['survey', 'user', 'computer', 'system', 'response', 'time'],
    ['eps', 'user', 'interface', 'system'],
    ['system', 'human', 'system', 'eps'],
    ['user', 'response', 'time'],
    ['trees'],
    ['graph', 'trees'],
    ['graph', 'minors', 'trees'],
    ['graph', 'minors', 'survey'],
    ['graph', 'minors', 'survey', 'human', 'interface']
]


class TestLdaWrapper(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)  # set fixed seed to get similar values everytime
        self.model = LdaTransformer(id2word=dictionary, num_topics=2, passes=100, minimum_probability=0, random_state=numpy.random.seed(0))
        self.model.fit(corpus)

    def testTransform(self):
        texts_new = ['graph', 'eulerian']
        bow = self.model.id2word.doc2bow(texts_new)
        matrix = self.model.transform(bow)
        self.assertEqual(matrix.shape[0], 1)
        self.assertEqual(matrix.shape[1], self.model.num_topics)
        texts_new = [['graph', 'eulerian'], ['server', 'flow'], ['path', 'system']]
        bow = []
        for i in texts_new:
            bow.append(self.model.id2word.doc2bow(i))
        matrix = self.model.transform(bow)
        self.assertEqual(matrix.shape[0], 3)
        self.assertEqual(matrix.shape[1], self.model.num_topics)

    def testPartialFit(self):
        for i in range(10):
            self.model.partial_fit(X=corpus)  # fit against the model again
        doc = list(corpus)[0]  # transform only the first document
        transformed = self.model.transform(doc)
        expected = numpy.array([0.13, 0.87])
        passed = numpy.allclose(sorted(transformed[0]), sorted(expected), atol=1e-1)
        self.assertTrue(passed)

    def testConsistencyWithGensimModel(self):
        # training an LdaTransformer with `num_topics`=10
        self.model = LdaTransformer(id2word=dictionary, num_topics=10, passes=100, minimum_probability=0, random_state=numpy.random.seed(0))
        self.model.fit(corpus)

        # training a Gensim LdaModel with the same params
        gensim_ldamodel = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=100, minimum_probability=0, random_state=numpy.random.seed(0))

        texts_new = ['graph', 'eulerian']
        bow = self.model.id2word.doc2bow(texts_new)
        matrix_transformer_api = self.model.transform(bow)
        matrix_gensim_model = gensim_ldamodel[bow]
        matrix_gensim_model_dense = matutils.sparse2full(matrix_gensim_model, 10)  # convert into dense representation to be able to compare with transformer output
        passed = numpy.allclose(matrix_transformer_api, matrix_gensim_model_dense, atol=1e-1)
        self.assertTrue(passed)

    def testCSRMatrixConversion(self):
        numpy.random.seed(0)  # set fixed seed to get similar values everytime
        arr = numpy.array([[1, 2, 0], [0, 0, 3], [1, 0, 0]])
        sarr = sparse.csr_matrix(arr)
        newmodel = LdaTransformer(num_topics=2, passes=100)
        newmodel.fit(sarr)
        bow = [(0, 1), (1, 2), (2, 0)]
        transformed_vec = newmodel.transform(bow)
        expected_vec = numpy.array([0.35367903, 0.64632097])
        passed = numpy.allclose(transformed_vec, expected_vec, atol=1e-1)
        self.assertTrue(passed)

    def testPipeline(self):
        model = LdaTransformer(num_topics=2, passes=10, minimum_probability=0, random_state=numpy.random.seed(0))
        with open(datapath('mini_newsgroup'), 'rb') as f:
            compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        data = cache
        id2word = Dictionary(map(lambda x: x.split(), data.data))
        corpus = [id2word.doc2bow(i.split()) for i in data.data]
        numpy.random.mtrand.RandomState(1)  # set seed for getting same result
        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        text_lda = Pipeline((('features', model,), ('classifier', clf)))
        text_lda.fit(corpus, data.target)
        score = text_lda.score(corpus, data.target)
        self.assertGreater(score, 0.40)

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(num_topics=3)
        model_params = self.model.get_params()
        self.assertEqual(model_params["num_topics"], 3)
        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(corpus)
        self.assertEqual(getattr(self.model.gensim_model, 'num_topics'), 3)

        # updating multiple params
        param_dict = {"eval_every": 20, "decay": 0.7}
        self.model.set_params(**param_dict)
        model_params = self.model.get_params()
        for key in param_dict.keys():
            self.assertEqual(model_params[key], param_dict[key])
        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(corpus)
        self.assertEqual(getattr(self.model.gensim_model, 'eval_every'), 20)
        self.assertEqual(getattr(self.model.gensim_model, 'decay'), 0.7)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        texts_new = ['graph', 'eulerian']
        loaded_bow = model_load.id2word.doc2bow(texts_new)
        loaded_matrix = model_load.transform(loaded_bow)

        # sanity check for transformation operation
        self.assertEqual(loaded_matrix.shape[0], 1)
        self.assertEqual(loaded_matrix.shape[1], model_load.num_topics)

        # comparing the original and loaded models
        original_bow = self.model.id2word.doc2bow(texts_new)
        original_matrix = self.model.transform(original_bow)
        passed = numpy.allclose(loaded_matrix, original_matrix, atol=1e-1)
        self.assertTrue(passed)

    def testModelNotFitted(self):
        lda_wrapper = LdaTransformer(id2word=dictionary, num_topics=2, passes=100, minimum_probability=0, random_state=numpy.random.seed(0))
        texts_new = ['graph', 'eulerian']
        bow = lda_wrapper.id2word.doc2bow(texts_new)
        self.assertRaises(NotFittedError, lda_wrapper.transform, bow)


class TestLsiWrapper(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)  # set fixed seed to get similar values everytime
        self.model = LsiTransformer(id2word=dictionary, num_topics=2)
        self.model.fit(corpus)

    def testTransform(self):
        texts_new = ['graph', 'eulerian']
        bow = self.model.id2word.doc2bow(texts_new)
        matrix = self.model.transform(bow)
        self.assertEqual(matrix.shape[0], 1)
        self.assertEqual(matrix.shape[1], self.model.num_topics)
        texts_new = [['graph', 'eulerian'], ['server', 'flow'], ['path', 'system']]
        bow = []
        for i in texts_new:
            bow.append(self.model.id2word.doc2bow(i))
        matrix = self.model.transform(bow)
        self.assertEqual(matrix.shape[0], 3)
        self.assertEqual(matrix.shape[1], self.model.num_topics)

    def testPartialFit(self):
        for i in range(10):
            self.model.partial_fit(X=corpus)  # fit against the model again
        doc = list(corpus)[0]  # transform only the first document
        transformed = self.model.transform(doc)
        expected = numpy.array([1.39, 0.0])
        passed = numpy.allclose(transformed[0], expected, atol=1)
        self.assertTrue(passed)

    def testPipeline(self):
        model = LsiTransformer(num_topics=2)
        with open(datapath('mini_newsgroup'), 'rb') as f:
            compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        data = cache
        id2word = Dictionary(map(lambda x: x.split(), data.data))
        corpus = [id2word.doc2bow(i.split()) for i in data.data]
        numpy.random.mtrand.RandomState(1)  # set seed for getting same result
        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        text_lsi = Pipeline((('features', model,), ('classifier', clf)))
        text_lsi.fit(corpus, data.target)
        score = text_lsi.score(corpus, data.target)
        self.assertGreater(score, 0.50)

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(num_topics=3)
        model_params = self.model.get_params()
        self.assertEqual(model_params["num_topics"], 3)
        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(corpus)
        self.assertEqual(getattr(self.model.gensim_model, 'num_topics'), 3)

        # updating multiple params
        param_dict = {"chunksize": 10000, "decay": 0.9}
        self.model.set_params(**param_dict)
        model_params = self.model.get_params()
        for key in param_dict.keys():
            self.assertEqual(model_params[key], param_dict[key])
        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(corpus)
        self.assertEqual(getattr(self.model.gensim_model, 'chunksize'), 10000)
        self.assertEqual(getattr(self.model.gensim_model, 'decay'), 0.9)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        texts_new = ['graph', 'eulerian']
        loaded_bow = model_load.id2word.doc2bow(texts_new)
        loaded_matrix = model_load.transform(loaded_bow)

        # sanity check for transformation operation
        self.assertEqual(loaded_matrix.shape[0], 1)
        self.assertEqual(loaded_matrix.shape[1], model_load.num_topics)

        # comparing the original and loaded models
        original_bow = self.model.id2word.doc2bow(texts_new)
        original_matrix = self.model.transform(original_bow)
        passed = numpy.allclose(loaded_matrix, original_matrix, atol=1e-1)
        self.assertTrue(passed)

    def testModelNotFitted(self):
        lsi_wrapper = LsiTransformer(id2word=dictionary, num_topics=2)
        texts_new = ['graph', 'eulerian']
        bow = lsi_wrapper.id2word.doc2bow(texts_new)
        self.assertRaises(NotFittedError, lsi_wrapper.transform, bow)


class TestLdaSeqWrapper(unittest.TestCase):
    def setUp(self):
        self.model = LdaSeqTransformer(id2word=dictionary_ldaseq, num_topics=2, time_slice=[10, 10, 11], initialize='gensim')
        self.model.fit(corpus_ldaseq)

    def testTransform(self):
        # transforming two documents
        docs = []
        docs.append(list(corpus_ldaseq)[0])
        docs.append(list(corpus_ldaseq)[1])
        transformed_vecs = self.model.transform(docs)
        self.assertEqual(transformed_vecs.shape[0], 2)
        self.assertEqual(transformed_vecs.shape[1], self.model.num_topics)

        # transforming one document
        doc = list(corpus_ldaseq)[0]
        transformed_vecs = self.model.transform(doc)
        self.assertEqual(transformed_vecs.shape[0], 1)
        self.assertEqual(transformed_vecs.shape[1], self.model.num_topics)

    def testPipeline(self):
        numpy.random.seed(0)  # set fixed seed to get similar values everytime
        with open(datapath('mini_newsgroup'), 'rb') as f:
            compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        data = cache
        test_data = data.data[0:2]
        test_target = data.target[0:2]
        id2word = Dictionary(map(lambda x: x.split(), test_data))
        corpus = [id2word.doc2bow(i.split()) for i in test_data]
        model = LdaSeqTransformer(id2word=id2word, num_topics=2, time_slice=[1, 1, 1], initialize='gensim')
        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        text_ldaseq = Pipeline((('features', model,), ('classifier', clf)))
        text_ldaseq.fit(corpus, test_target)
        score = text_ldaseq.score(corpus, test_target)
        self.assertGreater(score, 0.50)

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(num_topics=3)
        model_params = self.model.get_params()
        self.assertEqual(model_params["num_topics"], 3)
        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(corpus_ldaseq)
        self.assertEqual(getattr(self.model.gensim_model, 'num_topics'), 3)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        doc = list(corpus_ldaseq)[0]
        loaded_transformed_vecs = model_load.transform(doc)

        # sanity check for transformation operation
        self.assertEqual(loaded_transformed_vecs.shape[0], 1)
        self.assertEqual(loaded_transformed_vecs.shape[1], model_load.num_topics)

        # comparing the original and loaded models
        original_transformed_vecs = self.model.transform(doc)
        passed = numpy.allclose(loaded_transformed_vecs, original_transformed_vecs, atol=1e-1)
        self.assertTrue(passed)

    def testModelNotFitted(self):
        ldaseq_wrapper = LdaSeqTransformer(num_topics=2)
        doc = list(corpus_ldaseq)[0]
        self.assertRaises(NotFittedError, ldaseq_wrapper.transform, doc)


class TestRpWrapper(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(13)
        self.model = RpTransformer(num_topics=2)
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.model.fit(self.corpus)

    def testTransform(self):
        # tranform two documents
        docs = []
        docs.append(list(self.corpus)[0])
        docs.append(list(self.corpus)[1])
        matrix = self.model.transform(docs)
        self.assertEqual(matrix.shape[0], 2)
        self.assertEqual(matrix.shape[1], self.model.num_topics)

        # tranform one document
        doc = list(self.corpus)[0]
        matrix = self.model.transform(doc)
        self.assertEqual(matrix.shape[0], 1)
        self.assertEqual(matrix.shape[1], self.model.num_topics)

    def testPipeline(self):
        numpy.random.seed(0)  # set fixed seed to get similar values everytime
        model = RpTransformer(num_topics=2)
        with open(datapath('mini_newsgroup'), 'rb') as f:
            compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        data = cache
        id2word = Dictionary(map(lambda x: x.split(), data.data))
        corpus = [id2word.doc2bow(i.split()) for i in data.data]
        numpy.random.mtrand.RandomState(1)  # set seed for getting same result
        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        text_rp = Pipeline((('features', model,), ('classifier', clf)))
        text_rp.fit(corpus, data.target)
        score = text_rp.score(corpus, data.target)
        self.assertGreater(score, 0.40)

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(num_topics=3)
        model_params = self.model.get_params()
        self.assertEqual(model_params["num_topics"], 3)
        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(self.corpus)
        self.assertEqual(getattr(self.model.gensim_model, 'num_topics'), 3)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        doc = list(self.corpus)[0]
        loaded_transformed_vecs = model_load.transform(doc)

        # sanity check for transformation operation
        self.assertEqual(loaded_transformed_vecs.shape[0], 1)
        self.assertEqual(loaded_transformed_vecs.shape[1], model_load.num_topics)

        # comparing the original and loaded models
        original_transformed_vecs = self.model.transform(doc)
        passed = numpy.allclose(loaded_transformed_vecs, original_transformed_vecs, atol=1e-1)
        self.assertTrue(passed)

    def testModelNotFitted(self):
        rpmodel_wrapper = RpTransformer(num_topics=2)
        doc = list(self.corpus)[0]
        self.assertRaises(NotFittedError, rpmodel_wrapper.transform, doc)


class TestWord2VecWrapper(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)
        self.model = W2VTransformer(size=10, min_count=0, seed=42)
        self.model.fit(texts)

    def testTransform(self):
        # tranform multiple words
        words = []
        words = words + texts[0]
        matrix = self.model.transform(words)
        self.assertEqual(matrix.shape[0], 3)
        self.assertEqual(matrix.shape[1], self.model.size)

        # tranform one word
        word = texts[0][0]
        matrix = self.model.transform(word)
        self.assertEqual(matrix.shape[0], 1)
        self.assertEqual(matrix.shape[1], self.model.size)

    def testConsistencyWithGensimModel(self):
        # training a W2VTransformer
        self.model = W2VTransformer(size=10, min_count=0, seed=42)
        self.model.fit(texts)

        # training a Gensim Word2Vec model with the same params
        gensim_w2vmodel = models.Word2Vec(texts, size=10, min_count=0, seed=42)

        word = texts[0][0]
        vec_transformer_api = self.model.transform(word)  # vector returned by W2VTransformer
        vec_gensim_model = gensim_w2vmodel[word]  # vector returned by Word2Vec
        passed = numpy.allclose(vec_transformer_api, vec_gensim_model, atol=1e-1)
        self.assertTrue(passed)

    def testPipeline(self):
        numpy.random.seed(0)  # set fixed seed to get similar values everytime
        model = W2VTransformer(size=10, min_count=1)
        model.fit(w2v_texts)

        class_dict = {'mathematics': 1, 'physics': 0}
        train_data = [
            ('calculus', 'mathematics'), ('mathematical', 'mathematics'), ('geometry', 'mathematics'), ('operations', 'mathematics'), ('curves', 'mathematics'),
            ('natural', 'physics'), ('nuclear', 'physics'), ('science', 'physics'), ('electromagnetism', 'physics'), ('natural', 'physics')
        ]
        train_input = list(map(lambda x: x[0], train_data))
        train_target = list(map(lambda x: class_dict[x[1]], train_data))

        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        clf.fit(model.transform(train_input), train_target)
        text_w2v = Pipeline((('features', model,), ('classifier', clf)))
        score = text_w2v.score(train_input, train_target)
        self.assertGreater(score, 0.40)

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(negative=20)
        model_params = self.model.get_params()
        self.assertEqual(model_params["negative"], 20)
        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(texts)
        self.assertEqual(getattr(self.model.gensim_model, 'negative'), 20)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        word = texts[0][0]
        loaded_transformed_vecs = model_load.transform(word)

        # sanity check for transformation operation
        self.assertEqual(loaded_transformed_vecs.shape[0], 1)
        self.assertEqual(loaded_transformed_vecs.shape[1], model_load.size)

        # comparing the original and loaded models
        original_transformed_vecs = self.model.transform(word)
        passed = numpy.allclose(loaded_transformed_vecs, original_transformed_vecs, atol=1e-1)
        self.assertTrue(passed)

    def testModelNotFitted(self):
        w2vmodel_wrapper = W2VTransformer(size=10, min_count=0, seed=42)
        word = texts[0][0]
        self.assertRaises(NotFittedError, w2vmodel_wrapper.transform, word)


class TestAuthorTopicWrapper(unittest.TestCase):
    def setUp(self):
        self.model = AuthorTopicTransformer(id2word=dictionary, author2doc=author2doc, num_topics=2, passes=100)
        self.model.fit(corpus)

    def testTransform(self):
        # transforming multiple authors
        author_list = ['jill', 'jack']
        author_topics = self.model.transform(author_list)
        self.assertEqual(author_topics.shape[0], 2)
        self.assertEqual(author_topics.shape[1], self.model.num_topics)

        # transforming one author
        jill_topics = self.model.transform('jill')
        self.assertEqual(jill_topics.shape[0], 1)
        self.assertEqual(jill_topics.shape[1], self.model.num_topics)

    def testPartialFit(self):
        self.model.partial_fit(corpus_new, author2doc=author2doc_new)

        # Did we learn something about Sally?
        output_topics = self.model.transform('sally')
        sally_topics = output_topics[0]  # getting the topics corresponding to 'sally' (from the list of lists)
        self.assertTrue(all(sally_topics > 0))

    def testPipeline(self):
        # train the AuthorTopic model first
        model = AuthorTopicTransformer(id2word=dictionary, author2doc=author2doc, num_topics=10, passes=100)
        model.fit(corpus)

        # create and train clustering model
        clstr = cluster.MiniBatchKMeans(n_clusters=2)
        authors_full = ['john', 'jane', 'jack', 'jill']
        clstr.fit(model.transform(authors_full))

        # stack together the two models in a pipeline
        text_atm = Pipeline((('features', model,), ('cluster', clstr)))
        author_list = ['jane', 'jack', 'jill']
        ret_val = text_atm.predict(author_list)
        self.assertEqual(len(ret_val), len(author_list))

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(num_topics=3)
        model_params = self.model.get_params()
        self.assertEqual(model_params["num_topics"], 3)
        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(corpus)
        self.assertEqual(getattr(self.model.gensim_model, 'num_topics'), 3)

        # updating multiple params
        param_dict = {"passes": 5, "iterations": 10}
        self.model.set_params(**param_dict)
        model_params = self.model.get_params()
        for key in param_dict.keys():
            self.assertEqual(model_params[key], param_dict[key])
        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(corpus)
        self.assertEqual(getattr(self.model.gensim_model, 'passes'), 5)
        self.assertEqual(getattr(self.model.gensim_model, 'iterations'), 10)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        author_list = ['jill']
        loaded_author_topics = model_load.transform(author_list)

        # sanity check for transformation operation
        self.assertEqual(loaded_author_topics.shape[0], 1)
        self.assertEqual(loaded_author_topics.shape[1], self.model.num_topics)

        # comparing the original and loaded models
        original_author_topics = self.model.transform(author_list)
        passed = numpy.allclose(loaded_author_topics, original_author_topics, atol=1e-1)
        self.assertTrue(passed)

    def testModelNotFitted(self):
        atmodel_wrapper = AuthorTopicTransformer(id2word=dictionary, author2doc=author2doc, num_topics=10, passes=100)
        author_list = ['jill', 'jack']
        self.assertRaises(NotFittedError, atmodel_wrapper.transform, author_list)


class TestD2VTransformer(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)
        self.model = D2VTransformer(min_count=1)
        self.model.fit(d2v_sentences)

    def testTransform(self):
        # tranform multiple documents
        docs = []
        docs.append(w2v_texts[0])
        docs.append(w2v_texts[1])
        docs.append(w2v_texts[2])
        matrix = self.model.transform(docs)
        self.assertEqual(matrix.shape[0], 3)
        self.assertEqual(matrix.shape[1], self.model.size)

        # tranform one document
        doc = w2v_texts[0]
        matrix = self.model.transform(doc)
        self.assertEqual(matrix.shape[0], 1)
        self.assertEqual(matrix.shape[1], self.model.size)

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(negative=20)
        model_params = self.model.get_params()
        self.assertEqual(model_params["negative"], 20)

        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(d2v_sentences)
        self.assertEqual(getattr(self.model.gensim_model, 'negative'), 20)

    def testPipeline(self):
        numpy.random.seed(0)  # set fixed seed to get similar values everytime
        model = D2VTransformer(min_count=1)
        model.fit(d2v_sentences)

        class_dict = {'mathematics': 1, 'physics': 0}
        train_data = [
            (['calculus', 'mathematical'], 'mathematics'), (['geometry', 'operations', 'curves'], 'mathematics'),
            (['natural', 'nuclear'], 'physics'), (['science', 'electromagnetism', 'natural'], 'physics')
        ]
        train_input = list(map(lambda x: x[0], train_data))
        train_target = list(map(lambda x: class_dict[x[1]], train_data))

        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        clf.fit(model.transform(train_input), train_target)
        text_w2v = Pipeline((('features', model,), ('classifier', clf)))
        score = text_w2v.score(train_input, train_target)
        self.assertGreater(score, 0.40)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        doc = w2v_texts[0]
        loaded_transformed_vecs = model_load.transform(doc)

        # sanity check for transformation operation
        self.assertEqual(loaded_transformed_vecs.shape[0], 1)
        self.assertEqual(loaded_transformed_vecs.shape[1], model_load.size)

        # comparing the original and loaded models
        original_transformed_vecs = self.model.transform(doc)
        passed = numpy.allclose(sorted(loaded_transformed_vecs), sorted(original_transformed_vecs), atol=1e-1)
        self.assertTrue(passed)

    def testConsistencyWithGensimModel(self):
        # training a D2VTransformer
        self.model = D2VTransformer(min_count=1)
        self.model.fit(d2v_sentences)

        # training a Gensim Doc2Vec model with the same params
        gensim_d2vmodel = models.Doc2Vec(d2v_sentences, min_count=1)

        doc = w2v_texts[0]
        vec_transformer_api = self.model.transform(doc)  # vector returned by D2VTransformer
        vec_gensim_model = gensim_d2vmodel[doc]  # vector returned by Doc2Vec
        passed = numpy.allclose(vec_transformer_api, vec_gensim_model, atol=1e-1)
        self.assertTrue(passed)

    def testModelNotFitted(self):
        d2vmodel_wrapper = D2VTransformer(min_count=1)
        self.assertRaises(NotFittedError, d2vmodel_wrapper.transform, 1)


class TestText2BowTransformer(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)
        self.model = Text2BowTransformer()
        self.model.fit(dict_texts)

    def testTransform(self):
        # tranform one document
        doc = ['computer system interface time computer system']
        bow_vec = self.model.transform(doc)[0]
        expected_values = [1, 1, 2, 2]  # comparing only the word-counts
        values = list(map(lambda x: x[1], bow_vec))
        self.assertEqual(sorted(expected_values), sorted(values))

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(prune_at=1000000)
        model_params = self.model.get_params()
        self.assertEqual(model_params["prune_at"], 1000000)

    def testPipeline(self):
        with open(datapath('mini_newsgroup'), 'rb') as f:
            compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        data = cache
        text2bow_model = Text2BowTransformer()
        lda_model = LdaTransformer(num_topics=2, passes=10, minimum_probability=0, random_state=numpy.random.seed(0))
        numpy.random.mtrand.RandomState(1)  # set seed for getting same result
        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        text_lda = Pipeline((('bow_model', text2bow_model), ('ldamodel', lda_model), ('classifier', clf)))
        text_lda.fit(data.data, data.target)
        score = text_lda.score(data.data, data.target)
        self.assertGreater(score, 0.40)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        doc = dict_texts[0]
        loaded_transformed_vecs = model_load.transform(doc)

        # comparing the original and loaded models
        original_transformed_vecs = self.model.transform(doc)
        self.assertEqual(original_transformed_vecs, loaded_transformed_vecs)

    def testModelNotFitted(self):
        text2bow_wrapper = Text2BowTransformer()
        self.assertRaises(NotFittedError, text2bow_wrapper.transform, dict_texts[0])


class TestTfIdfTransformer(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)
        self.model = TfIdfTransformer(normalize=True)
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.model.fit(self.corpus)

    def testTransform(self):
        # tranform one document
        doc = corpus[0]
        transformed_doc = self.model.transform(doc)
        expected_doc = [[(0, 0.5773502691896257), (1, 0.5773502691896257), (2, 0.5773502691896257)]]
        self.assertTrue(numpy.allclose(transformed_doc, expected_doc))

        # tranform multiple documents
        docs = [corpus[0], corpus[1]]
        transformed_docs = self.model.transform(docs)
        expected_docs = [[(0, 0.5773502691896257), (1, 0.5773502691896257), (2, 0.5773502691896257)],
            [(3, 0.44424552527467476), (4, 0.44424552527467476), (5, 0.3244870206138555), (6, 0.44424552527467476), (7, 0.3244870206138555), (8, 0.44424552527467476)]]
        self.assertTrue(numpy.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(numpy.allclose(transformed_docs[1], expected_docs[1]))

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(normalize=False)
        model_params = self.model.get_params()
        self.assertEqual(model_params["normalize"], False)

        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(self.corpus)
        self.assertEqual(getattr(self.model.gensim_model, 'normalize'), False)

    def testPipeline(self):
        with open(datapath('mini_newsgroup'), 'rb') as f:
            compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        data = cache
        id2word = Dictionary(map(lambda x: x.split(), data.data))
        corpus = [id2word.doc2bow(i.split()) for i in data.data]
        tfidf_model = TfIdfTransformer()
        tfidf_model.fit(corpus)
        lda_model = LdaTransformer(num_topics=2, passes=10, minimum_probability=0, random_state=numpy.random.seed(0))
        numpy.random.mtrand.RandomState(1)  # set seed for getting same result
        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        text_tfidf = Pipeline((('tfidf_model', tfidf_model), ('ldamodel', lda_model), ('classifier', clf)))
        text_tfidf.fit(corpus, data.target)
        score = text_tfidf.score(corpus, data.target)
        self.assertGreater(score, 0.40)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        doc = corpus[0]
        loaded_transformed_doc = model_load.transform(doc)

        # comparing the original and loaded models
        original_transformed_doc = self.model.transform(doc)
        self.assertEqual(original_transformed_doc, loaded_transformed_doc)

    def testModelNotFitted(self):
        tfidf_wrapper = TfIdfTransformer()
        self.assertRaises(NotFittedError, tfidf_wrapper.transform, corpus[0])


class TestHdpTransformer(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)
        self.model = HdpTransformer(id2word=dictionary)
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.model.fit(self.corpus)

    def testTransform(self):
        # tranform one document
        doc = self.corpus[0]
        transformed_doc = self.model.transform(doc)
        expected_doc = [[0.81043386270128193, 0.049357139518070477, 0.035840906753517532, 0.026542006926698079, 0.019925705902962578, 0.014776690981729117, 0.011068909979528148]]
        self.assertTrue(numpy.allclose(transformed_doc, expected_doc))

        # tranform multiple documents
        docs = [self.corpus[0], self.corpus[1]]
        transformed_docs = self.model.transform(docs)
        expected_docs = [[0.81043386270128193, 0.049357139518070477, 0.035840906753517532, 0.026542006926698079, 0.019925705902962578, 0.014776690981729117, 0.011068909979528148],
            [0.0368655605, 0.709055041, 0.194436428, 0.0151706795, 0.0113863652, 1.00000000e-12, 1.00000000e-12]]
        self.assertTrue(numpy.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(numpy.allclose(transformed_docs[1], expected_docs[1]))

    def testPartialFit(self):
        for i in range(10):
            self.model.partial_fit(X=self.corpus)  # fit against the model again
            doc = list(self.corpus)[0]  # transform only the first document
            transformed = self.model.transform(doc)
        expected = numpy.array([0.76777752, 0.01757334, 0.01600339, 0.01374061, 0.01275931, 0.01126313, 0.01058131, 0.01167185])
        passed = numpy.allclose(sorted(transformed[0]), sorted(expected), atol=1e-1)
        self.assertTrue(passed)

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(var_converge=0.05)
        model_params = self.model.get_params()
        self.assertEqual(model_params["var_converge"], 0.05)

        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(self.corpus)
        self.assertEqual(getattr(self.model.gensim_model, 'm_var_converge'), 0.05)

    def testPipeline(self):
        with open(datapath('mini_newsgroup'), 'rb') as f:
            compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        data = cache
        id2word = Dictionary(map(lambda x: x.split(), data.data))
        corpus = [id2word.doc2bow(i.split()) for i in data.data]
        model = HdpTransformer(id2word=id2word)
        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        text_lda = Pipeline((('features', model,), ('classifier', clf)))
        text_lda.fit(corpus, data.target)
        score = text_lda.score(corpus, data.target)
        self.assertGreater(score, 0.40)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        doc = corpus[0]
        loaded_transformed_doc = model_load.transform(doc)

        # comparing the original and loaded models
        original_transformed_doc = self.model.transform(doc)
        self.assertTrue(numpy.allclose(original_transformed_doc, loaded_transformed_doc))

    def testModelNotFitted(self):
        hdp_wrapper = HdpTransformer(id2word=dictionary)
        self.assertRaises(NotFittedError, hdp_wrapper.transform, corpus[0])


class TestPhrasesTransformer(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)
        self.model = PhrasesTransformer(min_count=1, threshold=1)
        self.model.fit(phrases_sentences)

    def testTransform(self):
        # tranform one document
        doc = phrases_sentences[-1]
        phrase_tokens = self.model.transform(doc)[0]
        expected_phrase_tokens = [u'graph_minors', u'survey', u'human_interface']
        self.assertEqual(phrase_tokens, expected_phrase_tokens)

    def testPartialFit(self):
        new_sentences = [
            ['world', 'peace', 'humans', 'world', 'peace', 'world', 'peace', 'people'],
            ['world', 'peace', 'people'],
            ['world', 'peace', 'humans']
        ]
        self.model.partial_fit(X=new_sentences)  # train model with new sentences

        doc = ['graph', 'minors', 'survey', 'human', 'interface', 'world', 'peace']
        phrase_tokens = self.model.transform(doc)[0]
        expected_phrase_tokens = [u'graph_minors', u'survey', u'human_interface', u'world_peace']
        self.assertEqual(phrase_tokens, expected_phrase_tokens)

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(progress_per=5000)
        model_params = self.model.get_params()
        self.assertEqual(model_params["progress_per"], 5000)

        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(phrases_sentences)
        self.assertEqual(getattr(self.model.gensim_model, 'progress_per'), 5000)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        doc = phrases_sentences[-1]
        loaded_phrase_tokens = model_load.transform(doc)

        # comparing the original and loaded models
        original_phrase_tokens = self.model.transform(doc)
        self.assertEqual(original_phrase_tokens, loaded_phrase_tokens)

    def testModelNotFitted(self):
        phrases_transformer = PhrasesTransformer()
        self.assertRaises(NotFittedError, phrases_transformer.transform, phrases_sentences[0])

if __name__ == '__main__':
    unittest.main()
