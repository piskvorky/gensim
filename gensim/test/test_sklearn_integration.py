import six
import unittest
import numpy
import os
import codecs
import pickle

from scipy import sparse
try:
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.datasets import load_files
    from sklearn import linear_model
except ImportError:
    raise unittest.SkipTest("Test requires scikit-learn to be installed, which is not available")

from gensim.sklearn_integration.sklearn_wrapper_gensim_ldamodel import SklearnWrapperLdaModel
from gensim.sklearn_integration.sklearn_wrapper_gensim_lsimodel import SklearnWrapperLsiModel
from gensim.sklearn_integration.sklearn_wrapper_gensim_ldaseqmodel import SklLdaSeqModel
from gensim.corpora import Dictionary
from gensim import matutils

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
sstats_ldaseq = numpy.loadtxt(datapath_ldaseq('sstats_test.txt'))
dictionary_ldaseq = Dictionary(texts_ldaseq)
corpus_ldaseq = [dictionary_ldaseq.doc2bow(text) for text in texts_ldaseq]


class TestSklearnLDAWrapper(unittest.TestCase):
    def setUp(self):
        self.model = SklearnWrapperLdaModel(id2word=dictionary, num_topics=2, passes=100, minimum_probability=0, random_state=numpy.random.seed(0))
        self.model.fit(corpus)

    def testPrintTopic(self):
        topic = self.model.print_topics(2)

        for k, v in topic:
            self.assertTrue(isinstance(v, six.string_types))
            self.assertTrue(isinstance(k, int))

    def testTransform(self):
        texts_new = ['graph', 'eulerian']
        bow = self.model.id2word.doc2bow(texts_new)
        matrix = self.model.transform(bow)
        self.assertTrue(matrix.shape[0], 1)
        self.assertTrue(matrix.shape[1], self.model.num_topics)
        texts_new = [['graph', 'eulerian'], ['server', 'flow'], ['path', 'system']]
        bow = []
        for i in texts_new:
            bow.append(self.model.id2word.doc2bow(i))
        matrix = self.model.transform(bow)
        self.assertTrue(matrix.shape[0], 3)
        self.assertTrue(matrix.shape[1], self.model.num_topics)

    def testGetTopicDist(self):
        texts_new = ['graph', 'eulerian']
        bow = self.model.id2word.doc2bow(texts_new)
        doc_topics, word_topics, phi_values = self.model.get_topic_dist(bow, per_word_topics=True)

        for k, v in word_topics:
            self.assertTrue(isinstance(v, list))
            self.assertTrue(isinstance(k, int))
        for k, v in doc_topics:
            self.assertTrue(isinstance(v, float))
            self.assertTrue(isinstance(k, int))
        for k, v in phi_values:
            self.assertTrue(isinstance(v, list))
            self.assertTrue(isinstance(k, int))

    def testPartialFit(self):
        for i in range(10):
            self.model.partial_fit(X=corpus)  # fit against the model again
            doc = list(corpus)[0]  # transform only the first document
            transformed = self.model[doc]
            transformed_approx = matutils.sparse2full(transformed, 2)  # better approximation
        expected = [0.13, 0.87]
        passed = numpy.allclose(sorted(transformed_approx), sorted(expected), atol=1e-1)
        self.assertTrue(passed)

    def testCSRMatrixConversion(self):
        arr = numpy.array([[1, 2, 0], [0, 0, 3], [1, 0, 0]])
        sarr = sparse.csr_matrix(arr)
        newmodel = SklearnWrapperLdaModel(num_topics=2, passes=100)
        newmodel.fit(sarr)
        topic = newmodel.print_topics()
        for k, v in topic:
            self.assertTrue(isinstance(v, six.string_types))
            self.assertTrue(isinstance(k, int))

    def testPipeline(self):
        model = SklearnWrapperLdaModel(num_topics=2, passes=10, minimum_probability=0, random_state=numpy.random.seed(0))
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

        # updating multiple params
        param_dict = {"eval_every": 20, "decay": 0.7}
        self.model.set_params(**param_dict)
        model_params = self.model.get_params()
        for key in param_dict.keys():
            self.assertEqual(model_params[key], param_dict[key])


class TestSklearnLSIWrapper(unittest.TestCase):
    def setUp(self):
        self.model = SklearnWrapperLsiModel(id2word=dictionary, num_topics=2)
        self.model.fit(corpus)

    def testModelSanity(self):
        topic = self.model.print_topics(2)
        for k, v in topic:
            self.assertTrue(isinstance(v, six.string_types))
            self.assertTrue(isinstance(k, int))

    def testTransform(self):
        texts_new = ['graph', 'eulerian']
        bow = self.model.id2word.doc2bow(texts_new)
        matrix = self.model.transform(bow)
        self.assertTrue(matrix.shape[0], 1)
        self.assertTrue(matrix.shape[1], self.model.num_topics)
        texts_new = [['graph', 'eulerian'], ['server', 'flow'], ['path', 'system']]
        bow = []
        for i in texts_new:
            bow.append(self.model.id2word.doc2bow(i))
        matrix = self.model.transform(bow)
        self.assertTrue(matrix.shape[0], 3)
        self.assertTrue(matrix.shape[1], self.model.num_topics)

    def testPartialFit(self):
        for i in range(10):
            self.model.partial_fit(X=corpus)  # fit against the model again
            doc = list(corpus)[0]  # transform only the first document
            transformed = self.model[doc]
            transformed_approx = matutils.sparse2full(transformed, 2)  # better approximation
        expected = [1.39, 0.0]
        passed = numpy.allclose(sorted(transformed_approx), sorted(expected), atol=1e-1)
        self.assertTrue(passed)

    def testPipeline(self):
        model = SklearnWrapperLsiModel(num_topics=2)
        with open(datapath('mini_newsgroup'), 'rb') as f:
            compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        data = cache
        id2word = Dictionary(map(lambda x: x.split(), data.data))
        corpus = [id2word.doc2bow(i.split()) for i in data.data]
        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        text_lda = Pipeline((('features', model,), ('classifier', clf)))
        text_lda.fit(corpus, data.target)
        score = text_lda.score(corpus, data.target)
        self.assertGreater(score, 0.50)

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(num_topics=3)
        model_params = self.model.get_params()
        self.assertEqual(model_params["num_topics"], 3)

        # updating multiple params
        param_dict = {"chunksize": 10000, "decay": 0.9}
        self.model.set_params(**param_dict)
        model_params = self.model.get_params()
        for key in param_dict.keys():
            self.assertEqual(model_params[key], param_dict[key])


class TestSklLdaSeqModelWrapper(unittest.TestCase):
    def setUp(self):
        self.model = SklLdaSeqModel(id2word=dictionary_ldaseq, num_topics=2, time_slice=[10, 10, 11], initialize='own', sstats=sstats_ldaseq)
        self.model.fit(corpus_ldaseq)

    def testTransform(self):
        # transform one document
        doc = list(corpus_ldaseq)[0]
        vec = self.model.transform(doc)
        expected_vec = numpy.array([6.65778961e-04, 9.99334221e-01])
        self.assertTrue(numpy.allclose(vec, expected_vec))  # transformed entries must be equal up to sign

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(num_topics=3)
        model_params = self.model.get_params()
        self.assertEqual(model_params["num_topics"], 3)

        # updating multiple params
        param_dict = {"passes": 20, "chunksize": 200}
        self.model.set_params(**param_dict)
        model_params = self.model.get_params()
        for key in param_dict.keys():
            self.assertEqual(model_params[key], param_dict[key])


if __name__ == '__main__':
    unittest.main()
