class BaseTopicModel(object):
    def print_topic(self, topicno, topn=10):
        """
        Return a single topic as a formatted string. See `show_topic()` for parameters.

        >>> lsimodel.print_topic(10, topn=5)
        '-0.340 * "category" + 0.298 * "$M$" + 0.183 * "algebra" + -0.174 * "functor" + -0.168 * "operator"'

        """
        return ' + '.join(['%.3f*"%s"' % (v, k) for k, v in self.show_topic(topicno, topn)])

    def print_topics(self, num_topics=20, num_words=10):
        """Alias for `show_topics()` that prints the `num_words` most
        probable words for `topics` number of topics to log.
        Set `topics=-1` to print all topics."""
        return self.show_topics(num_topics=num_topics, num_words=num_words, log=True)
