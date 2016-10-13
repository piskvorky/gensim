class BaseTopicModel():
    def print_topic(self, topicno, topn=10):
        """
        Return a single topic as a formatted string. See `show_topic()` for parameters.

        >>> lsimodel.print_topic(10, topn=5)
        '-0.340 * "category" + 0.298 * "$M$" + 0.183 * "algebra" + -0.174 * "functor" + -0.168 * "operator"'

        """
        return ' + '.join(['%.3f*"%s"' % (v, k) for k, v in self.show_topic(topicno, topn)])
