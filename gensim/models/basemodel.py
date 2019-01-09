class BaseTopicModel(object):
    def print_topic(self, topicno, topn=10):
        """Get a single topic as a formatted string.

        Parameters
        ----------
        topicno : int
            Topic id.
        topn : int
            Number of words from topic that will be used.

        Returns
        -------
        str
            String representation of topic, like '-0.340 * "category" + 0.298 * "$M$" + 0.183 * "algebra" + ... '.

        """
        return ' + '.join('%.3f*"%s"' % (v, k) for k, v in self.show_topic(topicno, topn))

    def print_topics(self, num_topics=20, num_words=10):
        """Get the most significant topics (alias for `show_topics()` method).

        Parameters
        ----------
        num_topics : int, optional
            The number of topics to be selected, if -1 - all topics will be in result (ordered by significance).
        num_words : int, optional
            The number of words to be included per topics (ordered by significance).

        Returns
        -------
        list of (int, list of (str, float))
            Sequence with (topic_id, [(word, value), ... ]).

        """
        return self.show_topics(num_topics=num_topics, num_words=num_words, log=True)

    def get_topics(self):
        """Get words X topics matrix.

        Returns
        --------
        numpy.ndarray:
            The term topic matrix learned during inference, shape (`num_topics`, `vocabulary_size`).

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError
