#include "linesentence.h"


FastLineSentence::FastLineSentence(const std::string& filename) : fs_(filename) { }

std::vector<std::string> FastLineSentence::ReadSentence() {
	std::string line, word;
	std::getline(fs_, line);
	std::vector<std::string> res;

	std::istringstream iss(line);
	while (iss >> word) {
		res.push_back(word);
	}

	return res;
}