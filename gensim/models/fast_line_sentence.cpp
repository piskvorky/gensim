#include <stdexcept>
#include "fast_line_sentence.h"


FastLineSentence::FastLineSentence() : is_eof_(false) { }
FastLineSentence::FastLineSentence(const std::string& filename) : filename_(filename), fs_(filename), is_eof_(false) { }

std::vector<std::string> FastLineSentence::ReadSentence() {
    if (is_eof_) {
        return {};
    }
	std::string line, word;
	std::getline(fs_, line);
	std::vector<std::string> res;

	std::istringstream iss(line);
	while (iss >> word) {
		res.push_back(word);
	}

    if (fs_.eof()) {
        is_eof_ = true;
    }
	return res;
}
