#pragma once

#include <fstream>
#include <sstream>
#include <vector>


class FastLineSentence {
public:
	explicit FastLineSentence(const std::string& filename);

	std::vector<std::string> ReadSentence();
	inline bool IsEof() const { return is_eof_; }
private:
	std::ifstream fs_;
	bool is_eof_;
};
