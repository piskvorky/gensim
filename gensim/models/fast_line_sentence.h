#pragma once

#include <fstream>
#include <sstream>
#include <vector>


class FastLineSentence {
public:
    explicit FastLineSentence();
	explicit FastLineSentence(const std::string& filename, size_t offset = 0);

	std::vector<std::string> ReadSentence();
	inline bool IsEof() const { return is_eof_; }
	inline void Reset() { fs_.clear(); fs_.seekg(offset_); fs_.seekg(offset_); is_eof_ = false;  }

private:
    std::string filename_;
	std::ifstream fs_;
	size_t offset_;
	bool is_eof_;
};
