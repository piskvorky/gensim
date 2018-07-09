#pragma once

#include <fstream>
#include <sstream>
#include <vector>


class FastLineSentence {
public:
    explicit FastLineSentence();
	explicit FastLineSentence(const std::string& filename);

	std::vector<std::string> ReadSentence();
	inline bool IsEof() const { return is_eof_; }
	inline void Reset() { fs_ = std::ifstream(filename_); is_eof_ = false; }

private:
    std::string filename_;
	std::ifstream fs_;
	bool is_eof_;
};
