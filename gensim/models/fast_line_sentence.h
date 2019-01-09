#pragma once

#include <fstream>
#include <sstream>
#include <vector>


class FastLineSentence {
public:
    explicit FastLineSentence() : is_eof_(false) { }
	explicit FastLineSentence(const std::string& filename, size_t offset = 0) : filename_(filename),
                                                                                fs_(filename),
                                                                                offset_(offset),
                                                                                is_eof_(false) {
        fs_.seekg(offset_);
    }

	std::vector<std::string> ReadSentence() {
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

	inline bool IsEof() const { return is_eof_; }
	inline void Reset() { fs_.clear(); fs_.seekg(offset_); is_eof_ = false;  }

private:
    std::string filename_;
	std::ifstream fs_;
	size_t offset_;
	bool is_eof_;
};
