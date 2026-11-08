#pragma once

#include <fstream>
#include <sstream>
#include <vector>


class FastLineSentence {
public:
    explicit FastLineSentence() : offset_(0), end_offset_(-1), is_eof_(false) { }
    explicit FastLineSentence(const std::string& filename, size_t offset = 0, long long end_offset = -1) :
            filename_(filename), fs_(filename, std::ios::binary), offset_(offset),
            end_offset_(end_offset), is_eof_(false) {
        Reset();
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

        if (fs_.eof() || (end_offset_ >= 0 && fs_.tellg() >= end_offset_)) {
            is_eof_ = true;
        }
        return res;
    }

	inline bool IsEof() const { return is_eof_; }
	inline long long EndOffset() const { return end_offset_; }
	inline void Reset() {
        fs_.clear();
        fs_.seekg(offset_);
        is_eof_ = end_offset_ >= 0 && offset_ >= static_cast<size_t>(end_offset_);
    }

private:
    std::string filename_;
	std::ifstream fs_;
	size_t offset_;
    long long end_offset_;
	bool is_eof_;
};
