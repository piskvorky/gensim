#pragma once

#include <fstream>
#include <sstream>
#include <vector>


class FastLineSentence {
public:
	explicit FastLineSentence(const std::string& filename);

	std::vector<std::string> ReadSentence();
private:
	std::ifstream fs_;
};