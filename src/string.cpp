// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include "string.h"
#include "error.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>


std::string to_str(const std::vector<bool>& vv)
{
	std::deque<bool> v(vv.begin(), vv.end());
	return to_str(v.begin(), v.end());
}

std::string to_str(const Vec& v)
{
	std::ostringstream os;
	if(v.size() > 0)
		os << to_str(v[0]);
	for(size_t i = 1; i < v.size(); i++)
		os << "," << to_str(v[i]);
	return os.str();
}

std::string to_str(const Matrix& v)
{
	std::ostringstream os;
	for(size_t i = 0; i < v.rows(); i++)
	{
		if(i == 0)
			os << "[";
		else
			os << " ";
		os << to_str(v[i]);
		os << "\n";
	}
	os << "]";
	return os.str();
}
