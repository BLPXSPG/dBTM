/* 
   Copyright (C) 2011 Knowledge Media Institute, The Open University. All rights reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
	 you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Author: Chenghua Lin, c.lin@open.ac.uk 
*/
   
   
#ifndef	_MAP_TYPE_H
#define	_MAP_TYPE_H


#include <map>
#include <iostream>

using namespace std;


struct Word_atr { 
	int id; // vocabulary index
	int freqCount;
	int polarity;
	bool passFilterFlag; // initially set to false 
};

struct Word_Prior_Attr { 
	int id; // prior sentiment label
	vector<double> labDist; // label distribution
};

// map of words/terms [string => int]
typedef map<string, int> mapword2id;
// map of words/terms [int => string]
typedef map<int, string> mapid2word;
// map of words/attributes_of_words [string => word_attr]
typedef map<string, Word_atr> mapword2atr;

// map of word / word prior info [string => sentiment lab ID, sentiment label distribition]
typedef map<string, Word_Prior_Attr > mapword2prior;

// map of doc / doc label distribution [string => doc label distribition]
typedef map<string, vector<double> > mapname2labs;

#endif
