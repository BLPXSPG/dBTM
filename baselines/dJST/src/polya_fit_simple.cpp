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
   
   
//dataset dimension is S x nSamples, where S = 3; nSample = 100
//#include "model.h"
#include "polya_fit_simple.h"
#include "math_func.h"
#include <math.h>
#include <iostream>
#include <string>
#include <algorithm>


using namespace std;

int polya_fit_simple(int ** data, double * alpha, int _K, int _nSample) //(double ** ndl,  , int S, int M)
{ int K = _K;  // hyperparameter dimention 
  int nSample = _nSample; // total number of document in the case of JST 
  int polya_iter = 100000;  // maximum number of fixed point iterations
  int ifault1, ifault2;

  //double sum_alpha;
  double sum_alpha_old;
  double * old_alpha = NULL;
  double sum_g = 0; // sum_g = sum_digama(data[i][k] + old_alpha[k]),
  double sum_h = 0; // sum_h + sum_digama(data[i] + sum_alpha_old) , where data[i] = sum_data[i][k] for all k,
  double * data_row_sum = NULL; // the sum of the counts of each data sample P = {P_1, P_2,...,P_k}
  bool sat_state = false;
  int i, k, j;
  
  old_alpha = new double[K];
  for (k = 0; k < K; k++)
  { old_alpha[k] = 0;
  }
  
  data_row_sum = new double[nSample];
  for (i = 0; i < nSample; i++)
  { data_row_sum[i] = 0;
  }

  // data_row_sum
  for (i = 0; i < nSample; i++) 
  { for (k = 0; k < K; k++)
    {data_row_sum[i] += *(*(data+k)+i) ;  //data_row_sum[i] += *(*(data+i)+k) data[i][k] is the number of words in document i associated with sentiment label k, where i in the case of JST is document index, k is sentiment label index, 
    }
  }
  

 // simple fix point iteration
  for (i = 0; i < polya_iter; i++)
  {  // reset sum_alpha_old
	 sum_alpha_old = 0;
	 // update old_alpha after each iteration
	 for (j = 0; j < K; j++)
     { old_alpha[j] = *(alpha+j);
     }
 
     // calculate sum_alpha_old
     for (j = 0; j < K; j++)
  	 { sum_alpha_old += old_alpha[j];
	 }

	 for (k = 0; k < K; k++) // calcualte each {alpha_1, alpha_2,..., alpha_k} in term. 
     { //old_alpha[k] = alpha[k];
       //sum_alpha_old = 0;
	   sum_g = 0;
       sum_h = 0;           

       // calculate sum_g[k]
       for (j = 0; j < nSample; j++)
       { sum_g += digama( *(*(data+k)+j) + old_alpha[k], &ifault1);
       }

       // calculate sum_h
       for (j = 0; j < nSample; j++)
       { sum_h += digama(data_row_sum[j] + sum_alpha_old, &ifault1);
       }
       
	   // update alpha (new)
       *(alpha+k) = old_alpha[k] * (sum_g - nSample * digama(old_alpha[k], &ifault1)) / (sum_h - nSample * digama(sum_alpha_old, &ifault2));
     }
     
	 // terminate iteration ONLY if each dimension of {alpha_1, alpha_2, ... alpha_k} satisfy the termination criteria,  
     for (j = 0; j < K; j++) 
	 { if (fabs( *(alpha+j) - old_alpha[j]) > 0.0001) break; //  0.000001 must use 'fabs' function for floating point, abs function is for interger.  was set to 0.000001
	   if ( j == K-1) // maybe can remove this line. for checking whether all the dimension of the hyperparameter has been compared. 
	   { sat_state = true;
 		 cout<<"alpha_optimal: "<<*(alpha+0)<<"\t"<<*(alpha+1)<<"\t"<<*(alpha+2)<<endl;  //******* need to print to file 
		 cout<<"alpha_old: "<<old_alpha[0]<<"\t"<<old_alpha[1]<<"\t"<<old_alpha[2]<<endl;
		 cout<<"Difference: "<<fabs( *(alpha+0) - old_alpha[0])<<"\t"<<fabs( *(alpha+1) - old_alpha[1])<<"\t"<<fabs( *(alpha+2) - old_alpha[2])<<"\t"<<endl;
	   }
	 }

     // check whether to terminate the whole iteration
	 if(sat_state) 
	 { cout<<"Terminated at iteration: "<<i<<endl;
	   break;
	 }
	 else if(i == polya_iter-1)  cout<<"Haven't converged! Terminated at iteration: "<<i+1<<endl;


  } // for iteration


  for ( i = 0; i < K; i++)
  { cout<<*(alpha+i)<<"\t"; // ******* need to print to file 
    if (i == K-1) cout<<endl;
  }
 
 return 0;
}
