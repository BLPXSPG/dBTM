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
   
   
#ifndef	_MATH_FUNC_H
#define	_MATH_FUNC_H


//*************************  asa032.h   ************************************//
double alngam ( double xvalue, int *ifault );
double gamain ( double x, double p, int *ifault );
void gamma_inc_values ( int *n_data, double *a, double *x, double *fx );
double r8_abs ( double x );
void timestamp ( void );


//*************************  asa103.cpp   ************************************//
double digama ( double x, int *ifault );
void psi_values ( int *n_data, double *x, double *fx );
//void timestamp ( void );


//*************************  asa121.cpp   ************************************//
//void timestamp ( void );
double trigam ( double x, int *ifault );
void trigamma_values ( int *n_data, double *x, double *fx );


#endif
