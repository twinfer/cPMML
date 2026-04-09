
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_OPTIONS_H
#define CPMML_OPTIONS_H

// STRING_OPTIMIZATION uses a hash-based encoding for string values which is
// faster but not reversible. Disabled to support TextIndex expressions that
// need to recover the original string from a Sample value.
// #define STRING_OPTIMIZATION
// #define DEBUG

#endif
