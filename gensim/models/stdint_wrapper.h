/*
 * This file is here to support older versions of the MSVC compiler that don't
 * have stdint.h.
 */
#ifdef _MSC_VER
    #ifndef _MSC_STDINT_H_
        #if _MSC_VER < 1300
           typedef unsigned char     uint8_t;
           typedef unsigned int      uint32_t;
           typedef char              int8_t;
        #else
           typedef unsigned __int8   uint8_t;
           typedef unsigned __int32  uint32_t;
           typedef char              int8_t;
        #endif
    #endif
#else
   #include <stdint.h>
#endif
