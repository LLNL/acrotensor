//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_STRING_UTIL_HPP
#define ACROBATIC_STRING_UTIL_HPP

#include <string>

namespace acro
{

inline void str_replace_all(std::string &instr, const std::string &keystr, const std::string &repstr)
{
   std::size_t instr_pos = instr.find(keystr);
   while (instr_pos != std::string::npos)
   {
      instr.replace(instr_pos, keystr.length(), repstr);
      instr_pos = instr.find(keystr);
   }
}

inline void str_replace_all(std::string &instr, const std::string &keystr, const int repint)
{
   str_replace_all(instr, keystr, std::to_string(repint));
}

}

#endif //ACROBATIC_STRING_UTIL_HPP