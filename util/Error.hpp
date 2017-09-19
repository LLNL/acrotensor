//Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory
//Written by Aaron Fisher (fisher47@llnl.gov). LLNL-CODE-738419.
//All rights reserved.
//This file is part of Acrotensor. For details, see https://github.com/LLNL/acrotensor.

#ifndef ACROBATIC_ERROR_HPP
#define ACROBATIC_ERROR_HPP

#include <string>
#include <stdexcept>

namespace acro
{

#define GET_MACRO(_1,_2,NAME,...) NAME
#define ACROBATIC_ASSERT(...) GET_MACRO(__VA_ARGS__, ACROBATIC_ASSERT2, ACROBATIC_ASSERT1)(__VA_ARGS__)
#define ACROBATIC_ASSERT1(EX) if (!(EX)) throw_error(std::string(__FILE__) + ":  " + std::to_string(__LINE__));
#define ACROBATIC_ASSERT2(EX, STR) if (!(EX)) throw_error(std::string(__FILE__) + ":  " + std::to_string(__LINE__) + "  " + STR);

inline void throw_error(std::string error)
{
   throw std::runtime_error(error);
}

}

#endif //ACROBATIC_ERROR_HPP