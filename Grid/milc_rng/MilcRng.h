/*************************************************************************************
Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/milc_rng/MilcRng.h

Copyright (C) 2015

Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

See the full license in the file "LICENSE" in the top level distribution
directory
*************************************************************************************/
/*  END LEGAL */

/**
 * @file MilcRng.h
 * @author Curtis Taylor Peterson
 * @details
 * Adaptation of MILC's random number generator facilities. See Grid/milc_rng/README 
 * for licensing and attribution.
 * 
 * References:
 * * milc-qcd/milc_qcd/generic/ranstuff.c
 * * milc-qcd/milc_qcd/include/random.h
 * * milc-qcd/milc_qcd/libraries/rand_ahmat.c
 * * qex/src/rng/milcrng.nim
 */

#pragma once

#ifndef GRID_MILC_RNG_H
#define GRID_MILC_RNG_H

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <limits>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>

// "namespaces are one honking great idea — let's do more of those!" - Tim Peters
namespace MilcRng { 

//
// base types
//

static constexpr std::size_t registerWords = 7u;
static constexpr std::size_t stateWords = 12u;

template <class Real> struct Complex { Real re, im; };

template <class Real> 
struct AntiHermitian3x3 { Real m11, m22, m33; Complex<Real> m12, m13, m23; };

using MilcRngRegister = std::array<uint32_t, registerWords>;

using MilcRngState = std::array<uint32_t, stateWords>;

enum class MilcRngType { MilcRngV6 = 1, MilcRngV7 = 0 };

//
// constant metadata
//

static constexpr uint32_t idx1 = 69607u;
static constexpr uint32_t idx2 = 8u;
static constexpr uint32_t drawOffset = 12345u;
static constexpr uint32_t mask = 0x00FFFFFFu;
static constexpr uint32_t magic = 0x4d494c43u;

static constexpr float scale = 0x1p-24f;
static constexpr double sqrtOneThird = 0.57735026918962576451;
static constexpr double twiceSqrtOneThird = 1.15470053837925152902;

//
// MILC RNG engine
//

template <MilcRngType Rng = MilcRngType::MilcRngV7>
class MilcRngEngine {
/**
 * @class MilcRng
 * @brief MILC's random number generator
 * @author Curtis Taylor Peterson
 * @brief
 * MILC's random number generator is an xor of a forward shift register and an 
 * integer congruential generator. This is an adaptation of that generator.
 * Implements both the "v6" and "v7" variants of the generator. 
 */
public:
  static_assert(
    Rng == MilcRngType::MilcRngV6 || Rng == MilcRngType::MilcRngV7,
    "unsupported MILC RNG variant"
  );

public:
  using result_type = uint32_t;
  using State = MilcRngState;

public:
  static constexpr bool isMilcRngV6 = Rng == MilcRngType::MilcRngV6;
  static constexpr uint32_t seedOffset = isMilcRngV6 ? 100005u : 100000005u;
  static constexpr uint32_t format = 1u | (static_cast<uint32_t>(Rng) << 16);
  static constexpr uint64_t icShift = isMilcRngV6 ? 8u : 40u;

private:
  MilcRngRegister _register;
  uint32_t _multiplier = 0u;
  uint64_t _ic = 0u;

public: // seeding used in constructors
  void seed(uint32_t iseed = 1, uint32_t index = 0) {
    const uint32_t multiplier = idx1 + idx2*index;

    // forward shift register
    for (auto& word : _register) { 
      iseed = drawOffset + multiplier*iseed; 
      word = (iseed >> 8) & mask; 
    }
    iseed = drawOffset + multiplier*iseed;
    
    // integer congruential generator
    _ic = iseed;
    if (!isMilcRngV6 && (iseed & 0x80000000u)) 
    { _ic |= UINT64_C(0xffffffff00000000); }
    _multiplier = seedOffset + idx2*index;
  }

  void seed(std::seed_seq& seq) 
  { uint32_t word; seq.generate(&word, &word + 1); seed(word, 0); }

public:
  explicit MilcRngEngine(uint32_t iseed = 1, uint32_t index = 0) { seed(iseed, index); }
  explicit MilcRngEngine(std::seed_seq& seq) { seed(seq); }

public:
  static constexpr uint32_t min() { return 0u; }
  static constexpr uint32_t max() { return mask; }

public:
  uint32_t operator()() {
    // forward shift register
    const uint32_t register0 = (
      ((_register[5] >> 7) | (_register[6] << 17)) ^ 
      ((_register[4] >> 1) | (_register[5] << 23))
    ) & mask;
    for (int j = registerWords - 1; j != 0; --j) 
    { _register[j] = _register[j-1]; }
    _register[0] = register0;

    // integer congruential generator
    _ic = drawOffset + _multiplier*_ic;
    if (isMilcRngV6) { _ic = uint32_t(_ic); }

    // xor of both
    return register0 ^ uint32_t((_ic >> icShift) & mask);
  }

public:
  void discard(uint64_t count) { while (count--) { (*this)(); } }

public:
  MilcRngState state() const { 
    MilcRngState state{};
    state[0] = magic;
    state[1] = format;
    for (std::size_t j = 0; j != registerWords; ++j) 
    { state[j + 2] = _register[j]; }
    state[9] = uint32_t(_ic);
    state[10] = uint32_t(_ic >> 32);
    state[11] = _multiplier;
    return state;
  }

  void setState(const MilcRngState& state) { 
    uint32_t registerBits = 0u;

    // state validation
    if (state[0] != magic || state[1] != format) 
    { throw std::invalid_argument("MILC RNG state has different format or variant"); }
    
    for (int j = 0; j != registerWords; ++j) { 
      if (state[j + 2] > mask) 
      { throw std::invalid_argument("invalid MILC register word"); }
      registerBits |= state[j + 2] & (j == registerWords - 1 ? 0x7fu : mask);
    }

    if (registerBits == 0u)
    { throw std::invalid_argument("degenerate MILC register state"); }
  
    if (isMilcRngV6 && state[10] != 0) 
    { throw std::invalid_argument("invalid MilcRngV6 state"); }

    if ((state[11] & 7u) != 5u)
    { throw std::invalid_argument("invalid MILC multiplier"); }

    // set state
    for (int j = 0; j != registerWords; ++j) 
    { _register[j] = state[j + 2]; }
    _ic = uint64_t(state[9]) | (uint64_t(state[10]) << 32);
    _multiplier = state[11];
  }

public:
  friend bool operator==(const MilcRngEngine& lhs, const MilcRngEngine& rhs)
  { return lhs.state() == rhs.state(); }
  friend bool operator!=(const MilcRngEngine& lhs, const MilcRngEngine& rhs)
  { return !(lhs == rhs); }

public:
  friend std::ostream& operator<<(std::ostream& out, const MilcRngEngine& rng) {
    for (auto word : rng.state()) { out << word << " "; }
    return out;
  }

  friend std::istream& operator>>(std::istream& in, MilcRngEngine& rng) {
    MilcRngState state{};
    for (auto& word : state) { if (!(in >> word)) { return in; } }
    try { rng.setState(state); }
    catch (const std::invalid_argument&) { in.setstate(std::ios::failbit); }
    return in;
  }
};

//
// uniform distribution type
//

// forward declarations
template <class Real, class Rng> Real uniform(Rng& rng); 
template <class Real, class Rng> Real realGaussian(Rng& rng); 
template <class Real, class Rng> Complex<Real> complexGaussian(Rng& rng, double scale = 2.0); 
template <class Real, class Rng> AntiHermitian3x3<Real> antiHermitian3x3(Rng& rng); 

template <class Real>
class UniformDistribution {
public:
  UniformDistribution(Real a = 0.0, Real b = 1.0) { 
    if (a != 0.0 || b != 1.0) 
    { throw std::invalid_argument("MILC uniform requires [0, 1)"); }
  }

public:
  void reset() { }

public:
  template <class Rng>
  Real operator()(Rng& rng) { return uniform<Real>(rng); }
};

template <class Real>
class GaussianDistribution {
public:
  using result_type = double;

public:
  double _mean, _sdev;
  Real _spare{};
  bool _hasSpare = false;

public:
  double transform(Real x) const 
  { return _mean + _sdev*static_cast<double>(x); }

public:
  explicit GaussianDistribution(double mean = 0.0, double sdev = 1.0) 
    :_mean(mean), _sdev(sdev) { }

public:
  void reset() { _hasSpare = false; }

public:
  template <class Rng>
  double operator()(Rng& rng) {
    if constexpr (Rng::isMilcRngV6) {
      if (_hasSpare) { _hasSpare = false; return transform(_spare); }
      const auto z = complexGaussian<Real>(rng);
      _spare = z.im;
      _hasSpare = true;
      return transform(z.re);
    } else { return transform(realGaussian<Real>(rng)); }
  }

  template <class Rng>
  Complex<double> drawComplex(Rng& rng) {
    if constexpr (Rng::isMilcRngV6) { return {(*this)(rng), (*this)(rng)}; }
    else {
      const Complex<Real> z = complexGaussian<Real>(rng);
      return {transform(z.re), transform(z.im)};
    }
  }
};

//
// distribution sampling
//

template <class Real, class Rng>
Real uniform(Rng& rng) {
  static_assert(
    std::is_same<Real, float>::value || std::is_same<Real, double>::value,
    "MILC sampling precision must be float or double"
  );
  return Real(rng()) * Real(scale);
}

// "All models are wrong, but some are useful" - George E.P. Box
template <class Real, class Rng>
Complex<Real> complexGaussian(Rng& rng, double scale) { // classic Box-Muller
  // insisting on unit normalization (scale = 2.0) gives up bitwise reproducibility
  Real v1, v2;
  Real r, fac;

  do {
    v1 = 2.0*uniform<Real>(rng) - 1.0;
    v2 = 2.0*uniform<Real>(rng) - 1.0;
    r = v1*v1 + v2*v2;  
  } while (r >= 1.0);
  if (r == 0.0) { throw std::domain_error("MILC Gaussian has zero radius"); }
  
  fac = std::sqrt(-scale*std::log(static_cast<double>(r)) / static_cast<double>(r));
  return {Real(v2 * fac), Real(v1 * fac)};
}

template <class Real, class Rng>
Real realGaussian(Rng& rng) {
  static_assert(!Rng::isMilcRngV6, "MilcRngV6 uses complex pairs");
  return complexGaussian<Real>(rng).re;
}

template <class Real, class Rng>
AntiHermitian3x3<Real> antiHermitian3x3(Rng& rng) {
  AntiHermitian3x3<Real> matrix;
  
  // draws
  const Complex<Real> r = complexGaussian<Real>(rng, 1.0);
  const Complex<Real> x = complexGaussian<Real>(rng, 1.0);
  const Complex<Real> y = complexGaussian<Real>(rng, 1.0);
  const Complex<Real> z = complexGaussian<Real>(rng, 1.0);

  // diagonal matrix elements
  matrix.m11 = r.re + static_cast<Real>(sqrtOneThird) * r.im;
  matrix.m22 = -r.re + static_cast<Real>(sqrtOneThird) * r.im;
  matrix.m33 = -static_cast<Real>(twiceSqrtOneThird) * r.im;

  // upper triangular matrix elements
  if constexpr (Rng::isMilcRngV6) { 
    matrix.m12 = {x.re, y.im}; 
    matrix.m13 = {x.im, z.re}; 
    matrix.m23 = {y.re, z.im}; 
  } 
  else { matrix.m12 = x, matrix.m13 = y, matrix.m23 = z; }
  
  return matrix;
}

} // namespace Milc

#endif // GRID_MILC_RNG_H
