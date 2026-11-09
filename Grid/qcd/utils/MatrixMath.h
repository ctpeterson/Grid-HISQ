/*************************************************************************************
Grid physics library, www.github.com/paboyle/Grid
Source file: ./lib/qcd/utils/MatrixMath.h
Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

Copyright (C) 2023 

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

#pragma once

#ifndef QCD_UTILS_MATRIXMATH_H
#define QCD_UTILS_MATRIXMATH_H

NAMESPACE_BEGIN(Grid);

const int NUM_JACOBI_SWEEP = 32; // LAPACK does ~30, so 32 is pretty safe
const int NUM_GRAM_SCHMIDT_SWEEP = 2; // Kahan's "twice is enough"

namespace MatrixMath {
#define IS_SCALAR(T) std::is_arithmetic_v<Realified<T>>
#define SAME_REGISTER(z, r) std::is_same_v<decltype((z).v), decltype((r).v)>
#define REGISTER_OP(op, z, r) binary<decltype((r).v)>((z).v, (r).v, op())
#define FOR_LANES(lane, T) for (int lane = 0; lane < T::Nsimd(); ++lane)

#define MATRIX_MATH_UNARY(name, expression)                                    \
template<class T>                                                              \
accelerator_inline T name(const T& a) {                                        \
  typedef typename GridTypeMapper<T>::scalar_type Scalar;                      \
  auto operation = [] accelerator (Scalar x) -> Scalar { return expression; }; \
  if constexpr (std::is_arithmetic<T>::value) return operation(a);             \
  else return SimdApply(operation, a);                                         \
}

#define MATRIX_MATH_BINARY(name, expression)                                             \
template<class T, class U>                                                               \
accelerator_inline T name(const T& a, const U& b) {                                      \
  typedef typename GridTypeMapper<T>::scalar_type Scalar;                                \
  auto operation = [] accelerator (Scalar x, Scalar y) -> Scalar { return expression; }; \
  if constexpr (std::is_arithmetic<T>::value) return operation(a, b);                    \
  else return SimdApplyBinop(operation, a, T(b));                                        \
}

MATRIX_MATH_UNARY(absolute, std::fabs(x))
MATRIX_MATH_UNARY(isfinite, std::isfinite(x))
MATRIX_MATH_BINARY(less, x < y)
MATRIX_MATH_BINARY(lessEqual, x <= y)
MATRIX_MATH_BINARY(greater, x > y)
MATRIX_MATH_BINARY(greaterEqual, x >= y)
MATRIX_MATH_BINARY(minimum, std::fmin(x, y))
MATRIX_MATH_BINARY(maximum, std::fmax(x, y))

// lane i of a SIMD value, read or assigned through lane(z, i); a scalar, as on accelerators, is its own lane
// (inside an accelerator_for body Grid's lambda parameter lane hides this name: write MatrixMath::lane there)
template<class T> 
struct LaneRef {
  T& z; int i;
  accelerator_inline operator typename T::scalar_type() const { return z.getlane(i); }
  accelerator_inline void operator=(const typename T::scalar_type& s) { z.putlane(s, i); }
};

template<class T, int N> 
struct LaneRef<iMatrix<T, N>> {
  iMatrix<T, N>& z; int i;
  accelerator_inline operator typename iMatrix<T, N>::scalar_object() const { return extractLane(i, z); }
  accelerator_inline void operator=(const typename iMatrix<T, N>::scalar_object& s) { insertLane(i, z, s); }
};

template<class V> 
accelerator_inline auto lane(const V& z, int i) 
{ if constexpr (std::is_arithmetic_v<V>) return z; else return z.getlane(i); }

template<class V> 
accelerator_inline decltype(auto) lane(V& z, int i) 
{ if constexpr (std::is_arithmetic_v<V>) return (z); else return LaneRef<V>{z, i}; }

// complex lane i of a Realified mask, which holds two real lanes per complex lane: reads the first, assigns both
template<class T> 
struct MaskRef {
  T& z; int i;
  accelerator_inline operator typename T::scalar_type() const { return z.getlane(2*i); }
  accelerator_inline void operator=(const typename T::scalar_type& s) { z.putlane(s, 2*i); z.putlane(s, 2*i + 1); }
};

template<class V> 
accelerator_inline auto maskLane(const V& z, int i) 
{ if constexpr (std::is_arithmetic_v<V>) return z; else return z.getlane(2*i); }

template<class V> 
accelerator_inline decltype(auto) maskLane(V& z, int i) 
{ if constexpr (std::is_arithmetic_v<V>) return (z); else return MaskRef<V>{z, i}; }

template<class T> 
accelerator_inline T select(const T& mask, const T& a, const T& b) {
  if constexpr (std::is_arithmetic<T>::value) return mask ? a : b;
  else {
    T result;
    for (int lane = 0; lane < T::Nsimd(); ++lane)
    { result.putlane(mask.getlane(lane) ? a.getlane(lane) : b.getlane(lane), lane); }
    return result;
  }
}

// some typedefs

template<class T>
using Realified = typename GridTypeMapper<T>::Realified;

template<class T>
using RealScalar = typename GridTypeMapper<Realified<T>>::scalar_type; // one real number, even for SIMD T

template <int N>
using IsSupported = std::enable_if_t<N == 2 || N == 3, int>;

template <class T>
using IsScalar = std::enable_if_t<IS_SCALAR(T), int>;

// lane i of a matrix: the scalar matrix in that lane, or the matrix itself when it is already scalar

template<class T, int N> 
accelerator_inline auto lane(const iMatrix<T, N>& a, int i) { if constexpr (IS_SCALAR(T)) return a; else return extractLane(i, a); }

template<class T, int N> 
accelerator_inline decltype(auto) lane(iMatrix<T, N>& a, int i) { if constexpr (IS_SCALAR(T)) return (a); else return LaneRef<iMatrix<T, N>>{a, i}; }

// basic matrix-scalar operations

template <class T, int N>
accelerator_inline iMatrix<T, N> operator*(const iMatrix<T, N>& a, const Realified<T>& s) {
  iMatrix<T, N> b;
  const T* x = &a(0, 0);
  T* y = &b(0, 0);

  for (int k = 0; k < N*N; ++k) {
    if constexpr (IS_SCALAR(T)) { y[k] = x[k]*s; }
    else if constexpr (SAME_REGISTER(x[k], s)) { y[k].v = REGISTER_OP(MultSIMD, x[k], s); }
    else { FOR_LANES(i, T) { lane(y[k], i) = lane(x[k], i)*lane(s, 2*i); } }
  }

  return b;
}

template <class T, int N>
accelerator_inline iMatrix<T, N> operator*(const Realified<T>& s, const iMatrix<T, N>& a) { return a*s; }

template <class T, int N>
accelerator_inline iMatrix<T, N> operator*(const iMatrix<T, N>& a, const T& s) { iScalar<T> c; c() = s; return a*c; }

template <class T, int N>
accelerator_inline iMatrix<T, N> operator*(const T& s, const iMatrix<T, N>& a) { return a*s; }

template <class T, int N>
accelerator_inline iMatrix<T, N> operator/(const iMatrix<T, N>& a, const Realified<T>& s) {
  iMatrix<T, N> b;
  const T* x = &a(0, 0);
  T* y = &b(0, 0);

  for (int k = 0; k < N*N; ++k) {
    if constexpr (IS_SCALAR(T)) { y[k] = x[k]/s; }
    else if constexpr (SAME_REGISTER(x[k], s)) { y[k].v = REGISTER_OP(DivSIMD, x[k], s); }
    else { FOR_LANES(i, T) { lane(y[k], i) = lane(x[k], i)/lane(s, 2*i); } }
  }

  return b;
}

template <class T, int N>
accelerator_inline iMatrix<T, N> operator/(const iMatrix<T, N>& a, const T& s) { iScalar<T> c; c() = s; return a/c; }

template <class T, int N>
accelerator_inline iMatrix<T, N> select(const Realified<T>& mask, const iMatrix<T, N>& a, const iMatrix<T, N>& b) {
  iMatrix<T, N> c;
  const T* x = &a(0, 0);
  const T* y = &b(0, 0);
  T* z = &c(0, 0);

  for (int k = 0; k < N*N; ++k) {
    if constexpr (IS_SCALAR(T)) { z[k] = mask ? x[k] : y[k]; }
    else { FOR_LANES(i, T) { lane(z[k], i) = lane(mask, 2*i) ? lane(x[k], i) : lane(y[k], i); } }
  }

  return c;
}

template <class T>
accelerator_inline RealScalar<T> epsilon() { return std::numeric_limits<RealScalar<T>>::epsilon(); }

// basic matrix procedures

template <class T, int N>
accelerator_inline Realified<T> norm2(const iMatrix<T, N>& a)
{ return toReal(TensorRemove(innerProduct(a, a))); }

template <class T, int N>
accelerator_inline Realified<T> maxNorm(const iMatrix<T, N>& a, const Realified<T>& minimum = 0) {
  Realified<T> s = 0.0;
  const T* x = &a(0,0);

  for (int k = 0; k < N*N; ++k) {
    auto re = toReal(x[k]);
    auto im = toReal(timesMinusI(x[k]));
    s = select(isfinite(re), maximum(s, abs(re)), Realified<T>(INFINITY));
    s = select(isfinite(im), maximum(s, abs(im)), Realified<T>(INFINITY));
  }

  return select(isfinite(s), maximum(s, minimum), Realified<T>(-1.0));
}

template <class T, int N>
accelerator_inline Realified<T> isFinite(const iMatrix<T, N>& a) {
  Realified<T> ok = 1.0f;
  const T* x = &a(0,0);
  for (int k = 0; k < N*N; ++k) { ok *= isfinite(toReal(x[k])) * isfinite(toReal(timesMinusI(x[k]))); }
  return ok;
}

template <class T, int N>
accelerator_inline iMatrix<T, N> identity() { iMatrix<T, N> r; r = 1.0; return r; }

// basic linear algebra procedures

template <class T, int N>
accelerator_inline iMatrix<T, N> gram(const iMatrix<T, N>& a) { return adj(a)*a; }

template <class T, int N, IsSupported<N> = 0>
accelerator_inline iMatrix<T, N> adjugate(const iMatrix<T, N>& a) {
  iMatrix<T, N> b;
  if constexpr (N == 2) {
    b(0, 0) =  a(1, 1); b(0, 1) = -a(0, 1);
    b(1, 0) = -a(1, 0); b(1, 1) =  a(0, 0);
  } else {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        int i1 = (i + 1) % N, i2 = (i + 2) % N;
        int j1 = (j + 1) % N, j2 = (j + 2) % N;
        b(i, j) = a(j1, i1)*a(j2, i2) - a(j1, i2)*a(j2, i1);
    } }
  }
  return b;
}

template <class T, int N>
accelerator_inline T determinant(const iMatrix<T, N>& a, const iMatrix<T, N>& adj) // adj is adjugate
{ T det = 0; for (int i = 0; i < N; ++i) { det += a(0, i)*adj(i, 0); } return det; }

template <class T, int N, IsSupported<N> = 0>
accelerator_inline iVector<Realified<T>, N> hermEigs(const iMatrix<T, N>& q) {
  iVector<Realified<T>, N> e;
  iMatrix<T, N> unity = identity<T, N>();
  Realified<T> mean = toReal(TensorRemove(trace(q)))/Realified<T>(N);
  iMatrix<T, N> b = q - mean*unity;

  if constexpr (N == 2) { Realified<T> r = sqrt(norm2(b)/Realified<T>(2)); e(0) = mean + r; e(1) = mean - r; } 
  else {
    Realified<T> p = sqrt(norm2(b)/Realified<T>(6));
    Realified<T> p3 = p*p*p;
    Realified<T> cn = Realified<T>(0.5)*toReal(determinant(b, adjugate(b)));
    Realified<T> cd = select(greater(p3, 0.0), p3, Realified<T>(1));
    Realified<T> phi = acos(maximum(minimum(cn/cd, 1.0), -1.0))/Realified<T>(3);
    e(0) = mean + Realified<T>(2)*p*cos(phi);
    e(2) = mean + Realified<T>(2)*p*cos(phi + Realified<T>(2*M_PI/3));
    e(1) = Realified<T>(3)*mean - e(0) - e(2);
  }

  return e;
}

template <class T, int N, IsSupported<N> = 0>
accelerator_inline Realified<T> sylvester(iMatrix<T, N>& x, const iMatrix<T, N>& h, const iMatrix<T, N>& c) {
  static_assert(N == 2 || N == 3, "sylvester: only N = 2 and N = 3 are implemented");
  
  iMatrix<T, N> adjH = adjugate(h);
  Realified<T> t = toReal(TensorRemove(trace(h)));
  Realified<T> s = toReal(TensorRemove(trace(adjH)));
  Realified<T> r = toReal(determinant(h, adjH));
  Realified<T> ok;

  if constexpr (N == 2) {
    ok = greater(r, 0.0);
    Realified<T> c0 = Realified<T>(0.5)/t; 
    Realified<T> c1 = c0/r;
    x = c0*c + c1*(adjH*c*adjH);
  } else {
    ok = greater(r, Realified<T>(1e-2)*(s*t));
    Realified<T> c2 = Realified<T>(0.5)/(s*t - r); 
    Realified<T> c3 = c2*t; 
    Realified<T> c1 = c3/r; 
    Realified<T> c0 = c2*(s + t*t);
    x = c0*c + c1*(adjH*c*adjH);
    x = x + c2*(h*c*h - adjH*c - c*adjH) - c3*(h*c + c*h);
  }
  
  return ok;
}

template <class T, int N, IsScalar<T> = 0>
accelerator_inline Realified<T> columnNorm(const iMatrix<T, N>& a, int j) {
  Realified<T> r = 0.0f;

  for (int i = 0; i < N; ++i) { r += norm(a(i, j)); }
  if (r >= std::numeric_limits<Realified<T>>::min() && r <= std::numeric_limits<Realified<T>>::max()) 
  { return sqrt(r); }

  r = 0.0f;
  for (int i = 0; i < N; ++i) { r = std::hypot(r, std::hypot(real(a(i, j)), imag(a(i, j)))); }
  return r; // recomputed to protect against underflow/overflow; rare but necessary for numerical stability
}

// Cayley-Hamilton unitary projection

template <class T, int N, IsSupported<N> = 0>
accelerator_inline Realified<T> isPolar(const iMatrix<T, N>& u, const iMatrix<T, N>& v) {
  // checks if v is unitary and u is close to v up to a Hermitian factor
  iMatrix<T, N> unity = identity<T, N>(), h = adj(v)*u;
  iMatrix<T, N> adjH;
  RealScalar<T> tol = std::fmax(RealScalar<T>(1e-12), 32.0f*epsilon<T>());
  Realified<T> ok = lessEqual(norm2(gram(v) - unity), tol*tol); // unitarity check

  ok *= lessEqual(norm2(h - adj(h)), tol*tol*norm2(u)); // hermitian check
  h = Realified<T>(0.5f)*(h + adj(h)); 
  adjH = adjugate(h);
  for (int i = 0; i < N; ++i) // lastly, check for positive-semidefiniteness
  { ok *= greaterEqual(toReal(h(i, i)), -tol)*greaterEqual(toReal(adjH(i, i)), -tol); }
  return ok*greaterEqual(toReal(determinant(h, adjH)), -tol);
}

template <class T, int N, IsSupported<N> = 0>
accelerator_inline Realified<T> checkEigs(
  const iVector<Realified<T>, N>& e,
  const iMatrix<T, N>& u,
  const RealScalar<T> relTol,
  const Realified<T>& minimum,
  const RealScalar<T> cutoff = 0.0f
) {
  T det = determinant(u, adjugate(u));
  Realified<T> det2 = toReal(adj(det)*det), product = 1.0f, ok = 1.0f;
  RealScalar<T> tol = relTol > 0.0f ? std::fmax(relTol, 8.0f*epsilon<T>()) : relTol;

  // conditions for falling back on SVD [Eqn. C22 of PRD(75)054502]
  for (int i = 0; i < N; ++i) { ok *= greater(e(i), 0.0f)*greaterEqual(e(i), minimum); product *= e(i); }
  ok *= greater(det2, 0.0f)*lessEqual(absolute(product - det2), tol*det2);

  // an eigenvalue within rounding of the force cutoff makes the filter decision uncertain
  if (cutoff > 0.0f) { ok *= greater(absolute(e(N - 1) - Realified<T>(cutoff)), 4.0f*sqrt(epsilon<T>())*absolute(e(0))); }
  return ok;
}

template <class T, int N, IsSupported<N> = 0>
accelerator_inline iMatrix<T, N> hermSqrt(const iMatrix<T, N>& q, const iVector<Realified<T>, N>& e) {
  iMatrix<T, N> unity = identity<T, N>();

  if constexpr (N == 2) {
    Realified<T> r0 = sqrt(e(0)), r1 = sqrt(e(1));
    return (r0*r1*unity + q)/(r0 + r1); // (u'u)^(1/2)
  } else {
    Realified<T> r0 = sqrt(e(0)), r1 = sqrt(e(1)), r2 = sqrt(e(2));
    Realified<T> us = r0 + r1 + r2, vs = r0*r1 + r0*r2 + r1*r2, ws = r0*r1*r2; // u,v,w [Eqn. C6 of PRD(75)054502]
    Realified<T> denom = (r0 + r1)*(r0 + r2)*(r1 + r2);
    return (us*ws/denom)*unity + ((us*us - vs)/denom)*q - gram(q)/denom; // (u'u)^(1/2)
  }
}

template <class T, int N, IsSupported<N> = 0>
accelerator_inline iMatrix<T, N> hermInvSqrt(const iMatrix<T, N>& q, const iVector<Realified<T>, N>& e) {
  iMatrix<T, N> unity = identity<T, N>();

  if constexpr (N == 2) {
    Realified<T> r0 = sqrt(e(0)), r1 = sqrt(e(1));
    return ((e(0) + e(1) + r0*r1)*unity - q)/(r0*r1*(r0 + r1)); // (u'u)^(-1/2)
  } else {
    // Cayley-Hamilton: "u, w" coefficients [Eqn. C6 of PRD(75)054502]
    Realified<T> r0 = sqrt(e(0)), r1 = sqrt(e(1)), r2 = sqrt(e(2));
    Realified<T> us = r0 + r1 + r2, ws = r0*r1*r2;
    Realified<T> denom = ws*(r0 + r1)*(r0 + r2)*(r1 + r2);

    // Cayley-Hamilton: "f0, f1, f2" coefficients [Eqn. C7 of PRD(75)054502; Eqn. 29 of PRD(82)074501]
    Realified<T> f0 = (ws*us*us + e(0)*r0*(e(1) + e(2)) + e(1)*r1*(e(0) + e(2)) + e(2)*r2*(e(0) + e(1)))/denom;
    Realified<T> f1 = -((e(0) + e(1) + e(2))*us + ws)/denom;
    Realified<T> f2 = us/denom;
    return f0*unity + f1*q + f2*gram(q); // (u'u)^(-1/2)
  }
}

template <class T, int N, IsSupported<N> = 0>
accelerator_inline Realified<T> polar(
  iMatrix<T, N>& v,
  const iMatrix<T, N>& u,
  const RealScalar<T> relTol = 1e-8,
  const RealScalar<T> minSigma = 0.0f
) {
  /**
   * @brief U(N) unitary projection via Cayley-Hamilton
   * @author Curtis Taylor Peterson
   * @details
   * This procedure implements a U(N) projection of a general complex NxN matrix
   * using Cayley-Hamilton. For details about the Cayley-Hamilton approach, see the
   * references provided above; namely the OG paper by Hasenfratz et al and later
   * work by the MILC collaboration. Please note that this procedure is modelled
   * after the approach taken in Quantum EXpressions by James Osborn and Xiao-Yong Jin.
   * Returns 0 for links that should fall back on SVD.
   */
  Realified<T> scale = maxNorm(u);
  Realified<T> ok = greater(scale, 0.0f), minimum = (minSigma/scale)*(minSigma/scale);
  iMatrix<T, N> a = u/scale, q = gram(a);
  iVector<Realified<T>, N> e = hermEigs(q); // Cayley-Hamilton: eigenvalues of q = u†u

  ok *= checkEigs(e, a, relTol, minimum);
  v = a*hermInvSqrt(q, e); // u (u'u)^(-1/2)
  return ok*isPolar(a, v);
}

// singular value decomposition

template <class T, int N, IsScalar<T> = 0>
accelerator_inline int jacobi(
  iMatrix<T, N>& u, 
  iMatrix<T, N>& v, 
  iVector<Realified<T>, N>& s,
  int p,
  int q
) {
  /**
   * @brief Perform Jacobi rotation
   * @author Curtis Taylor Peterson
   * @details
   * A Jacobi rotation r is a 2-dimensional rotation embedded in an NxN matrix that is constructed to zero out the
   * `(p, q)` element of the gram matrix `g` of `u diag(s)`; that is,
   * (1) g' := r^dag * g * r = r^dag * (u*diag(s))^dag * (u*diag(s)) * r
   * with g'(p, q) = 0. `v` is the unitary accumulation of such successive Jacobi rotations. Our first goal is to 
   * calculate the Jacobi rotation `r`, with
   *         |  ...           ...            ...           ...           ...  |
   *         |  ...       cos(theta)         ...  e^(i phi) sin(theta)   ...  |
   * (2) r = |  ...           ...            ...           ...           ...  |
   *         |  ... -e^(-i phi) sin(theta)   ...       cos(theta)        ...  |
   *         |  ...           ...            ...           ...           ...  |
   * with `e^(i phi)` some overall phase. Letting
   * (3a) u_p := column p of u
   * (3b) u_q := column q of u,
   * one has
   * (4) g(p, q) = s(p) s(q) <u_p, u_q> = s(p) s(q) rho e^(i phi)
   * with <u_p, u_q> := u_p^dag u_q and 
   * (5) rho := |<u_p, u_q>|; 
   * as such, g(p, q) ~ 0 if s(p) = 0, s(q) = 0, or rho ~ 0, so there's nothing to be done. Otherwise,
   * (6) tan(2 theta) = 2 s(p) s(q) rho / (s(q)^2 - s(p)^2)
   * zeroes out g(p, q), as desired. This on its own can overflow or underflow. To that end, define
   * (7a) gamma := s(p) s(q) rho / max(s(p), s(q))^2
   * (7b) delta := |s(q)^2 - s(p)^2| / max(s(p), s(q))^2,
   * which can be cleverly rewritten as
   * (8a) gamma = rho ratio
   * (8b) delta = 1 - ratio^2
   * with
   * (9) ratio := min(s(p), s(q)) / max(s(p), s(q)),
   * so that
   * (10) t := |tan(theta)| = 2 gamma / (delta + sqrt(delta^2 + 4 gamma^2))
   * from the classic double angle formula. Then
   * (11a) cos(theta) = 1 / sqrt(1 + t^2)
   * (11b) |sin(theta)| = t cos(theta),
   * with theta < 0 if s(p) > s(q) and theta > 0 otherwise. Our second goal is to apply `r`. `v` simply becomes
   * (12) v' = v * r.
   * `u diag(s)` is also multiplied by `r`, but its new columns are divided by the old s(p) and s(q) so that elements
   * of `s` are never multiplied together. Letting
   * (13) w := e^(i phi) |sin(theta)| / ratio = 2 cos(theta) <u_p, u_q> / (delta + sqrt(delta^2 + 4 gamma^2)),
   * this gives
   * (14a) u_p' = cos(theta) u_p + ratio^2 w^dag u_q
   * (14b) u_q' = cos(theta) u_q - w u_p
   * if s(p) > s(q) and
   * (15a) u_p' = cos(theta) u_p - w^dag u_q
   * (15b) u_q' = cos(theta) u_q + ratio^2 w u_p
   * otherwise. u_p' and u_q' are no longer of unit length, so, finally, for j = p, q,
   * (16a) s'(j) = s(j) ||u_j'||
   * (16b) u_j' <- u_j' / ||u_j'||.
   * This is one rotation of Hestenes' one-sided Jacobi SVD [Hestenes]; the scale-free form of Eqns (7)-(10) is
   * that of LAPACK's zgesvj [Drmač and Veselić].
   * Returns 0 if there was nothing to be done, 1 if `u`, `v`, and `s` were updated, and -1 if rho is not finite.
   */
  T inner = 0.0f;
  T rot, sin;
  Realified<T> sp = s(p), sq = s(q);
  Realified<T> tan, cos, denom;
  Realified<T> rho, ratio, gamma, delta;

  if (sp == 0 || sq == 0) { return 0; }
  for (int i = 0; i < N; i++) { inner += adj(u(i, p))*u(i, q); } // Eqn (4)
  rho = abs(inner); // Eqn (5)
  if (rho <= 8.0f*epsilon<T>()) { return 0; } else if (!std::isfinite(rho)) { return -1; }

  ratio = std::fmin(sp, sq) / std::fmax(sp, sq); // Eqn (9)
  delta = (1.0f - ratio)*(1.0f + ratio);         // Eqn (8b)
  gamma = rho * ratio;                           // Eqn (8a)

  denom = delta + sqrt(delta*delta + 4.0f*gamma*gamma); // Eqns (10), (13)
  tan = 2.0f*gamma/denom;               // Eqn (10)
  cos = 1.0f/sqrt(1.0f + tan*tan);      // Eqn (11a)
  rot = (2.0f*cos/denom)*inner;         // Eqn (13)
  sin = (sp > sq ? -ratio : ratio)*rot; // Eqns (2), (11b)

  for (int i = 0; i < N; ++i) {
    T ux = u(i, p), uy = u(i, q);
    T vx = v(i, p), vy = v(i, q);
    
    u(i, p) = cos*ux + (sp > sq ? ratio*ratio : -1.0f)*adj(rot)*uy; // Eqns (14a), (15a)
    u(i, q) = (sp > sq ? -1.0f : ratio*ratio)*rot*ux + cos*uy;      // Eqns (14b), (15b)
  
    v(i, p) = cos*vx - adj(sin)*vy; // Eqn (12)
    v(i, q) = sin*vx + cos*vy;      // Eqn (12)
  }

  for (int j : {p, q}) {
    Realified<T> length = columnNorm(u, j);
    s(j) *= length;                                                           // Eqn (16a)
    if (length > 0.0f) { for (int i = 0; i < N; ++i) { u(i, j) /= length; } } // Eqn (16b)
  }

  return 1;
}

template <class T, int N>
accelerator_inline void gramSchmidt(iMatrix<T, N>& u, int j) {
  /**
   * @brief Performs Gram-Schmidt orthogonalization on the `j`-th column of matrix `u`.
   * @author Curtis Taylor Peterson
   * @details
   * Classic Gram-Schmidt: iteratively projects the `j`-th column of `u` onto the subspace orthogonal to the 
   * previous columns and normalizes it. Two passes suffice; see Kahan's "twice is enough" [Parlett].
   */
  for (int sweep = 0; sweep < NUM_GRAM_SCHMIDT_SWEEP; ++sweep) {
    for (int k = 0; k < j; ++k) {
      T inner = 0.0f;
      for (int i = 0; i < N; ++i) { inner += adj(u(i, k))*u(i, j); }
      for (int i = 0; i < N; ++i) { u(i, j) -= inner*u(i, k); }
  } }
}

template <class T, int N, IsScalar<T> = 0>
accelerator_inline bool svd(
  iMatrix<T, N>& u,
  iMatrix<T, N>& v,
  iVector<Realified<T>, N>& s,
  const iMatrix<T, N>& a
) {
  /**
   * @brief One-sided Jacobi singular value decomposition (SVD)
   * @author Curtis Taylor Peterson
   * @details
   * One-sided Jacobi after Hestenes, in the manner of LAPACK's zgesvj [Hestenes; Drmač and Veselić].
   * Decomposes `a` into
   * (1) a = u * diag(s) * adj(v)
   * where `u` and `v` are unitary matrices and `s` contains the singular values of `a`. This is achieved by 
   * applying a sequence of Jacobi rotations to reduce `a` into a diagonal form, from which the singular values 
   * and unitary matrices are extracted. Most of the algorithm complexity lies in the careful application of the 
   * Jacobi rotations and the subsequent Gram-Schmidt cleanup to ensure numerical stability and orthogonality of 
   * the resulting unitary matrices; please see the detailed documentation calculating the Jacobi rotations and 
   * Gram-Schmidt cleanup for more information.
   */
  bool converged = false;
  iMatrix<T, N> unity = identity<T, N>();
  Realified<T> tol = std::fmax(Realified<T>(1e-13), 32.0f*epsilon<T>());
  Realified<T> tmp;

  // prep work: normalize columns
  u = a;
  v = unity;
  for (int j = 0; j < N; ++j) {
    s(j) = 1.0f;
    for (int k = 0; k < 2; ++k) {
      tmp = columnNorm(u, j);
      s(j) *= tmp;
      if (tmp > 0.0f) { for (int i = 0; i < N; ++i) { u(i, j) /= tmp; } }
  } }

  // one-sided Jacobi sweeps
  for (int sweep = 0; sweep < NUM_JACOBI_SWEEP; ++sweep) {
    int rot = 0;
    for (int p = 0; p < N - 1; ++p) { for (int q = p + 1; q < N; ++q) { rot |= jacobi(u, v, s, p, q); } }
    if (rot < 0) { return false; } else if (rot == 0) { converged = true; break; }
  }
  if (!converged) { return false; }

  // sort singular values in descending order
  for (int p = 0; p < N - 1; ++p) { 
    for (int q = p + 1; q < N; ++q) {
      if (s(p) < s(q)) {
        tmp = s(p);
        s(p) = s(q);
        s(q) = tmp;
        for (int i = 0; i < N; ++i) {
          T x = u(i, p), y = v(i, p);
          u(i, p) = u(i, q);
          u(i, q) = x;
          v(i, p) = v(i, q);
          v(i, q) = y;
  } } } }

  // Gram-Schmidt cleanup of columns with s(j) = 0
  for (int j = 0; j < N; ++j) {
    if (s(j) != 0.0f) { continue; }
    iVector<T, N> best;
    Realified<T> bestNrm = 0.0f;

    for (int axis = 0; axis < N; ++axis) {
      for (int i = 0; i < N; ++i) { u(i, j) = i == axis ? T(1.0f) : T(0.0f); }
      gramSchmidt(u, j);
      tmp = columnNorm(u, j);
      if (tmp > bestNrm) { bestNrm = tmp; for (int i = 0; i < N; ++i) { best(i) = u(i, j); } }
    }

    if (!(bestNrm)) { return false; }
    for (int i = 0; i < N; ++i) { u(i, j) = best(i)/bestNrm; }
  }

  tol *= N*tol;
  return (norm2(adj(u)*u - unity) <= tol) && (norm2(adj(v)*v - unity) <= tol);
}

// derivatives of unitary projection

template <class T, int N, IsSupported<N> = 0>
accelerator_inline Realified<T> polarDeriv(
  iMatrix<T, N>& f,
  const iMatrix<T, N>& g,
  const iMatrix<T, N>& v,
  const iMatrix<T, N>& u
) {
  /**
   * @brief Derivative of unitary projection
   * @author Curtis Taylor Peterson
   * @details
   * This is a very clever approach to calculating the derivative of the unitary
   * projection; it was invented by the authors of the original HISQ paper
   * [PRD75(2007)054502] and expanded upon by James Osborn and Xiao-Yong Jin in the
   * Quantum EXpressions code. The objective is to calculate
   * (1) dQ/dU = CZ + U dZ/dU,
   * where
   * (2) Z = (X'X)^{-1/2}
   * (3) Q = XZ,
   * (4) C = dX/dU.
   * We can solve for dZ/dU using the identity
   * (5) dZ/dU = -Q'CZ = Y dZ/dU + dZ/dU Y
   * with
   * (6) Y = (X'X)^{1/2}.
   * This is a so-called "Sylvester" system of equations, and it has an analytic
   * solution for N = 2 and N = 3 that is calculated in sylvester. Here it is arranged
   * to avoid forming Z with u = v h and the upstream gradient g = dS/dv,
   * (7) h x + x h = -i (c - c'), c = v'g,
   * (8) dS/du = i v x.
   */
  Realified<T> ok = 1.0f;
  iMatrix<T, N> h, c, x;

  h = adj(v)*u;                // (u'u)^(1/2)
  h = Realified<T>(0.5f)*(h + adj(h));
  c = adj(v)*g;
  c = timesMinusI(c - adj(c)); // right-hand side of Eqn. [7]
  ok *= sylvester(x, h, c);    // solve Sylvester [7]
  f = timesI(v*x);             // Eqn. [8]
  return ok*isFinite(f);
}

// the force filter [Eqn. 44 of PRD(82)074501]: its cutoff, and on return the lanes it was applied to
template <class T>
struct ForceFilter { RealScalar<T> cutoff = 0.0f; Realified<T> applied = 0.0f; };

template <class T, int N, IsSupported<N> = 0>
accelerator_inline Realified<T> polarDerivCayleyHamilton(
  iMatrix<T, N>& f,
  const iMatrix<T, N>& g,
  const iMatrix<T, N>& u,
  iVector<Realified<T>, N> e,
  ForceFilter<T>& filter
) {
  /**
   * @brief Derivative of unitary projection via Cayley-Hamilton
   * @author David Clarke and Curtis Taylor Peterson
   * @details
   * This procedure implements the derivative of the U(N) projection using the
   * Cayley-Hamilton approach. For details about this approach, see the references
   * provided above; namely the OG paper by Hasenfratz et al and later work by the
   * MILC collaboration. See polarDeriv for description of derivative.
   *
   * `e` holds the eigenvalues of u'u in descending order: from hermEigs when they
   * pass checkEigs [Eqn. C22 of PRD(75)054502], and otherwise the squared singular
   * values of u [Eqn. 43 of PRD(82)074501].
   */
  Realified<T> scale = maxNorm(u, sqrt(filter.cutoff));
  Realified<T> delta = (filter.cutoff/scale)/scale;
  Realified<T> ok = greater(scale, 0.0f);
  iMatrix<T, N> a = u/scale, unity = identity<T, N>(), q = gram(a), h, z, c, x;

  // force filter [Eqn. C23 of PRD(75)054502; Eqn. 44 of PRD(82)074501]
  // SVD replaces only the eigenvalues. The force filter shifts Q as well.
  for (int i = 0; i < N; ++i) { e(i) = (e(i)/scale)/scale; }
  filter.applied = less(e(N - 1), delta);
  delta = select(filter.applied, delta, Realified<T>(0.0f));
  for (int i = 0; i < N; ++i) { e(i) += delta; }
  q = q + delta*unity;

  // calculate derivative
  h = hermSqrt(q, e); // (u'u)^(1/2)
  z = adjugate(h);
  z = z/toReal(determinant(h, z)); // (u'u)^(-1/2)
  c = adj(a)*g;
  c = z*(c + adj(c))*z;
  sylvester(x, h, c);
  f = (g*z - a*x)/scale;
  return ok*isFinite(f);
}

// macro sanitization

#undef IS_SCALAR
#undef SAME_REGISTER
#undef REGISTER_OP
#undef FOR_LANES
#undef MATRIX_MATH_UNARY
#undef MATRIX_MATH_BINARY

// Complex multiply/add = 6/2 real operations.
// Special functions count as one; copies, signs, masks and communication are excluded.
template <int N>
struct DenseMatrixFlopModel {
  static constexpr uint64_t multiply = N*N*(8*N - 2);
  static constexpr uint64_t add = 2*N*N;
  static constexpr uint64_t scale = 2*N*N;
  static constexpr uint64_t inner = 8*N*N;
  static constexpr uint64_t trace = 2*(N - 1);
  static constexpr uint64_t adjugate = N == 3 ? 14*N*N : 0;
  static constexpr uint64_t determinant = 8*N;
  static constexpr uint64_t hermEigs = N == 3 ? 283 : 55;
  static constexpr uint64_t hermSqrt = N == 3 ? 310 : 28;
  static constexpr uint64_t hermInvSqrt = N == 3 ? 322 : 32;
  static constexpr uint64_t sylvester = N == 3 ? 1932 : 158;
  static constexpr uint64_t checkEigs = adjugate + determinant + 6 + N + 2;
  static constexpr uint64_t isPolar = 2*multiply + 3*add + scale + 3*inner
    + 3 + adjugate + determinant;
  static constexpr uint64_t polar = 3 + scale + multiply + hermEigs
    + checkEigs + hermInvSqrt + multiply + isPolar;
  static constexpr uint64_t polarDerivCayleyHamilton = 3 + scale + multiply + 3*N + scale + add
    + hermSqrt + adjugate + determinant + scale
    + 5*multiply + add + sylvester + add + scale;
  static constexpr uint64_t polarDeriv = 3*multiply + 2*add + scale + sylvester;
};

}

NAMESPACE_END(Grid);

#endif // QCD_UTILS_MATRIXMATH_H