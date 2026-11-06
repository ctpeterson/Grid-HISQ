/*************************************************************************************
Grid physics library, www.github.com/paboyle/Grid
Source file: ./lib/qcd/utils/UnitaryProjection.h
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

#include <Grid/perfmon/Tracing.h>
#include <Grid/qcd/utils/MatrixMath.h>

#pragma once

#ifndef QCD_UTILS_UNITARYPROJECTION_H
#define QCD_UTILS_UNITARYPROJECTION_H

NAMESPACE_BEGIN(Grid);

// macros

// make scope more explicit
#define ACCELERATOR_SCOPE(exec) { exec; } \

// colour matrix mu of a site: direction mu of a gauge field, the link of a link field
template <class T, int N> 
accelerator_inline auto& siteLink(iVector<T, N>& site, int mu) { return site(mu)(); }

template <class T> 
accelerator_inline auto& siteLink(iScalar<T>& site, int) { return site()(); }

// coalesced read/write of colour matrix mu at a site
#define ACCREAD(site, mu) coalescedRead(siteLink((site), (mu)))
#define ACCWRITE(wsite, mu, u) coalescedWrite(siteLink((wsite), (mu)), (u))

// iterate the scalar lanes of a SIMD type on CPUs and the current lane on accelerators
#define simd_for(lane, Simd)                                                \
  for (int lane = acceleratorSIMTlane(Simd::Nsimd()); lane < Simd::Nsimd(); \
    lane += sizeof(Simd)/sizeof(coalescedRead(Simd())))

// failure signal for kernels

// write-once failure word in pinned host memory, raised by failing lanes; any failure aborts, so it is zeroed once and never reset
inline uint64_t* failWord() {
  static uint64_t* failed = [] { uint64_t* p = (uint64_t*)acceleratorAllocHost(sizeof(uint64_t)); *p = 0; return p; }();
  return failed;
}

// raise it for the rejected lanes of a Realified mask: one thread per lane on accelerators, a relaxed atomic
// store from the threads sharing the word on CPUs
template <class Mask>
accelerator_inline void raiseFailure(uint64_t* failed, const Mask& ok) {
  if constexpr (std::is_arithmetic_v<Mask>) { if (ok == 0.0) *failed = 1; }
  else for (int i = 0; i < Mask::Nsimd()/2; ++i) 
  { if (MatrixMath::maskLane(ok, i) == 0.0) { __atomic_store_n(failed, 1, __ATOMIC_RELAXED); } }
}

// one more on a counter in device memory, from a kernel thread
accelerator_inline void count(uint64_t& n) {
#if defined(GRID_SIMT) && !defined(GRID_SYCL)
  atomicAdd((unsigned long long*)&n, 1ull);
#else
  __atomic_fetch_add(&n, 1, __ATOMIC_RELAXED);
#endif
}

// useful data structures

// MILC's hisq_svd_counter and hisq_force_filter_counter: links that used the singular value decomposition, and
// derivative links the force filter shifted. The counters live in device memory, raised by kernel threads;
// read() copies them back for this rank since the last reset (GlobalSum them for a run total)
struct UnitaryProjectionReport {
  uint64_t svd = 0, filter = 0;

  // from a kernel thread: one more link through the SVD, and the lanes of a mask the force filter was applied to
  accelerator_inline void countSVD() { count(svd); }
  template <class Mask> accelerator_inline void countFilter(const Mask& applied) {
    if constexpr (std::is_arithmetic_v<Mask>) { if (applied != 0.0) count(filter); }
    else for (int i = 0; i < Mask::Nsimd()/2; ++i) if (MatrixMath::maskLane(applied, i) != 0.0) count(filter);
  }

  static UnitaryProjectionReport* counters() {
    static UnitaryProjectionReport* c = [] {
      auto* p = (UnitaryProjectionReport*)acceleratorAllocDevice(sizeof(UnitaryProjectionReport));
      acceleratorMemSet(p, 0, sizeof(UnitaryProjectionReport)); 
      return p;
    }();
    return c;
  }

  static UnitaryProjectionReport read(bool reset = true) {
    UnitaryProjectionReport r; 
    acceleratorCopyFromDevice(counters(), &r, sizeof(r));
    if (reset) { acceleratorMemSet(counters(), 0, sizeof(r)); }
    return r;
  }
};

enum UnitaryProjectionMethod { CayleyHamiltonProjection, SingularValueDecompositionProjection };

enum UnitaryProjectionDerivativeMethod { CayleyHamiltonDerivative, SylvesterDerivative };

struct UnitaryProjectionContext {
  UnitaryProjectionMethod projectionMethod;
  UnitaryProjectionDerivativeMethod derivativeMethod;

  RealD derivativeEigenvalueCutoff = 5e-5;
  RealD relativeSVDTolerance = 1e-8;
  RealD absoluteSVDTolerance = 1e-8;

  bool backupSVD = true;
  bool svdOnlyDerivative = false;

  UnitaryProjectionContext(
    UnitaryProjectionMethod projectionMethod = CayleyHamiltonProjection,
    UnitaryProjectionDerivativeMethod derivativeMethod = SylvesterDerivative
  ): projectionMethod(projectionMethod), derivativeMethod(derivativeMethod) { }
  
  UnitaryProjectionContext(
    UnitaryProjectionDerivativeMethod derivativeMethod,
    UnitaryProjectionMethod projectionMethod = CayleyHamiltonProjection
  ): projectionMethod(projectionMethod), derivativeMethod(derivativeMethod) { }
  
  bool operator==(const UnitaryProjectionContext& other) const {
    return projectionMethod == other.projectionMethod &&
           derivativeMethod == other.derivativeMethod &&
           derivativeEigenvalueCutoff == other.derivativeEigenvalueCutoff &&
           relativeSVDTolerance == other.relativeSVDTolerance &&
           absoluteSVDTolerance == other.absoluteSVDTolerance &&
           backupSVD == other.backupSVD &&
           svdOnlyDerivative == other.svdOnlyDerivative;
  }

  bool operator!=(const UnitaryProjectionContext& other) const { return !(*this == other); }

  void setDerivativeEigenvalueCutoff(RealD cutoff) { 
    GRID_ASSERT(derivativeMethod == CayleyHamiltonDerivative);
    GRID_ASSERT(cutoff >= 0.0);
    derivativeEigenvalueCutoff = cutoff; 
  }

  void setSVDOnlyDerivative(bool svdOnly) { 
    GRID_ASSERT(derivativeMethod == CayleyHamiltonDerivative);
    svdOnlyDerivative = svdOnly; 
  }

  void setBackupSVD(bool backup) { 
    GRID_ASSERT(projectionMethod == CayleyHamiltonProjection);
    backupSVD = backup; 
  }

  void setRelativeSVDTolerance(RealD tol) { 
    GRID_ASSERT(projectionMethod == CayleyHamiltonProjection);
    relativeSVDTolerance = tol; 
  }

  void setAbsoluteSVDTolerance(RealD tol) { 
    GRID_ASSERT(projectionMethod == CayleyHamiltonProjection);
    absoluteSVDTolerance = tol; 
  }
};

// unitary projection class

template<class Gimpl> 
class UnitaryProjection: public Gimpl {
/**
 * @class Grid::UnitaryProjection
 * @brief Defines object representing unitary projection
 * @author Curtis Taylor Peterson
 * @details
 * Class for performing unitary projection of gauge links. The projection is performed using Cayley-Hamilton as a 
 * default and optionally Jacobi-based singular value decomposition (SVD) as a fallback.
 * 
 * References:
 * * Hasenfratz, A. et al.: https://doi.org/10.1103/PhysRevD.78.014515
 * * MILC Collaboration: https://doi.org/10.1103/PhysRevD.75.054502
 * * Quantum EXpressions (QEX): https://github.com/jcosborn/qex
 * * QOPQDP [SciDAC]: https://github.com/usqcd-software/qopqdp
 * 
 * Acknowledgements:
 *   Curtis Taylor Peterson would like to thank James Osborn and Xiao-Yong Jin for 
 *   developing/testing the implementations of unitary projection in Quantum 
 *   EXpressions, from which this implementation is based and has been tested against.
 *   
 *   This material is based upon work supported by the U.S. Department of Energy, 
 *   Office of Science, Office of Advanced Scientific Computing Research, Scientific 
 *   Discovery through Advanced Computing (SciDAC) program.
 * 
 *   Michael Lynch added support for proper precision inhertience
*/
public: 
  INHERIT_GIMPL_TYPES(Gimpl);
  typedef typename RealPart<Scalar>::type RealScalar;
  typedef decltype(coalescedRead(Simd())) AccComplex;
  
public:
  template <class Field> 
  static constexpr int Nlinks = sizeof(typename Field::vector_object)/sizeof(typename GaugeLinkField::vector_object); // Nd or 1

private:
  UnitaryProjectionContext ctx;

public:
  UnitaryProjection() {
    GRID_ASSERT((Nc == 2 || Nc == 3) && "unitary projection only supported for Nc = 2 and Nc = 3 for now");
    ctx = UnitaryProjectionContext(CayleyHamiltonProjection, SylvesterDerivative);
  }

  UnitaryProjection(UnitaryProjectionContext ctx): ctx(ctx)
  { GRID_ASSERT((Nc == 2 || Nc == 3) && "unitary projection only supported for Nc = 2 and Nc = 3 for now"); }

public:
  template <class Field>
  void project(Field& y, const Field& x) {
    /**
     * @brief U(N) unitary projection via Cayley-Hamilton or SVD
     * @author Curtis Taylor Peterson
     * @details
     * This method implements a U(N) projection of a general complex NxN matrix, N = 2 or 3,
     * using Cayley-Hamilton or the one-sided Jacobi singular value decomposition in
     * MatrixMath. For details about the Cayley-Hamilton approach, see the references
     * provided above; namely the OG paper by Hasenfratz et al and later work by the
     * MILC collaboration. Please note that this method is modelled after the approach
     * taken in Quantum EXpressions by James Osborn and Xiao-Yong Jin, aside from the
     * singular value decomposition.
     *
     * The usual Cayley-Hamilton polynomial is calculated across SIMD lanes. If it
     * fails the polar identities or the eigenvalue checks, one-sided Jacobi
     * starts again from the original link in the same kernel. This also supplies
     * a unitary completion for singular links. Without the SVD backup, or with
     * SVD-only projection, failing links raise a word and the host aborts. Both
     * paths keep the links on the accelerator.
     */
    using namespace MatrixMath;
    tracePush("UnitaryProjection::project");
    
    const bool svdOnly = ctx.projectionMethod == SingularValueDecompositionProjection;
    const bool backup = ctx.backupSVD || svdOnly;
    uint64_t* failed = failWord();

    UnitaryProjectionReport* counters = UnitaryProjectionReport::counters();

    RealScalar relTol = ctx.relativeSVDTolerance;
    RealScalar minSig = sqrt(RealScalar(ctx.absoluteSVDTolerance));
    
    y.Checkerboard() = x.Checkerboard();
    
    ACCELERATOR_SCOPE(
      autoView(xv, x, AcceleratorRead);
      autoView(yv, y, AcceleratorWrite);

      accelerator_for(idx, x.Grid()->oSites()*Nlinks<Field>, Simd::Nsimd(), {
        int n = idx/Nlinks<Field>, mu = idx % Nlinks<Field>;

        iMatrix<AccComplex, Nc> a = ACCREAD(xv[n], mu);
        iMatrix<AccComplex, Nc> b;
        
        Realified<AccComplex> ok = svdOnly ? Realified<AccComplex>(0.0f) : MatrixMath::polar(b, a, relTol, minSig);
        if (backup) simd_for(l, Simd) {
          if (MatrixMath::maskLane(ok, l) == 0.0) {
            iMatrix<Scalar, Nc> al = MatrixMath::lane(a, l); 
            iMatrix<Scalar, Nc> u, v;
            iVector<RealScalar, Nc> s;

            bool fine = svd(u, v, s, al);
            
            MatrixMath::maskLane(ok, l) = fine; 
            MatrixMath::lane(b, l) = u*adj(v);

            counters->countSVD();
        } }

        raiseFailure(failed, ok);
        ACCWRITE(yv[n], mu, b);
      });
    )
    GRID_ASSERT(*failed == 0 && "unitary projection failed on some link");

    tracePop("UnitaryProjection::project");
  }

  template <class Field>
  void derivative(Field& dydx, const Field& dzdy, const Field& y, const Field& x) {
    /**
     * @brief Derivative of U(N) unitary projection
     * @author Curtis Taylor Peterson
     * @details
     * Cayley-Hamilton across SIMD lanes; lanes that fail its checks, or every lane
     * for the SVD-only derivative, use the squared singular values of u in the
     * same kernel when the SVD backup is on [Eqn. 43 of PRD(82)074501]. Lanes
     * that still fail raise the failure word and the host aborts.
     * 
     * Sylvester derivative
     * ---------------------
     * This is a very clever approach to calculating the derivative of the unitary
     * projection; it was invented by the authors of the original HISQ paper
     * [PRD75(2007)054502] and expanded upon by James Osborn and Xiao-Yong Jin in the
     * Quantum EXpressions code; see MatrixMath::polarDeriv.
     *
     * Write u = v H, with v unitary and H = v'u positive Hermitian. The
     * variation of v is tangent to U(N), so v' dv is anti-Hermitian. This gives
     * (1) H X + X H = v' G - G' v,
     * (2) dS/du = v X,
     * where G = dS/dv is the upstream gradient. This is the same Sylvester
     * problem as the QEX approach, arranged to avoid forming H^{-1} and then
     * cancelling its large contributions. The analytic N = 2 and N = 3 solution
     * is calculated across SIMD lanes in sylvester. Difficult links use the
     * Cayley-Hamilton derivative with SVD eigenvalues instead. All paths use
     * the input precision and keep the links on the accelerator.
     * When projected links are not supplied, derivative(dydx, dzdy, x) projects
     * them into a temporary field first.
     * 
     * Cayley-Hamilton derivative
     * --------------------------
     * This method implements the derivative of the U(N) projection using the
     * Cayley-Hamilton approach. For details about this approach, see the references
     * provided above; namely the OG paper by Hasenfratz et al and later work by the
     * MILC collaboration.
     *
     * This method follows Appendix C of PRD82(2010)074501. The eigenvalues of
     * q = u'u are calculated analytically and checked against det(q). If the
     * relative determinant error is too large or an eigenvalue is too small,
     * replace the eigenvalues by the squared singular values of u [Eqn. 43].
     * When the smallest eigenvalue is below the force cutoff, shift all three
     * eigenvalues and q by the cutoff [Eqn. 44]. The corrected eigenvalues then
     * determine the Cayley-Hamilton square root. Its inverse and the analytic
     * Sylvester solution give the derivative, as in the original implementation;
     * See MatrixMath::polarDerivCayleyHamilton.
     */
    using namespace MatrixMath;
    tracePush("UnitaryProjection::derivative");

    const bool cayleyHamilton = ctx.derivativeMethod == CayleyHamiltonDerivative;
    const bool svdOnly = cayleyHamilton && ctx.svdOnlyDerivative;
    const bool backup = ctx.backupSVD || svdOnly;
    uint64_t* failed = failWord();

    UnitaryProjectionReport* counters = UnitaryProjectionReport::counters();

    RealScalar relTol = ctx.relativeSVDTolerance;
    RealScalar absTol = ctx.absoluteSVDTolerance;
    RealScalar cutoff = cayleyHamilton ? ctx.derivativeEigenvalueCutoff : 0.0f;

    dydx.Checkerboard() = x.Checkerboard();
    ACCELERATOR_SCOPE(
      autoView(xv, x, AcceleratorRead);
      autoView(yv, y, AcceleratorRead);
      autoView(dzdyv, dzdy, AcceleratorRead);
      autoView(dydxv, dydx, AcceleratorWrite);

      accelerator_for(idx, x.Grid()->oSites()*Nlinks<Field>, Simd::Nsimd(), {
        int n = idx/Nlinks<Field>, mu = idx % Nlinks<Field>;

        iMatrix<AccComplex, Nc> link = ACCREAD(xv[n], mu), chain = ACCREAD(dzdyv[n], mu);
        iMatrix<AccComplex, Nc> proj, deriv;
        Realified<AccComplex> ok = 0.0f; 
        ForceFilter<AccComplex> filter{cutoff};

        if (svdOnly) { } // every lane rejected, all go to the fallback
        else if (cayleyHamilton) {
          iVector<Realified<AccComplex>, Nc> eigs = hermEigs(gram(link));
          ok = polarDerivCayleyHamilton(deriv, chain, link, eigs, filter);
          if (backup) { ok *= checkEigs(eigs, link, relTol, absTol, cutoff); }
          counters->countFilter(filter.applied*ok);
        } else { proj = ACCREAD(yv[n], mu); ok = polarDeriv(deriv, chain, proj, link); }

        if (backup) {
          simd_for(l, Simd) {
            if (MatrixMath::maskLane(ok, l) == 0.0) { 
              iMatrix<Scalar, Nc> linkSite = MatrixMath::lane(link, l);
              iMatrix<Scalar, Nc> chainSite = MatrixMath::lane(chain, l);
              iMatrix<Scalar, Nc> u, v, derivSite;
              iVector<RealScalar, Nc> eigs;
              
              ForceFilter<Scalar> filterSite{cutoff};

              bool okSite = svd(u, v, eigs, linkSite);
              for (int i = 0; i < Nc; ++i) { eigs(i) *= eigs(i); }
              okSite = okSite && polarDerivCayleyHamilton(derivSite, chainSite, linkSite, eigs, filterSite) > 0.0;

              MatrixMath::maskLane(ok, l) = okSite;
              MatrixMath::lane(deriv, l) = derivSite;

              counters->countSVD();
              counters->countFilter(filterSite.applied);
        } } }
        
        raiseFailure(failed, ok);
        ACCWRITE(dydxv[n], mu, deriv); 
      });
    )
    GRID_ASSERT(*failed == 0 && "derivative of unitary projection failed on some link");
  
    tracePop("UnitaryProjection::derivative");
  }

  template <class Field>
  void derivative(Field& dydx, const Field& dzdy, const Field& x) {
    if (ctx.derivativeMethod == CayleyHamiltonDerivative) derivative(dydx, dzdy, x, x); // MILC never reads the projection
    else { Field y(x.Grid()); project(y, x); derivative(dydx, dzdy, y, x); }
  }
};

NAMESPACE_END(Grid);

#endif // QCD_UTILS_UNITARYPROJECTION_H
