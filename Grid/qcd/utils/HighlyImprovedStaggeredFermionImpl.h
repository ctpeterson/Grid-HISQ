/*************************************************************************************
Grid physics library, www.github.com/paboyle/Grid
Source file: ./lib/qcd/utils/HighlyImprovedStaggeredFermionImpl.h
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

/**
 * @file HighlyImprovedStaggeredFermionImpl.h
 * @brief Interface for implementation of highly improved staggered fermions (HISQ)
 * @author Curtis Taylor Peterson
 * @details
 * This header file is meant to act as an interface for both Grid and the 
 * MILC codebase to utilize the "highly improved staggered quark" action
 * within Grid. The interface itself is composed of a single core component class
 *
 * Massive credit to Shaun Lahert for finding the most critical optimization that
 * directly affected the run time, that being the slight refactoring to 
 * GeneralLocalStencil that bulk transfers the lookup table to the GPU. 
 * 
 * * HighlyImprovedStaggeredFermionImpl: Class providing support for highly
 *   improved staggered fermions in Grid. EXtoses methods for fat7/asqtad 
 *   smearing and its derivative with optional inclusion of Lepage term needed 
 *   for second level of HISQ smearing. 
 * 
 * References:
 * * Quantum EXtressions (QEX): https://github.com/jcosborn/qex
 * * QOPQDP [SciDAC]: https://github.com/usqcd-software/qopqdp
 * * SIMULATEeQCD: https://github.com/LatticeQCD/SIMULATeQCD
 * * Follana, E. et al.: https://doi.org/10.1103/PhysRevD.75.054502
 * * MILC Collaboration (2010): https://doi.org/10.1103/PhysRevD.82.074501
 * 
 * Acknowledgements:
 *   Curtis Taylor Peterson would like to thank James Osborn for developing/testing
 *   the implementations of HISQ in QEX and QOPQDP, from which the "fast" option for 
 *   the fat7/asqtad smearing and derivativative are based and have been tested against.
 *   
 *   This material is based upon work supported by the U.S. Department of Energy, 
 *   Office of Science, Office of Advanced Scientific Computing Research, Scientific 
 *   Discovery through Advanced Computing (SciDAC) program.
*/

#pragma once 

#ifndef QCD_UTILS_HISQ_IMPL_H
#define QCD_UTILS_HISQ_IMPL_H 

#include <Grid/qcd/utils/Transporter.h>
#include <Grid/qcd/utils/UnitaryProjection.h>
#include <Grid/qcd/action/ActionParams.h>

#include <Grid/perfmon/Tracing.h>

NAMESPACE_BEGIN(Grid);

//
// macros
//

// loop excluding no indices
#define HISQLOOP0(exec) for (int mu = 0; mu < Nd; ++mu) { exec; } \

// loop excluding one index
#define HISQLOOP1(exec) for (int nu = 0; nu < Nd; ++nu)   \
  { if (nu == mu) {continue;} else exec; }                \

// loop excluding two indices
#define HISQLOOP2(exec) for (int i = 0; i < Nd; ++i)     \
  { if ((i == mu) or (i == nu)) {continue;} else exec; } \

// loop excluding three indices
#define HISQLOOP3(exec) for (int j = 0; j < Nd; ++j)                 \
  { if ((j == mu) or (j == nu) or (j == i)) {continue;} else exec; } \

// encapsulates conditional execution of Lepage calculations for cleanliness
#define HISQLEPAGE(exec) if (ctx.lepage != 0.0) { exec; } \

// encapsulates conditional execution of Naik calculations for cleanliness
#define HISQNAIK(exec) if (ctx.naik != 0.0) { HISQLOOP0(exec) } \

// better notation for ternary eXtression
#define when(cond, valTrue, valFalse) ((cond) ? (valTrue) : (valFalse)) \

//
// useful consts
//

// halo depth
const int DEPTH = 1;

// usual 4D HISQ coefficients: defined for convenience
const double // Lepage & Naik
  LEPAGE = -1.0/8.0,
  NAIK = -1.0/24.0;
const double // fat-7
  F7L1 = -LEPAGE,
  F7L3 = -0.5*F7L1,
  F7L5 = -0.25*F7L3,
  F7L7 = 0.0625*NAIK;
const double // asqtad
  ASQL1 = -8.0*LEPAGE,
  ASQL3 = F7L3,
  ASQL5 = F7L5,
  ASQL7 = F7L7; 
const double // Naik epsilon heavy quark mass coefficients
  NAIKEPS1 = -27.0/40.0,
  NAIKEPS2 = 327.0/1120.0,
  NAIKEPS3 = -15607.0/268800.0,
  NAIKEPS4 = -73697.0/3942400.0;
const bool BACKUPSVD = true; // use backup SVD in unitary projection when applicable
const double // unitary projection parameters
  REUNITCUTOFF = 1e-20,         // base-level cutoff on eigenvalues
  REUNITDERIVCUTOFF = 5e-5,     // base-level cutoff on eigenvalues for derivative
  RELBACKUPSVDTOLERANCE = 1e-8, // relative tolerance for triggering backup SVD
  ABSBACKUPSVDTOLERANCE = 1e-8; // absolute tolerance for triggering backup SVD

//
// useful data structures
//

struct HISFContext {
  /**
   * @brief Context for highly improved staggered fermion smearing and projection
   * @author Curtis Taylor Peterson
   * @details
   * Context structure for passing parameters to HISF smearing and projection methods.
   */
  // fat7 smearing parameters
  RealD c0 = 0.0, c1 = 0.0, c2 = 0.0, c3 = 0.0;

  // asqtad parameters
  RealD lepage = 0.0;
  RealD naik = 0.0;

  // unitary projection parameters
  bool backupSVD = false;
  bool svdOnly = false;
  RealD relSVDTolerance = 0.0;
  RealD absSVDTolerance = 0.0;
  RealD eigenCutoff = 0.0;

  HISFContext() { };

  HISFContext(
    RealD c0, 
    RealD c1, 
    RealD c2, 
    RealD c3,
    RealD lepage, 
    RealD naik,
    bool svdOnly,
    bool backupSVD,
    RealD relSVDTolerance,
    RealD absSVDTolerance,
    RealD eigenCutoff
  ):c0(c0), 
    c1(c1), 
    c2(c2), 
    c3(c3),
    lepage(lepage), 
    naik(naik),
    svdOnly(svdOnly),
    backupSVD(backupSVD),
    relSVDTolerance(relSVDTolerance),
    absSVDTolerance(absSVDTolerance),
    eigenCutoff(eigenCutoff) { };
  
  HISFContext(RealD c0, RealD c1, RealD c2, RealD c3, RealD lepage, RealD naik):
    c0(c0), c1(c1), c2(c2), c3(c3), lepage(lepage), naik(naik) { };
  
  HISFContext(RealD c0, RealD c1, RealD c2, RealD c3): 
    c0(c0), c1(c1), c2(c2), c3(c3), lepage(0.0), naik(0.0) { };
  
  HISFContext(
    bool svdOnly,
    bool backupSVD, 
    RealD relSVDTolerance, 
    RealD absSVDTolerance, 
    RealD eigenCutoff
  ):svdOnly(svdOnly),
    backupSVD(backupSVD), 
    relSVDTolerance(relSVDTolerance), 
    absSVDTolerance(absSVDTolerance), 
    eigenCutoff(eigenCutoff) { };

  // Exact configuration equality, including legacy projection parameters.
  bool operator==(const HISFContext& other) const {
    return c0 == other.c0 && c1 == other.c1 &&
           c2 == other.c2 && c3 == other.c3 &&
           lepage == other.lepage && naik == other.naik &&
           backupSVD == other.backupSVD && svdOnly == other.svdOnly &&
           relSVDTolerance == other.relSVDTolerance &&
           absSVDTolerance == other.absSVDTolerance &&
           eigenCutoff == other.eigenCutoff;
  }

  bool operator!=(const HISFContext& other) const { return !(*this == other); }
};

struct MILCContext {
  HISFContext fat7;
  HISFContext asqtad;
  std::vector<Real> naikFactors;
  std::vector<Real> naikEpsilons;
  std::vector<int> naikOrders;

  MILCContext(
    HISFContext fat7,
    HISFContext asqtad,
    std::vector<Real> naikFactors,
    std::vector<Real> naikEpsilons,
    std::vector<int> naikOrders
  ):
    fat7(fat7),
    asqtad(asqtad),
    naikFactors(naikFactors),
    naikEpsilons(naikEpsilons),
    naikOrders(naikOrders) { };
  
  int numNaiks() const { return naikEpsilons.size(); }

  int order(int species) const { return naikOrders[species]; }

  Real epsilon(int species) const { return naikEpsilons[species]; }

  Real factor(int naik) const { return naikFactors[naik]; }
};

//
// highly improved staggered fermion implementation
//

template <class Gimpl>
class HighlyImprovedStaggeredFermionImpl: Gimpl {
/**
 * @brief Highly improved staggered fermion implementation in Grid
 * @author Curtis Taylor Peterson
 * @details
 * Highly improved staggered fermion (HISF/HISQ) implementation in Grid. EXtoses 
 * methods for FNAL/MILC and non-FNAL/MILC implementations of Highly Improved Staggered
 * Fermions. This class boasts an implementation of HISQ that is low in communication 
 * overhead whilst retaining a low memory footprint by Grid's PaddedCell and 
 * GeneralLocalStencil through the PeriodicBC::Transporter class.
 */
  
public: INHERIT_GIMPL_TYPES(Gimpl);

private:
  typedef typename Simd::scalar_type GridScalar;
  typedef typename std::vector<GaugeLinkField> GaugeLorentzField;
  typedef typename std::vector<ComplexField> StaggeredPhases;
  using Transporter = PeriodicBC::Transporter<Gimpl>;

public:
  GridCartesian* ugrid;

  StaggeredImplParams params;
  StaggeredPhases stagPhases;

private:
  int _depth = -1;

  int _cached = false;
  PeriodicBC::TransporterStencil _stencil;

private:
  void init(GridCartesian* grid, bool calculateStaggeredPhases) {
    assert(Nc == 3 && "HISQ only suppored for SU(3)");
    assert(Nd == 4 && "HISQ only supported for 4 dimensions");
    this->ugrid = grid;
    if (calculateStaggeredPhases) calcStagPhases(stagPhases);
  }

public:
  HighlyImprovedStaggeredFermionImpl(
    GridCartesian* grid, 
    const StaggeredImplParams params,
    bool calculateStaggeredPhases = true
  ): stagPhases(Nd, grid), params(params)
  { init(grid, calculateStaggeredPhases); }

  HighlyImprovedStaggeredFermionImpl(
    GridCartesian* grid,
    bool calculateStaggeredPhases = true
  ): stagPhases(Nd, grid), params(StaggeredImplParams({1, 1, 1, -1}))
  { init(grid, calculateStaggeredPhases); }

  // prevent duplicate ownership of PaddedCell grids
  HighlyImprovedStaggeredFermionImpl(const HighlyImprovedStaggeredFermionImpl&) = delete;
  HighlyImprovedStaggeredFermionImpl& operator=(const HighlyImprovedStaggeredFermionImpl&) = delete;

private:
  void calcStagPhases(StaggeredPhases& phases) {
    /**
     * @brief HISQ gauge configuration constructor
     * @author Curtis Taylor Peterson, Peter Boyle
     * @details
     * Staggered phi follow the "MILC convention", which treats the fourth 
     * direction as the "time" coordinate:
     * (2a) eta_0 = (-1)^{x3}       <---| 
     * (2b) eta_1 = (-1)^{x3+x0}        | convention in
     * (2c) eta_2 = (-1)^{x3+x0+x1}     | this code
     * (2d) eta_3 = 1               <---|
     * Though awkard, this convention follows that of most texts on
     * relativity. This is opposed to the convention that one often finds in 
     * lattice gauge theory textbooks, where the "time" direction is x0:
     * (3a) eta_0 = 1
     * (3b) eta_1 = (-1)^{x0}
     * (3c) eta_2 = (-1)^{x0+x1}
     * (3d) eta_3 = (-1)^{x0+x1+x2}
     * Boundary conditions are imposed by "rephasing" the links 
     * on the boundary, with "1" for periodic and "-1" for anti-periodic.
     */
    GridBase *grid = phases[0].Grid();
    Lattice<iScalar<vInteger>> x(grid), y(grid), t(grid);
    Lattice<iScalar<vInteger>> tx(grid), txy(grid), xyzt(grid), coor(grid);
    ComplexField phi(grid);

    LatticeCoordinate(x, 0);
    LatticeCoordinate(y, 1);
    LatticeCoordinate(t, 3);
    tx = t + x;
    txy = tx + y;

    for (int mu = 0; mu < Nd; mu++) { 
      // for boundary phi
      int N = grid->GlobalDimensions()[mu] - 1;
      auto bpha = params.boundary_phases[mu];
      GridScalar dirichlet(real(bpha), imag(bpha));

      // staggered phases x boundary phases
      LatticeCoordinate(coor, mu);
      phi = 1.0;
      if (mu == 0) { phi = where(mod(t, 2) == (Integer)0,   phi, -phi); }
      if (mu == 1) { phi = where(mod(tx, 2) == (Integer)0,  phi, -phi); }
      if (mu == 2) { phi = where(mod(txy, 2) == (Integer)0, phi, -phi); }
      phases[mu] = where(coor == (Integer)N, dirichlet*phi, phi);
    }
  }

private:
  // wrap PokeIndex (insert gauge link field into gauge field layout)
  inline void intoGauge(GaugeField& Uout, const GaugeLinkField& U, int mu)
  { PokeIndex<LorentzIndex>(Uout, U, mu); }

  // wrap PeekIndex (extract gauge link field form gauge field layout)
  inline const GaugeLinkField toLink(const GaugeField& U, int mu)
  { return PeekIndex<LorentzIndex>(U, mu); }

  // break up memory layout of gauge field into std::vector of gauge link fields
  inline const GaugeLorentzField toLorentz(const GaugeField& U) 
  { GaugeLorentzField u(Nd, U.Grid()); HISQLOOP0(u[mu] = toLink(U, mu)) return u; }

  // aggregate std::vector of gauge link fields into layout of gauge field
  inline const GaugeField toGauge(GaugeLorentzField& u) 
  { GaugeField U(u[0].Grid()); HISQLOOP0(intoGauge(U, u[mu], mu)) return U; }

public:
  /** @brief rephases gauge links with staggered and Dirichlet boundary phases */
  void rephase(GaugeField& X, const GaugeField& W) 
  { for (int mu = 0; mu < Nd; ++mu) intoGauge(X, stagPhases[mu]*toLink(W, mu), mu); }

//
// HISQ smearing & unitary projection
//

public:
  void smear(
    GaugeField& X,
    GaugeField& WWW,
    const GaugeField& W,
    const HISFContext ctx
  ) {
    /**
     * @brief "Effective 3-link" fat7/asqtad + Lepage (optional) smear
     * @author Curtis Taylor Peterson
     * @details
     * Fat7/Asqtad links can be constructed recursively from 3-link smears. This 
     * insight is attributed to James Osborn (Argonne National Laboratory),
     * who used it to implement the Fat7/Asqtad smearing in QOPQDP and Qauntum
     * EXtressions (QEX). This method follows a similar pattern to James' 
     * implementation and combines it with Grid's padded cell. Note that many other 
     * codes, such as MILC, use a "path-based" approach by default, which tend to
     * less efficiently reuse computations from lower levels in the smearing.
     * 
     * Define
     * (1) Sν(U;n) := Uν(n) U(n+ν) Uν(n+μ)† + Uν(n-ν)† U(n-ν) U(n-ν+μ).
     * for some generic SU(N) field U(n). I call the first contribution the "upper 
     * staple" and the second the "lower staple". From (1), recursively define
     * (2) Sν(n) := Sν(Uμ;n),
     * (3) Sνj(n) := Sj(Sν;n),
     * and
     * (4) Sνij(n) := Si(Sνj;n).
     * Then an fat7 link Vμ(n) in four dimensions can be eXtressed as a composition 
     * of sums
     * (5) Vμ(n) = c0 Uμ(n)            [5a]
     *     + ∑_{ν≠μ}(c1 Sν(n)          [5b]
     *     + ∑_{i≠μ,ν}[c2 Sνi(n)       [5c]
     *     + ∑_{j≠μ,ν,i} c3 Sνij(n)])  [5d]
     * And that's it. The key thing to note is that we can reuse the staples
     * at lower levels to construct staples at higher levels.
     * 
     * The optional Lepage term adds an additional contribution to (5b):
     * (6) + lepage * Sν(Sν(n)),
     * where Sν(n) is treated as the "link" field in the Lepage staple. We can reuse
     * the code that calculates the symmetric staple, so long as we correct for the
     * additional 1-link terms that doing so adds to the smearing by compensating
     * in the 1-link coefficient. 
     * 
     * Reusing staples from lower levels comes at the cost of having to perform more
     * halo exchanges. As such, this method tends to perform poorly on the GPU 
     * architectures of the mid-2020s. This affects the derivative as well. We've 
     * a separate GPU path that does not reuse staples, hence reducing the 
     * communication overhead dramatically.
     * 
     * References:
     * * Quantum EXtressions (QEX): https://github.com/jcosborn/qex
     * * QOPQDP [SciDAC]: https://github.com/usqcd-software/qopqdp
     * * Follana, E. et al.: https://doi.org/10.1103/PhysRevD.75.054502
     */
    tracePush("HighlyImprovedStaggeredFermionImpl::smear");

    PaddedCell cell(DEPTH, ugrid);
    Transporter w = _cached ? Transporter(cell, W, _stencil) : Transporter(cell, W);
    GridBase* pgrid = (GridBase*)cell.grids.back();
    GaugeLorentzField x(Nd, pgrid);
    GaugeLinkField s3(pgrid), s5(pgrid);
    
    // 1-, 3-, 5-, and 7-link terms + lepage
    HISQLOOP0( // Eqn 5a
      x[mu] = (ctx.c0 - 6.0*ctx.lepage)*w.link(mu);
      HISQLOOP1( // Eqn 5b
        s3 = w.exchange(w.staple(mu, nu), mu, nu);
        x[mu] += ctx.c1*s3; 
        HISQLEPAGE(x[mu] += ctx.lepage*w.staple(s3, mu, nu)) // Eqn 6
        HISQLOOP2( // Eqn 5c
          s5 = w.exchange(w.staple(s3, mu, i), mu, i);
          x[mu] += ctx.c2*s5;
          HISQLOOP3(x[mu] += ctx.c3*w.staple(s5, mu, j)) // Eqn 5d
    ) ) )
    X = w.toTightGrid(toGauge(x));

    // Naik term
    HISQNAIK(x[mu] = ctx.naik*w.CovShiftFwd(mu, w.exchange(w.CovShiftFwd(mu), mu)))
    if (ctx.naik != 0.0) WWW = w.toTightGrid(toGauge(x));

    // Cache stencil
    if (!_cached) { _stencil = w.stencil(); _cached = true; }

    tracePop("HighlyImprovedStaggeredFermionImpl::smear");
  }

  void smear(GaugeField& X, GaugeField& WWW, const GaugeField& W) {
    HISFContext asqtadCtx(ASQL1, ASQL3, ASQL5, ASQL7, LEPAGE, NAIK); 
    smear(X, WWW, W, asqtadCtx); 
  }

  void smear(GaugeField& X, const GaugeField& W, const HISFContext ctx) 
  { smear(X, X, W, ctx); }

  void smear(GaugeField& X, const GaugeField& W) { 
    HISFContext fat7Ctx(F7L1, F7L3, F7L5, F7L7);  
    smear(X, X, W, fat7Ctx); 
  }

  void project(GaugeField& V, const GaugeField& U, const HISFContext ctx) { 
    UnitaryProjectionContext projCtx(
      ctx.svdOnly ? SingularValueDecompositionProjection : CayleyHamiltonProjection
    );
    if (!ctx.svdOnly) projCtx.setBackupSVD(ctx.backupSVD);
    projCtx.setRelativeSVDTolerance(ctx.relSVDTolerance);
    projCtx.setAbsoluteSVDTolerance(ctx.absSVDTolerance);
    UnitaryProjection<Gimpl> projection(projCtx);
    projection.project(V, U);
  }

  void project(GaugeField& V, const GaugeField& U) { 
    UnitaryProjectionContext projCtx(CayleyHamiltonProjection);
    projCtx.setBackupSVD(BACKUPSVD);
    projCtx.setRelativeSVDTolerance(RELBACKUPSVDTOLERANCE);
    projCtx.setAbsoluteSVDTolerance(ABSBACKUPSVDTOLERANCE);
    UnitaryProjection<Gimpl> projection(projCtx);
    projection.project(V, U);
  }

  void project(GaugeField& V, const GaugeField& U, const UnitaryProjectionContext ctx) {
    UnitaryProjection<Gimpl> projection(ctx);
    projection.project(V, U);
  }

//
// HISQ smearing & unitary projection derivatives
//

public:
  void smearDerivative(
    GaugeField& dXdU,
    const GaugeField& dSdX,
    const GaugeField& dSdWWW,
    const GaugeField& W,
    const HISFContext ctx
  ) {
    /**
     * @brief "Effective 3-link" fat7/asqtad + Lepage (optional) derivative
     * @author Curtis Taylor Peterson
     * @details
     * Calculates the derivative of fat7/asqtad + Lepage links, including chain
     * rule. See the comments under `smear` method for details on the smearing.
     * This method follows the "effective 3-link" force in QOPQDP and Quantum
     * EXpressions (QEX), written by James Osborn (Argonne National Laboratory),
     * and combines it with Grid's padded cell.
     *
     * Taking Vμ(n) to be a smeared link, write the input chain rule as Cμ(n), with
     * (1) dS = Re ∑_{μ,n} tr[ Cμ(n)† dVμ(n) ].
     * The output Fμ(n) uses the same convention with V replaced by U. Multiplying
     * by the link and projecting onto the Lie algebra is left to the caller.
     *
     * Consider a single symmetric staple Sν(u), as defined in `smear`. Its upper
     * contribution is
     * ν      ---🠢
     * 🠡      🠡   🠣   (Fig 1)
     * -🠢 μ   ....
     * where the dotted line closes the staple with the chain rule. Differentiating
     * the middle link passes the chain back to u; differentiating the side links
     * adds to the force on Uν. Including the lower staple gives
     * (2) dS = Re ∑_n tr[ Sν(C)† du + Dν(u,C)† dUν ].
     * Here Dν is the four-term side derivative. In Transporter, the two outputs
     * are `staple(C, mu, nu)` and `stapleDerivative(u, C, nu, mu)`. Notice that the
     * directions in the second call are reversed. We calculate both together
     * so that their common links and matrix products can be reused.
     *
     * We now fix the direction μ of the smeared link, and follow its chain rule
     * backwards through the smearing. The key thing to note is that several
     * paths use the same 3-link staple. We can add their chain rules together
     * before differentiating that staple, just as we reuse it in the smearing.
     * Suppressing the site index n, first store
     * (3) s3[ν] = Sν(Uμ),
     * and set d5[ν] to zero. For each ν≠μ, construct
     * (4) s5 = c3 ∑_{i,j≠μ,ν; i≠j} Sj(s3[i]) + lepage*s3[ν].
     * This is the middle link for the outermost staple of the 7-link and Lepage
     * terms. Applying (2) adds Dν(s5,Cμ) to Fν and gives d3 = Sν(Cμ).
     * The Lepage chain contributes lepage*d3 to d5[ν]. The remaining 5- and
     * 7-link chains can then be combined:
     * (5) d3 = c3*d3 + c2*Cμ.
     * For each remaining pair i,j, applying (2) to Si(s3[j]) gives
     * (6) Fi += Di(s3[j],d3),    d5[j] += Si(d3).
     * Thus d5[ν] collects all contributions to the chain on s3[ν]. Once the
     * sum over the outermost staples is complete, add the direct 3-link chain:
     * (7) c = c1*Cμ + d5[ν].
     * One last application of (2) gives
     * (8) Fν += Dν(Uμ,c),        Fμ += Sν(c).
     * And that's it. The 1-link contribution initializes F before these loops,
     * with coefficient c0 - 6*lepage to compensate for the extra 1-link paths
     * in Sν(Sν(Uμ)). The Lepage term shares the same intermediates throughout.
     *
     * The Naik derivative is calculated separately. Writing U, U+, U-, ... for
     * links displaced along μ, and C for the chain on the long link, its three
     * contributions are
     * (9) naik * [ C U++† U+† + U-† C- U+† + U-† U--† C-- ].
     * Shifting U†C once gives `back`; it is shared by the second and third
     * contributions. The first uses the shifted product U+†U†. This needs only
     * nearest-neighbor exchanges.
     *
     * References:
     * * Quantum EXpressions (QEX): https://github.com/jcosborn/qex
     * * QOPQDP [SciDAC]: https://github.com/usqcd-software/qopqdp
     * * Follana, E. et al.: https://doi.org/10.1103/PhysRevD.75.054502
     */
    tracePush("HighlyImprovedStaggeredFermionImpl::smearDerivative");

    PaddedCell cell(DEPTH, ugrid);
    Transporter w = _cached ? Transporter(cell, W, _stencil) : Transporter(cell, W);
    GridBase* pgrid = (GridBase*)cell.grids.back();
    GaugeLorentzField dxdw = toLorentz(w.toPaddedGrid(dSdX));
    GaugeLorentzField dxdu(Nd, pgrid), s3(Nd, pgrid), d5(Nd, pgrid);
    GaugeLinkField s5(pgrid), d3(pgrid), c(pgrid);

    // fat7 + lepage (lepage won't execute if lepage == 0.0)
    HISQLOOP0(dxdu[mu] = (ctx.c0 - 6.0*ctx.lepage)*dxdw[mu])
    HISQLOOP0(
      HISQLOOP1(
        s3[nu] = w.exchange(w.staple(mu, nu), mu, nu); // Eqn 3
        d5[nu] = Zero();
      )
      HISQLOOP1(
        s5 = Zero(); // Eqn 4
        HISQLOOP2(HISQLOOP3(w.addStaple(s5, s3[i], mu, j, ctx.c3)))
        HISQLEPAGE(s5 += ctx.lepage*s3[nu])
        w.refresh(s5, mu, nu);
        d3 = Zero();
        w.addStapleDerivative(d3, dxdu[nu], s5, dxdw[mu], mu, nu);
        HISQLEPAGE(d5[nu] += ctx.lepage*d3)
        d3 = ctx.c3*d3 + ctx.c2*dxdw[mu]; // Eqn 5
        HISQLOOP2(
          w.refresh(d3, mu, i);
          HISQLOOP3(w.addStapleDerivative(d5[j], dxdu[i], s3[j], d3, mu, i)) // Eqn 6
        )
      )
      HISQLOOP1(
        c = ctx.c1*dxdw[mu] + d5[nu]; // Eqn 7
        w.refresh(c, mu, nu);
        w.addStapleDerivative(dxdu[mu], dxdu[nu], w.link(mu), c, mu, nu); // Eqn 8
      )
    )

    // Naik ("long link") -- Eqn 9
    if (ctx.naik != 0.0) dxdw = toLorentz(w.toPaddedGrid(dSdWWW));
    HISQNAIK(
      // reuse the temporary fields from the staples
      auto& unext = s5;
      auto& chain = d3;
      auto& back = c;

      unext = w.CovShiftIdentFwd(mu, w.link(mu));
      chain = adj(w.link(mu))*dxdw[mu];
      w.refresh(chain, mu);
      
      back = w.CovShiftIdentBck(mu, chain);
      chain = adj(unext)*adj(w.link(mu));
      w.refresh(chain, mu);
      
      dxdu[mu] += ctx.naik*(dxdw[mu]*w.CovShiftIdentFwd(mu, chain) + back*adj(unext));
      chain = adj(w.link(mu))*back;
      w.refresh(chain, mu);
      
      dxdu[mu] += ctx.naik*w.CovShiftIdentBck(mu, chain);
    )

    // extract from padded grid
    dXdU = w.toTightGrid(toGauge(dxdu));

    // Cache stencil
    if (!_cached) { _stencil = w.stencil(); _cached = true; }

    tracePop("HighlyImprovedStaggeredFermionImpl::smearDerivative");
  }

  void smearDerivative(
    GaugeField& dXdU,
    const GaugeField& dSdX,
    const GaugeField& dSdWWW,
    const GaugeField& W
  ) { 
    HISFContext asqtadCtx(ASQL1, ASQL3, ASQL5, ASQL7, LEPAGE, NAIK); 
    smearDerivative(dXdU, dSdX, dSdWWW, W, asqtadCtx); 
  }

  void smearDerivative(
    GaugeField& dXdU,
    const GaugeField& dSdX,
    const GaugeField& W,
    const HISFContext ctx
  ) { smearDerivative(dXdU, dSdX, dSdX, W, ctx); }

  void smearDerivative(GaugeField& dXdU, const GaugeField& dSdX, const GaugeField& W) { 
    HISFContext fat7Ctx(F7L1, F7L3, F7L5, F7L7);  
    smearDerivative(dXdU, dSdX, W, fat7Ctx);
  }

  void projectionDerivative(
    GaugeField& dVdU, 
    const GaugeField& dZdV, 
    const GaugeField& U,
    const HISFContext ctx
  ) {
    UnitaryProjectionContext projCtx(MIMDCollaborationDerivative);
    projCtx.setDerivativeEigenvalueCutoff(ctx.eigenCutoff);
    projCtx.setSVDOnlyDerivative(ctx.svdOnly);
    projCtx.setBackupSVD(ctx.backupSVD);
    projCtx.setRelativeSVDTolerance(ctx.relSVDTolerance);
    projCtx.setAbsoluteSVDTolerance(ctx.absSVDTolerance);
    UnitaryProjection<Gimpl> projection(projCtx);
    projection.derivative(dVdU, dZdV, U); 
  }

  void projectionDerivative(
    GaugeField& dVdU, 
    const GaugeField& dZdV, 
    const GaugeField& U
  ) { 
    UnitaryProjectionContext projCtx(MIMDCollaborationDerivative);
    projCtx.setDerivativeEigenvalueCutoff(REUNITDERIVCUTOFF);
    projCtx.setBackupSVD(BACKUPSVD);
    projCtx.setRelativeSVDTolerance(RELBACKUPSVDTOLERANCE);
    projCtx.setAbsoluteSVDTolerance(ABSBACKUPSVDTOLERANCE);
    UnitaryProjection<Gimpl> projection(projCtx);
    projection.derivative(dVdU, dZdV, U); 
  }

  void projectionDerivative(
    GaugeField& dVdU, 
    const GaugeField& dZdV, 
    const GaugeField& V,
    const GaugeField& U
  ) { 
    UnitaryProjectionContext projCtx(JinOsbornDerivative);
    UnitaryProjection<Gimpl> projection(projCtx);
    projection.derivative(dVdU, dZdV, V, U); 
  }

//
// MILC interface for HISQ smearing
//

public:
  void milcSmearDerivative(
    GaugeField& UdSdU,
    const std::vector<GaugeField>& dSdX,
    const std::vector<GaugeField>& dSdWWW,
    const GaugeField& W,
    const GaugeField& V,
    const GaugeField& U,
    const MILCContext ctx
  ) {
    GridBase* grid = U.Grid();
    GaugeField tdSdW(grid), dSdW(grid), dSdV(grid), dSdU(grid);

    dSdW = Zero();
    for (int species = 0; species < ctx.numNaiks(); ++species) {
      RealD eps = ctx.epsilon(species);
      HISFContext asqCtx(
        ctx.asqtad.c0 + eps/8.0, 
        ctx.asqtad.c1, 
        ctx.asqtad.c2, 
        ctx.asqtad.c3, 
        ctx.asqtad.lepage, 
        ctx.asqtad.naik*(1.0 + eps)
      );
      smearDerivative(tdSdW, adj(dSdX[species]), adj(dSdWWW[species]), W, asqCtx);
      dSdW += tdSdW;
    }

    projectionDerivative(dSdV, dSdW, V, ctx.fat7);
    smearDerivative(dSdU, dSdV, U, ctx.fat7);
    
    for (int mu = 0; mu < Nd; ++mu) {
      auto u = PeekIndex<LorentzIndex>(U, mu);
      auto dsdu = PeekIndex<LorentzIndex>(dSdU, mu);
      PokeIndex<LorentzIndex>(UdSdU, -dsdu*adj(u), mu);
    }
  }

  template <typename FermionField>
  void milcSmearDerivative(
    GaugeField& UdSdU,
    const GaugeField& W,
    const GaugeField& V,
    const GaugeField& U,
    const std::vector<FermionField>& vecx, 
    const MILCContext ctx
  ) {
    /**
     * @brief MILC interface for full HISQ smearing derivative
     * @author David Clarke and Curtis Taylor Peterson
     * @details
     * Full HISQ smearing derivative as implemented in the MILC codebase. This method
     * is meant to provide an interface for MILC users to utilize the HISQ
     * implementation in Grid.
     * 
     * We are calculating the force using the rational approximation. The goal is 
     * that we can approximate 
     * (1) (Mdag M)^(-4/q) = alpha_0 + sum_l alpha_l/(M^dag M + beta_l). 
     * Hence the index l runs over the introduce a different "Naik epsilon" for each M. 
     * Hence, we can think of the total applicationof this operator as having an index 
     * species, running over the different Naik epsilons; for each species there is a 
     * possibly different order_species, then the operator has an index l running up to 
     * order_species. All terms with species=0 correspond to epsilon_Naik = 0.
     * 
     * References:
     * * MILC Collaboration (2010): https://doi.org/10.1103/PhysRevD.82.074501
     */
    int l = 0;
    
    GridBase* grid = vecx[0].Grid();
    GridCartesian* cgrid = static_cast<GridCartesian*>(grid);
    GridRedBlackCartesian* rbgrid = SpaceTimeGrid::makeFourDimRedBlackGrid(cgrid);

    std::vector<GaugeField> dSdX(ctx.numNaiks(), grid); 
    std::vector<GaugeField> dSdWWW(ctx.numNaiks(), grid);
    
    GaugeLinkField t(grid);

    // process MILC inputs for solution vectors with different Naik epsilons
    for (int species = 0; species < ctx.numNaiks(); ++species) {
      dSdX[species] = Zero();
      dSdWWW[species] = Zero();
      
      // outer product
      for (int i = 0; i < ctx.order(species); ++i) {
        GaugeField tdSdX(grid), tdSdWWW(grid);
        FermionField X(grid), Y(grid);
        FermionField x(rbgrid), y(rbgrid);
        
        X = Zero(), Y = Zero();
        x = Zero(), y = Zero();

        // extract Xl solution vector
        pickCheckerboard(Even, x, vecx[l]);
        setCheckerboard(X, x);

        // extract Yl solution vector
        pickCheckerboard(Odd, y, vecx[l]);
        setCheckerboard(Y, y);

        // accumulate outer products
        for (int mu = 0; mu < Nd; ++mu) {
          // 1-link contribution
          t = outerProduct(Cshift(Y, mu, 1), X); 
          t -= outerProduct(Cshift(X, mu, 1), Y);
          PokeIndex<LorentzIndex>(tdSdX, ctx.factor(l)*t, mu);
          
          // 3-link (Naik) contribution
          t = outerProduct(Cshift(Y, mu, 3), X);
          t -= outerProduct(Cshift(X, mu, 3), Y);
          PokeIndex<LorentzIndex>(tdSdWWW, ctx.factor(l)*t, mu);
        }

        // increment
        dSdX[species] += tdSdX;
        dSdWWW[species] += tdSdWWW;
        ++l;
    } }
    
    // calculate full HISQ force
    milcSmearDerivative(UdSdU, dSdX, dSdWWW, W, V, U, ctx);
  }

};

// undefine macros to prevent conflicts
#undef HISQLOOP0
#undef HISQLOOP1
#undef HISQLOOP2
#undef HISQLOOP3
#undef HISQLEPAGE
#undef HISQNAIK
#undef when

NAMESPACE_END(Grid);

#endif // QCD_UTILS_HISQ_IMPL_H
