/*************************************************************************************
Grid physics library, www.github.com/paboyle/Grid 

Source file: ./lib/qcd/action/gauge/PlaqPlusRectanglePlusParallelogramAction.h

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

See the full license in the file "LICENSE" in the top level distribution directory
*************************************************************************************/
/*  END LEGAL */

/**
 * @file PlaqPlusRectanglePlusParallelogramAction.h
 * @brief Interface for implementation of one-loop gauge actions
 * @author Curtis Taylor Peterson
 * @details
 * Implementation of one-loop gauge action in Grid, including plaquette, rectangle,
 * and parallelogram terms. The one-loop gauge action is given is
 * (1) S = β [ c_plaq ∑(1 - 1/Nc ReTr(U_plaquette))
 *           + c_rect ∑(1 - 1/Nc ReTr(U_rectangle))
 *           + c_parallelogram ∑(1 - 1/Nc ReTr(U_parallelogram)) ].
 * See references below for details on each term. 
 * 
 * Note that PeriodicSymanzikOneLoopGaugeAction is normalized such that
 * (2) β = 10/g0^2.
 * This is to be consistent with the MILC code.
 * 
 * References:
 * * Alford, M., Dimm, W., Lepage, G.P., Hockney, G., Mackenzie, P.B. (1995): 
 *   https://doi.org/10.1016/0370-2693(95)01131-9
 * * Follana, E. et al.: https://doi.org/10.1103/PhysRevD.75.054502
 * * Lepage, G.P., Mackenzie, P.B. (1993): https://doi.org/10.1103/PhysRevD.48.2250
 * * MILC Collaboration (2010): https://doi.org/10.1103/PhysRevD.82.074501
 * 
 * Acknowledgements:
 *   Curtis Taylor Peterson would like to thank James Osborn for developing/testing
 *   the implementations of the Symanzik one-loop gauge action in QEX and QOPQDP, from 
 *   which the parallelogram force is based on and has been tested against.
 *   
 *   This material is based upon work supported by the U.S. Department of Energy, 
 *   Office of Science, Office of Advanced Scientific Computing Research, Scientific 
 *   Discovery through Advanced Computing (SciDAC) program.
 */

#pragma once 

#ifndef QCD_PERIODIC_PLAQ_PLUS_RECTANGLE_PLUS_PARALLELOGRAM_GAUGE_ACTION_H
#define QCD_PERIODIC_PLAQ_PLUS_RECTANGLE_PLUS_PARALLELOGRAM_GAUGE_ACTION_H

#include <Grid/qcd/utils/Transporter.h>
#include <Grid/perfmon/Tracing.h>

NAMESPACE_BEGIN(Grid);

//
// useful macros for readability
//

// more readable if statement for plaquette
#define PLAQUETTE(exec) if (ctx.cp != 0.0) { exec; }

// more readable if statement for rectangle
#define RECTANGLE(exec) if (ctx.cr != 0.0) { exec; }

// more readable if statement for parallelogram
#define PARALLELOGRAM(exec) if (ctx.cpg != 0.0) { exec; }

// more readable if statement for one-loop
#define IMPROVED(exec) if ((ctx.cr != 0.0) or (ctx.cpg != 0.0)) { exec; }

//
// useful data structure for one-loop gauge action
//

struct OneLoopGaugeActionContext {
  RealD beta, cp, cr, cpg;

  OneLoopGaugeActionContext(RealD beta, RealD cp, RealD cr, RealD cpg): 
    beta(beta), 
    cp(beta*cp), 
    cr(beta*cr), 
    cpg(beta*cpg) { };

  OneLoopGaugeActionContext(const OneLoopGaugeActionContext& ctx):
    beta(ctx.beta), cp(ctx.cp), cr(ctx.cr), cpg(ctx.cpg) { };
};

//
// helper procedures for MILC coefficients with nf sea-quark flavors
// 

inline RealD oneLoopMILCRectangleCoefficient(RealD u0, int nf)
{ return -(1.0 - (0.6264 - 1.1746 * nf) * std::log(u0)) / (20.0 * u0 * u0); }

inline RealD oneLoopMILCParallelogramCoefficient(RealD u0, int nf)
{ return (0.0433 - 0.0156 * nf) * std::log(u0) / (u0 * u0); }

//
// one-loop gauge action class
//

template<class Gimpl>
class PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction: public Action<typename Gimpl::GaugeField> {
/**
 * @brief One-loop gauge action implementation in Grid
 * @author Curtis Taylor Peterson
 * @details
 * Action implementation for Symanzik one-loop gauge action. Comprised of plaquette,
 * rectangle, and parallelogram terms. See references above and therein for details.
 */

public: INHERIT_GIMPL_TYPES(Gimpl);

private:
  typedef typename std::vector<GaugeLinkField> GaugeLorentzField;

public:
  GridBase* grid;  

  const RealD DEPTH = 1;
  PaddedCell cell;

  OneLoopGaugeActionContext context;
  RealD invNc, actionNorm;

private:
  PeriodicBC::TransporterStencil _stencil;

  void init(GridBase* tightGrid, PaddedCell& pcell) { 
    grid = (GridBase*)cell.grids.back(); 
    invNc = 1.0/RealD(Nc);
    actionNorm = RealD(Nd*(Nd - 1))*tightGrid->gSites();
    _stencil = PeriodicBC::TransporterStencil(grid, DEPTH);
  }

public:
  PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction(
    GridCartesian* tightGrid
  ):cell(DEPTH, tightGrid), context(0.0, 0.0, 0.0, 0.0) { init(tightGrid, cell); };

  PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction(
    GridCartesian* tightGrid, 
    const OneLoopGaugeActionContext ctx
  ):cell(DEPTH, tightGrid), context(ctx) { init(tightGrid, cell); };
  
  PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction(
    GridCartesian* tightGrid, 
    RealD beta, 
    RealD cp, 
    RealD cr, 
    RealD cpg
  ):cell(DEPTH, tightGrid), context(OneLoopGaugeActionContext(beta, cp, cr, cpg)) 
  { init(tightGrid, cell); };

public:
  virtual std::string action_name() { 
    return "PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction";
  }

  virtual std::string LogParameters() {
    std::stringstream sstream;
    std::string actionName = "[" + action_name() + "] ";
    sstream << GridLogMessage << actionName << "c_plaq: " << context.cp << std::endl;
    sstream << GridLogMessage << actionName << "c_rect: " << context.cr << std::endl;
    sstream << GridLogMessage << actionName << "c_para: " << context.cpg << std::endl;
    return sstream.str();
  }

public:
  // break up memory layout of gauge field into std::vector of gauge link fields
  inline const GaugeLorentzField toLorentz(const GaugeField& U) { 
    GaugeLorentzField u(Nd, grid);
    for (int mu = 0; mu < Nd; ++mu) u[mu] = PeekIndex<LorentzIndex>(U, mu);
    return u; 
  }

  // aggregate std::vector of gauge link fields into layout of gauge field
  inline const GaugeField toGauge(GaugeLorentzField& u) { 
    GaugeField U(grid); 
    for (int mu = 0; mu < Nd; ++mu) PokeIndex<LorentzIndex>(U, u[mu], mu);
    return U; 
  }

public:
  RealD S(const GaugeField &U, const OneLoopGaugeActionContext ctx) {
    /**
     * @brief One-loop gauge action
     * @author Curtis Taylor Peterson
     * @details
     * Calculates full Symanzik one-loop gauge action using the forward and
     * backward staples Fμν and Bμν defined in `Transporter::plaquetteStaples`.
     * Write
     * (1) <A,B> := tr[A†B].
     * The plaquette and the two rectangles in the μ-ν plane are then
     * (2) Pμν(n) = <Uμ(n),Fμν(n)>,
     * (3) Rμν(n) = <Bμν(n),Fμν(n)> + <Bνμ(n),Fνμ(n)>.
     * Only the real parts enter the action. Notice that (2)-(3) are scalar
     * contractions: once the staples are available, no further matrix products
     * are needed for either contribution. `loopTrace` keeps these contractions
     * in one lattice expression, without constructing temporary scalar fields.
     *
     * The parallelograms can also be closed using the same staples. Fix μ>ν and
     * collect paths through the remaining directions ρ:
     * (4) Hμν(n) = ∑_{ρ≠μ,ν} Bμρ(n)† Bνρ(n)
     *             + ∑_{ρ>μ} Fμρ(n)† Fνρ(n).
     * Closing these paths gives
     * (5) Gμν(n) = <Uν(n+μ), Hμν(n) Uμ(n+ν)>.
     * Link unitarity cancels the backtracking segments in these products. For
     * each unordered direction triple, the backward sum supplies three distinct
     * parallelograms; the forward sum supplies the fourth. The restriction ρ>μ
     * in the forward sum prevents counting the same loop again in another plane.
     * There are qμν = (Nd-2) + (Nd-1-μ) loops in (5), or sixteen altogether in
     * four dimensions. In `parallelogramLoops`, Hμν is `paths`; qμν is the
     * number of staple pairs supplied by the caller.
     *
     * With β included in the context coefficients, the action is
     * (6) S = (1/Nc) ∑_{n,μ>ν} [ cp (Nc - Re Pμν(n))
     *                         + cr (2Nc - Re Rμν(n))
     *                         + cpg (qμν Nc - Re Gμν(n)) ].
     * The key thing to note is that (4) adds paths before multiplying by their
     * common closing link. All contractions accumulate into one scalar field,
     * followed by interior extraction and a single global sum. The factor
     * `actionNorm` used during accumulation is removed on return.
     *
     * When only the plaquette is requested, <Aνμ,Aμν> from `plaquetteStaples`
     * gives the same real trace as (2). In that case we construct just the two
     * corners and avoid constructing the full set of staples.
     *
     * References:
     * * Quantum EXpressions (QEX): https://github.com/jcosborn/qex
     * * QOPQDP [SciDAC]: https://github.com/usqcd-software/qopqdp
     */
    RealD cp = invNc*ctx.cp/actionNorm;
    RealD cr = invNc*ctx.cr/actionNorm;
    RealD cpg = invNc*ctx.cpg/actionNorm;

    PeriodicBC::Transporter<Gimpl> u(cell, U, _stencil);
    GaugeLorentzField sf, sb;
    ComplexField action(grid);

    tracePush("PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction::S");
    action = Zero();
    if ((ctx.cr == 0.0) and (ctx.cpg == 0.0)) {
      for (int mu = 1; mu < Nd; ++mu) {
        for (int nu = 0; nu < mu; ++nu) {
          GaugeLinkField ta = u.CovShiftFwd(mu, nu);
          GaugeLinkField tb = u.CovShiftFwd(nu, mu);
          action += cp*(RealD(Nc) - localInnerProduct(tb, ta));
      } }
    } else {
      u.plaquetteStaples(sf, sb);
      for (int mu = 1; mu < Nd; ++mu) {
        for (int nu = 0; nu < mu; ++nu) {
          int mn = u.stapleIndex(mu, nu), nm = u.stapleIndex(nu, mu);
          if ((ctx.cp != 0.0) or (ctx.cr != 0.0))
            action += cp*(RealD(Nc) - u.loopTrace(u.link(mu), sf[mn]))
              + cr*(RealD(2*Nc) - u.loopTrace(sb[mn], sf[mn]) - u.loopTrace(sb[nm], sf[nm]));
          PARALLELOGRAM(
            typename PeriodicBC::Transporter<Gimpl>::GaugeLinkPairs pairs;
            for (int ro = 0; ro < Nd; ++ro) {
              if ((ro == mu) or (ro == nu)) continue;
              int mr = u.stapleIndex(mu, ro);
              int nr = u.stapleIndex(nu, ro);
              pairs.emplace_back(&sb[mr], &sb[nr]);
              if (ro > mu) pairs.emplace_back(&sf[mr], &sf[nr]);
            }
            action += cpg*(RealD(Nc*pairs.size()) - u.parallelogramLoops(pairs, mu, nu));
          )
      } }
    }
    auto actionSumTensor = sum(cell.Extract(action));
    auto actionSum = TensorRemove(actionSumTensor);
    tracePop("PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction::S");
    
    return actionNorm*actionSum.real();
  }

  virtual void deriv(
    const GaugeField& U, 
    GaugeField& dSdU, 
    const OneLoopGaugeActionContext ctx
  ) {
    /**
     * @brief One-loop gauge action derivative
     * @author Curtis Taylor Peterson
     * @details
     * Calculates full Symanzik one-loop gauge action derivative. See
     * `Transporter::plaquetteStaples` for Fμν and Bμν, and `S` for the action.
     * Implementation of parallelogram derivative is thanks to James Osborn,
     * who wrote the efficient parallelogram force in QOP/QDP and QEX that
     * this is based on. I cannot thank James enough for his help.
     *
     * Write Kμ(n) for the weighted sum of open paths from n to n+μ, stored in
     * `dsdu`. The local coefficients cp, cr, and cpg include β/(2Nc). We first
     * add the plaquette contribution:
     * (1) Kμ(n) += cp ∑_{ν≠μ} [ Fμν(n) + Bμν(n) ].
     * When neither rectangle nor parallelogram is requested, Transporter adds
     * these staples directly without storing the full set of F and B fields.
     *
     * For the rectangle, fix an ordered pair μ≠ν and define
     * (2) C(n) = Uμ(n) Uν(n+μ),
     * (3) H(n) = Bμν(n) Uν(n+μ) + Uμ(n) Fνμ(n+μ).
     * These are `corner` and `paths` in `addRectangleStaples`. The outer links
     * can be factored out of the three upper ν and three lower μ staples:
     * (4) Kν(n) += cr [ C(n) Fμν(n+ν)† + H(n) Uμ(n+ν)† ],
     * (5) L(n) = Bνμ(n)† C(n) + Uν(n)† H(n),
     * (6) Kμ(n) += cr L(n-ν).
     * Here L is the temporary field `backward`. Reversing μ and ν supplies the
     * other three staples in each direction. Equations (2)-(5) require seven
     * matrix products per ordered pair. The shifted values in (6) are computed
     * from the existing padded inputs, so L needs no additional halo exchange.
     *
     * For the parallelogram, fix μ and ν, then sum over ρ>ν with ρ≠μ. For each
     * ρ, take Xμ = Fμρ, Xν = Fνρ and then Xμ = Bμρ, Xν = Bνρ. Accumulate the
     * following four path sums over both choices of X and all such ρ:
     * (7)  A(n) = ∑ Xν(n) Xμ(n+ν),
     * (8)  B(n) = ∑ Xμ(n+ν) Xν(n+μ)†,
     * (9)  C(n) = ∑ Xν(n-ν)† Xμ(n-ν),
     * (10) D(n) = ∑ Xμ(n-ν) Xν(n-ν+μ).
     * These are `ta`, `tb`, `tc`, and `td` in `addStapleCrossTerms`.
     * Completing the staples gives
     * (11) Kμ(n) += cpg [ A(n) Uν(n+μ)† + Uν(n) B(n)
     *                    + C(n) Uν(n-ν+μ) + Uν(n-ν)† D(n) ].
     * The restriction ρ>ν counts each pair of transverse directions once.
     * The key thing to note is that their outer ν links do not depend on ρ or
     * the choice of forward/backward staple. We can therefore perform all four
     * sums before the final products in (11). These sums stay local to each
     * site; they do not require temporary lattice fields or halo exchanges.
     *
     * Once all contributions have been added, the output is
     * (12) dSdUμ(n) = -Ta[ Kμ(n) Uμ(n)† ],
     * where Ta projects onto traceless anti-Hermitian matrices. For an
     * anti-Hermitian traceless variation Uμ(ε) = exp(ε Pμ) Uμ, this satisfies
     * (13) dS/dε = -2 Re ∑_{n,μ} tr[ Pμ(n) dSdUμ(n) ].
     * Linearity lets us apply (12) once to the combined contributions, then
     * aggregate the link fields and extract the interior once.
     *
     * References:
     * * QOPQDP [SciDAC]: https://github.com/usqcd-software/qopqdp
     * * Quantum EXpressions (QEX): https://github.com/jcosborn/qex
     */
    RealD cp = 0.5*invNc*ctx.cp;
    RealD cr = 0.5*invNc*ctx.cr;
    RealD cpg = 0.5*invNc*ctx.cpg;

    PeriodicBC::Transporter<Gimpl> u(cell, U, _stencil);
    GaugeLorentzField dsdu(Nd, grid), sf, sb;

    tracePush("PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction::deriv");
    IMPROVED(u.plaquetteStaples(sf, sb))
    for (int mu = 0; mu < Nd; ++mu) {
      dsdu[mu] = Zero();
      for (int nu = 0; nu < Nd; ++nu) {
        if (nu == mu) continue;
        PLAQUETTE(
          if (sf.empty()) u.addStaple(dsdu[mu], mu, nu, cp);
          else dsdu[mu] += cp*(sf[u.stapleIndex(mu, nu)] + sb[u.stapleIndex(mu, nu)]);
        )
    } }

    RECTANGLE(
      for (int mu = 0; mu < Nd; ++mu) {
        for (int nu = 0; nu < Nd; ++nu) {
          if (nu == mu) continue;
          int mn = u.stapleIndex(mu, nu);
          int nm = u.stapleIndex(nu, mu);
          u.addRectangleStaples(dsdu[mu], dsdu[nu], sf[mn], sf[nm], sb[mn], sb[nm], mu, nu, cr);
      } }
    )

    PARALLELOGRAM(
      for (int mu = 0; mu < Nd; ++mu) {
        for (int nu = 0; nu < Nd - 1; ++nu) {
          int nro = Nd - nu - 1 - (mu > nu);
          if ((nu == mu) or (nro == 0)) continue;
          typename PeriodicBC::Transporter<Gimpl>::GaugeLinkPairs pairs;
          
          for (int ro = nu + 1; ro < Nd; ++ro) {
            if (ro == mu) continue;
            int mr = u.stapleIndex(mu, ro);
            int nr = u.stapleIndex(nu, ro);
            for (int side = 0; side < 2; ++side) 
            { auto& s = side ? sb : sf; pairs.emplace_back(&s[mr], &s[nr]); }
          }
          
          u.addStapleCrossTerms(dsdu[mu], pairs, mu, nu, cpg);
      } }
    )

    for (int mu = 0; mu < Nd; ++mu) dsdu[mu] = -Ta(dsdu[mu]*adj(u.link(mu)));
    dSdU = cell.Extract(toGauge(dsdu));
    tracePop("PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction::deriv");
  }

public:
  virtual void refresh(
    const GaugeField& U, 
    GridSerialRNG& sRNG, 
    GridParallelRNG& pRNG
  ) { }

  virtual RealD S(const GaugeField &U) { return S(U, context); }

  virtual void deriv(const GaugeField& U, GaugeField& dSdU) 
  { deriv(U, dSdU, context); }
};

NAMESPACE_END(Grid);

#endif
