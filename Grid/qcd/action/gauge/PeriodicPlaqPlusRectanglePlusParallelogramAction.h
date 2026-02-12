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

#include <Grid/qcd/utils/Transporters.h>
#include <Grid/perfmon/Tracing.h>

NAMESPACE_BEGIN(Grid);

//
// useful macros for readability
//

// more readable if statement for plaquette
#define PLAQUETTE(exec) if (ctx.cp != 0.0) exec;

// more readable if statement for rectangle
#define RECTANGLE(exec) if (ctx.cr != 0.0) exec;

// more readable if statement for parallelogram
#define PARALLELOGRAM(exec) if (ctx.cpg != 0.0) exec;

// more readable if statement for one-loop
#define IMPROVED(exec) if ((ctx.cr != 0.0) or (ctx.cpg != 0.0)) exec;

//
// useful data structure for one-loop gauge action
//

struct OneLoopGaugeActionContext {
  RealD beta, cp, cr, cpg;

  OneLoopGaugeActionContext(RealD beta, RealD cp, RealD cr, RealD cpg): 
    beta(beta), 
    cp(cp), 
    cr(cr), 
    cpg(cpg) { 
    cp *= beta;
    cr *= beta;
    cpg *= beta; 
  };

  OneLoopGaugeActionContext(const OneLoopGaugeActionContext& ctx):
    beta(ctx.beta), cp(ctx.cp), cr(ctx.cr), cpg(ctx.cpg) { };
  
  OneLoopGaugeActionContext(RealD beta, RealD u0, int nf): beta(beta) {
    // Eqn. (A2) of [MILC Collaboration (2010): https://doi.org/10.1103/PhysRevD.82.074501]
    RealD flavors = RealD(nf);
    RealD crf = 1. - (0.6264 - 1.1746*flavors)*std::log(u0);
    RealD cpgf = (0.0433 - 0.0156*flavors)*std::log(u0);
    cp = beta;
    cr = -beta*crf/20./u0/u0;
    cpg = u0 == 1.0 ? 0.0 : beta*cpgf/u0/u0;
  }
};

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
  void init(GridBase* tightGrid, PaddedCell& pcell) { 
    grid = (GridBase*)cell.grids.back(); 
    invNc = 1.0/RealD(Nc);
    actionNorm = RealD(Nd*(Nd - 1))*tightGrid->gSites();
  }

public:
  PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction(
    GridCartesian* tightGrid
  ):cell(DEPTH, tightGrid) { init(tightGrid, cell); };

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
     * Calculates full Symanzik one-loop gauge action. Rectangle contribution 
     * reuses plaquette corners. Parallelogram does not attemp to reuse any 
     * pre-computed constructs, though it could and should (TODO). There are four
     * parallelograms per mu-nu pair. In some sense, they are pi/2 "rotations" of 
     * each other in the mu-nu plane. 
     */
    RealD cp = invNc*ctx.cp/actionNorm;
    RealD cr = invNc*ctx.cr/actionNorm;
    RealD cpg = 2.0*invNc*ctx.cpg/actionNorm;

    PeriodicBC::Transporters<Gimpl> u(cell, U);
    
    LatticeComplex action(grid);

    GaugeLinkField diag(grid);
    GaugeLinkField ta(grid), tb(grid);
    GaugeLinkField tc(grid), td(grid);

    tracePush("PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction::S");
    diag = 1.0, action = 0.0;
    for (int mu = 1; mu < Nd; ++mu) {
      for (int nu = 0; nu < mu; ++nu) {
        ta = u.CovShiftFwd(mu, u.link(nu));
        tb = u.CovShiftFwd(nu, u.link(mu));

        PLAQUETTE(action += cp*trace(diag - ta*adj(tb)))

        RECTANGLE(
          tc = u.CovShiftFwd(mu, ta, false);
          td = u.CovShiftFwd(mu, tb, u.CovShiftIdentFwd(nu, mu), false);
          action += cr*trace(diag - tc*adj(td));

          tc = u.CovShiftFwd(nu, tb, false);
          td = u.CovShiftFwd(nu, ta, u.CovShiftIdentFwd(mu, nu), false);
          action += cr*trace(diag - tc*adj(td));
        )

        PARALLELOGRAM(
          for (int i = 0; i < nu; ++i) {
            tc = u.CovShiftFwd(mu, u.CovShiftFwd(nu, i), false);
            td = u.CovShiftFwd(i, u.CovShiftFwd(nu, mu), false);
            action += cpg*trace(diag - tc*adj(td)); // ++

            tc = u.CovShiftFwd(nu, u.CovShiftFwd(i, mu), false);
            td = u.CovShiftFwd(mu, u.CovShiftFwd(i, nu), false); 
            action += cpg*trace(diag - tc*adj(td)); // --

            tc = u.CovShiftFwd(i, u.CovShiftFwd(mu, nu), false);
            td = u.CovShiftFwd(nu, u.CovShiftFwd(mu, i), false);
            action += cpg*trace(diag - tc*adj(td)); // +-

            tc = u.CovShiftFwd(mu, u.CovShiftBck(nu, i), false);
            td = u.CovShiftFwd(i, u.CovShiftBck(nu, mu), false);
            action += cpg*trace(diag - tc*adj(td)); // -+
        } ) 
    } } 
    auto actionSumTensor = sum(u.toTightGrid(action));
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
     * Calculates full Symanzik one-loop gauge action derivative. Entire force can
     * be calculated from just staples. Rectangle reuses some plaquette staples.
     * Parallelogram reuses some rectangle staples. Implementation of parallelogram
     * derivative is thanks to James Osborn, who wrote the efficient implementation
     * of the parallelogram force in QOP/QDP and QEX that this is based on. I cannot
     * thank James enough for his help.
     * 
     * References:
     * * QOPQDP [SciDAC]: https://github.com/usqcd-software/qopqdp
     * * Quantum EXpressions (QEX): https://github.com/jcosborn/qex
     */
    RealD cp = 0.5*invNc*ctx.cp;
    RealD cr = 0.5*invNc*ctx.cr;
    RealD cpg = invNc*ctx.cpg;

    PeriodicBC::Transporters<Gimpl> u(cell, U); 
    
    GaugeLorentzField dsdu = toLorentz(u.toPaddedGrid(dSdU));
    GaugeLinkField ta(grid), tb(grid);
    GaugeLinkField tc(grid), td(grid);

    std::vector<GaugeLorentzField> sf(Nd, GaugeLorentzField(Nd, grid));
    std::vector<GaugeLorentzField> sb(Nd, GaugeLorentzField(Nd, grid));

    // plaquette + rectangle force; parallelogram preparation
    tracePush("PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction::deriv");
    for (int mu = 0; mu < Nd; ++mu) {
      dsdu[mu] = Zero();
      for (int nu = 0; nu < Nd; ++nu) {
        if (nu == mu) continue;

        // rectangle + plaquette force

        ta = u.upperStaple(mu, nu);
        tb = u.lowerStaple(mu, nu);

        PLAQUETTE(dsdu[mu] += cp*(ta + tb))

        IMPROVED(
          tc = u.rightStaple(mu, nu);
          td = u.leftStaple(mu, nu);
        )

        RECTANGLE(
          dsdu[mu] += cr*u.upperStaple(u.link(nu), tc, mu, nu, false);
          dsdu[mu] += cr*u.lowerStaple(u.link(nu), tc, mu, nu, false);
          dsdu[mu] += cr*u.upperStaple(td, u.link(nu), mu, nu, false);
          dsdu[mu] += cr*u.lowerStaple(td, u.link(nu), mu, nu, false);
          dsdu[mu] += cr*u.upperStaple(ta, mu, nu, false);
          dsdu[mu] += cr*u.lowerStaple(tb, mu, nu, false);
        )

        // parallelogram preparation

        PARALLELOGRAM(
          if (mu < nu) continue;
          sf[mu][nu] = ta;
          sb[mu][nu] = tb;
          sf[nu][mu] = tc;
          sb[nu][mu] = td;
        )
      }
      dsdu[mu] = -Ta(dsdu[mu]*adj(u.link(mu)));
    }
    dSdU = u.toTightGrid(toGauge(dsdu));

    // parallelogram force
    PARALLELOGRAM(
      for (int mu = 0; mu < Nd; ++mu) {
        dsdu[mu] = Zero();
        for (int nu = 0; nu < Nd; ++nu) {
          if (nu == mu) continue;
          for (int ro = nu + 1; ro < Nd; ++ro) {
            if ((ro == mu) or (ro == nu)) continue;
            dsdu[mu] += u.staple(sf[mu][ro], sf[nu][ro], u.link(nu), mu, nu, false);
            dsdu[mu] += u.staple(sf[mu][ro], u.link(nu), sf[nu][ro], mu, nu, false);
            dsdu[mu] += u.staple(sb[mu][ro], sb[nu][ro], u.link(nu), mu, nu, false);
            dsdu[mu] += u.staple(sb[mu][ro], u.link(nu), sb[nu][ro], mu, nu, false);
        } }
        dsdu[mu] = -cpg*Ta(dsdu[mu]*adj(u.link(mu)));
      }
      dSdU += u.toTightGrid(toGauge(dsdu));
    )
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

template <class Gimpl>
class PeriodicSymanzikOneLoopGaugeAction: public PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction<Gimpl> {
/**
 * @brief Symanzik one-loop gauge action implementation in Grid
 * @author Curtis Taylor Peterson
 * @details
 * Wrapper for PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction with one-loop
 * coefficients. Note that bare gauage coupling is normalized such that
 * (1) β = 10/g0^2.
 * This is to be consistent with the MILC code.
 */

public:
  INHERIT_GIMPL_TYPES(Gimpl);

public:
  PeriodicSymanzikOneLoopGaugeAction(
    GridCartesian* tightGrid,
    RealD beta,
    RealD u0,
    int nf
  ):PeriodicPlaqPlusRectanglePlusParallelogramGaugeAction<Gimpl>(
      tightGrid, 
      OneLoopGaugeActionContext(beta, u0, nf)
    ) { };

public:
  virtual std::string action_name() { return "PeriodicSymanzikOneLoopGaugeAction"; }
};
NAMESPACE_END(Grid);

#endif