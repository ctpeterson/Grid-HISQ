/**
 * @file OneLoopGaugeAction.h
 * @brief Interface for implementation of one-loop gauge actions
 * @author Curtis Taylor Peterson
 * @details
 * ...
 * 
 * References:
 * * Alford, M., Dimm, W., Lepage, G.P., Hockney, G., Mackenzie, P.B. (1995): 
 *   https://doi.org/10.1016/0370-2693(95)01131-9
 * * Follana, E. et al.: https://doi.org/10.1103/PhysRevD.75.054502
 * * Lepage, G.P., Mackenzie, P.B. (1993): https://doi.org/10.1103/PhysRevD.48.2250
 * * MILC Collaboration (2010): https://doi.org/10.1103/PhysRevD.82.074501
 */

#pragma once 

#ifndef QCD_ONE_LOOP_GAUGE_ACTION_H
#define QCD_ONE_LOOP_GAUGE_ACTION_H

#include <Grid/qcd/utils/Transporters.h>

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

//
// useful data structures for one-loop gauge action
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
};

//
// one-loop gauge action class
//

template<class Gimpl>
class OneLoopGaugeAction: public Action<typename Gimpl::GaugeField> {
/**
 * @brief One-loop gauge action implementation in Grid
 * @author Curtis Taylor Peterson
 * @details
 * ...
 * 
 * References:
 * * Alford, M., Dimm, W., Lepage, G.P., Hockney, G., Mackenzie, P.B. (1995): 
 *   https://doi.org/10.1016/0370-2693(95)01131-9
 * * Follana, E. et al.: https://doi.org/10.1103/PhysRevD.75.054502
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
  OneLoopGaugeAction(GridCartesian* tightGrid): cell(DEPTH, tightGrid) 
  { init(tightGrid, cell); };

  OneLoopGaugeAction(GridCartesian* tightGrid, const OneLoopGaugeActionContext ctx):
    cell(DEPTH, tightGrid), context(ctx) { init(tightGrid, cell); };
  
  OneLoopGaugeAction(
    GridCartesian* tightGrid, 
    RealD beta, 
    RealD cp, 
    RealD cr, 
    RealD cpg
  ):cell(DEPTH, tightGrid), context(OneLoopGaugeActionContext(beta, cp, cr, cpg)) 
  { init(tightGrid, cell); };

public:
  virtual std::string action_name() { return "OneLoopGaugeAction"; }

  virtual std::string LogParameters() {
    std::stringstream sstream;
    std::string actionName = "[" + action_name() + "] ";
    sstream << GridLogMessage << actionName << "c_plaq: " << context.cp << std::endl;
    sstream << GridLogMessage << actionName << "c_rect: " << context.cr << std::endl;
    sstream << GridLogMessage << actionName << "c_para: " << context.cpg << std::endl;
    return sstream.str();
  }

private:
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
     * ...
     */
    RealD cp = invNc*ctx.cp/actionNorm;
    RealD cr = invNc*ctx.cr/actionNorm;
    RealD cpg = 2.0*invNc*ctx.cpg/actionNorm;
    PeriodicBC::Transporters<Gimpl> u(cell, U);
    LatticeComplex action(grid);
    GaugeLinkField diag(grid);
    GaugeLinkField ta(grid), tb(grid);
    GaugeLinkField tc(grid), td(grid);

    diag = 1.0, action = 0.0;
    for (int mu = 1; mu < Nd; ++mu) {
      for (int nu = 0; nu < mu; ++nu) {
        ta = u.CovShiftFwd(mu, u.link(nu));
        tb = u.CovShiftFwd(nu, u.link(mu));

        PLAQUETTE(action += cp*trace(diag - ta*adj(tb)))

        RECTANGLE(
          tc = u.CovShiftFwd(mu, ta);
          td = u.CovShiftFwd(mu, tb, u.CovShiftIdentFwd(nu, mu));
          action += cr*trace(diag - tc*adj(td));

          tc = u.CovShiftFwd(nu, tb);
          td = u.CovShiftFwd(nu, ta, u.CovShiftIdentFwd(mu, nu));
          action += cr*trace(diag - tc*adj(td));
        )

        PARALLELOGRAM(
          for (int i = 0; i < nu; ++i) {
            tc = u.CovShiftFwd(mu, u.CovShiftFwd(nu, i));
            td = u.CovShiftFwd(i, u.CovShiftFwd(nu, mu));
            action += cpg*trace(diag - tc*adj(td)); // ++

            /*
            tc = u.CovShiftFwd(nu, u.CovShiftFwd(i, mu));
            td = u.CovShiftFwd(mu, u.CovShiftFwd(i, nu)); 
            action += cpg*trace(diag - tc*adj(td)); // --

            tc = u.CovShiftFwd(i, u.CovShiftFwd(mu, nu));
            td = u.CovShiftFwd(nu, u.CovShiftFwd(mu, i));
            action += cpg*trace(diag - tc*adj(td)); // +-

            tc = u.CovShiftFwd(mu, u.CovShiftBck(nu, i));
            td = u.CovShiftFwd(i, u.CovShiftBck(nu, mu));
            action += cpg*trace(diag - tc*adj(td)); // -+
            */
        } ) 
    } } 

    auto actionSumTensor = sum(u.toTightGrid(action));
    auto actionSum = TensorRemove(actionSumTensor);
    return actionNorm*actionSum.real();
  }

  virtual void deriv(
    const GaugeField& U, 
    GaugeField& dSdU, 
    const OneLoopGaugeActionContext ctx
  ) {
    std::vector<int> mus = {0, 0, 0, 1};
    std::vector<int> nus = {1, 1, 2, 2};
    std::vector<int> ros = {2, 3, 3, 3};

    RealD cp = 0.5*invNc*ctx.cp;
    RealD cr = 0.5*invNc*ctx.cr;
    RealD cpg = invNc*ctx.cpg;

    PeriodicBC::Transporters<Gimpl> u(cell, U); 
    
    GaugeLorentzField dsdu = toLorentz(u.toPaddedGrid(dSdU));
    GaugeLinkField ta(grid), tb(grid);
    GaugeLinkField tc(grid), td(grid);

    // plaquette and rectangle

    for (int mu = 0; mu < Nd; ++mu) dsdu[mu] = Zero();

    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = 0; nu < Nd; ++nu) {
        if (nu == mu) continue;

        ta = u.upperStaple(mu, nu);
        tb = u.lowerStaple(mu, nu);

        PLAQUETTE(dsdu[mu] += cp*(ta + tb))

        RECTANGLE( 
          tc = u.leftStaple(mu, nu);
          td = u.rightStaple(mu, nu);

          dsdu[mu] += cr*u.upperStaple(tc, u.link(nu), mu, nu);
          dsdu[mu] += cr*u.lowerStaple(tc, u.link(nu), mu, nu);
          dsdu[mu] += cr*u.upperStaple(u.link(nu), td, mu, nu);
          dsdu[mu] += cr*u.lowerStaple(u.link(nu), td, mu, nu);
          dsdu[mu] += cr*u.upperStaple(ta, mu, nu);
          dsdu[mu] += cr*u.lowerStaple(tb, mu, nu);
        )
      }
      dsdu[mu] = -Ta(dsdu[mu]*adj(u.link(mu)));
    }
    dSdU = u.toTightGrid(toGauge(dsdu));

    // parallelogram

    PARALLELOGRAM(
      for (int mu = 0; mu < Nd; ++mu) dsdu[mu] = Zero();

      for (int idx = 0; idx < mus.size(); ++idx) {
        int mu = mus[idx]; 
        int nu = nus[idx]; 
        int ro = ros[idx];

        tc = u.CovShiftFwd(mu, u.CovShiftFwd(nu, ro));
        td = u.CovShiftFwd(ro, u.CovShiftFwd(nu, mu));
        dsdu[mu] += tc*adj(td); // ++ mu derivative

        tc = u.CovShiftFwd(nu, u.CovShiftFwd(ro, u.reverse(mu)));
        td = u.CovShiftBck(mu, u.CovShiftFwd(ro, nu));
        dsdu[nu] += tc*adj(td); // ++ nu derivative

        tc = u.CovShiftFwd(ro, u.CovShiftBck(mu, u.reverse(nu)));
        td = u.CovShiftBck(nu, u.CovShiftBck(mu, ro));
        dsdu[ro] += tc*adj(td); // ++ ro derivative
      }

      for (int mu = 0; mu < Nd; ++mu) dsdu[mu] = cpg*Ta(dsdu[mu]);
      dSdU += u.toTightGrid(toGauge(dsdu));
    )
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