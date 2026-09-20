/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/qcd/action/pseudofermion/FourFlavorStaggeredEvenEvenRatio.h

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
  @file FourFlavorStaggeredEvenEvenRatio.h
  @author Curtis Taylor Peterson
*/

#pragma once
#include <Grid/Grid.h>

#ifndef QCD_PSEUDOFERMION_FOUR_FLAVOR_STAGGERED_EVEN_EVEN_RATIO_H
#define QCD_PSEUDOFERMION_FOUR_FLAVOR_STAGGERED_EVEN_EVEN_RATIO_H

NAMESPACE_BEGIN(Grid);

template <class Impl>
class FourFlavorStaggeredEvenEvenRatioPseudoFermionAction: public Action<typename Impl::GaugeField> {
/**
 * @class Grid::FourFlavorStaggeredEvenEvenRatioPseudoFermionAction
 * @brief Four flavor staggered even-even ratio pseudofermion action
 * @author Curtis Taylor Peterson
 * @details 
 * Implements four-flavor ratio action for even-even (i.e., reduced) staggered
 * pseudofermion fields:
 * (1) S = Phi^dag N (D^dag D)^-1 N^dag Phi,
 * where Phi is defined only on even sites and
 * (2) N = K_N + m_N
 * is the staggered Dirac operator for the numerator and
 * (3) D = K_D + m_D
 * is the staggered Dirac operator for the denominator with
 * (4) K_N = K_D.
 * This last relation is absolutely crucial
 * 
 * ***CAUTION***: If you are to use this, you need to take the Eqn (4) condition 
 * seriously. If that is not the case, you should not be using this class.
 * 
 * We may later wish to support and extended class of operators not satisfying Eqn (4),
 * as restricting to that class of operators makes this implementation only useful 
 * for Hasenbusch mass preconditioning.
 */
public: 
  INHERIT_IMPL_TYPES(Impl);
  using Action<GaugeField>::bindLinks;

private:
  RealD _scale, _delta;

  FermionOperator<Impl>& NumOp;
  FermionOperator<Impl>& DenOp;
  
  OperatorFunction<FermionField>& DerivativeSolver;
  OperatorFunction<FermionField>& ActionSolver;

  LinkCoordinator<Impl> Links{&NumOp, &DenOp};

public:
  FermionField Phi;

public:
  FourFlavorStaggeredEvenEvenRatioPseudoFermionAction(
    FermionOperator<Impl>& _NumOp, 
	  FermionOperator<Impl>& _DenOp, 
	  OperatorFunction<FermionField>& DS,
	  OperatorFunction<FermionField>& AS
  ):NumOp(_NumOp), 
    DenOp(_DenOp), 
    DerivativeSolver(DS), 
    ActionSolver(AS), 
    Phi(_NumOp.FermionRedBlackGrid()) { 
    _scale = std::sqrt(0.5);
    _delta = NumOp.Mass()*NumOp.Mass() - DenOp.Mass()*DenOp.Mass();
    Phi.Checkerboard() = Even;
    Phi = Zero();
  };

  virtual std::string action_name() 
  { return "FourFlavorStaggeredEvenEvenRatioPseudoFermionAction"; }

  virtual std::string LogParameters() { 
    std::stringstream sstream;
    sstream << GridLogMessage 
            << "[" << action_name() << "]" 
            << " m_num: " << NumOp.Mass() 
            << " m_den: " << DenOp.Mass() 
            << std::endl;
    return sstream.str();
  }

public: // opt-in link interface: needs documentation
  void useLinkMap() { Links.useLinkMap(); }

  void bindLinks(const void* op, const LinkBinding<GaugeField>& bind)
  { Links.select(op, bind); }

  void bindLinks(
    const LinkBinding<GaugeField>& numBind,
    const LinkBinding<GaugeField>& denBind
  ) { bindLinks(NumOp.identity(), numBind); bindLinks(DenOp.identity(), denBind); }

private:
  void _refresh(GridParallelRNG& pRNG) {
    /**
     * @brief Pseudofermion heatbath
     * @author Curtis Taylor Peterson
     * @details
     * Given the action of Eqn (1) in the class documentation, one wishes to generate 
     * a pseudofermion field Phi as
     * (1) Phi = (N^dag)^{-1} D^dag eta |_{even}. 
     * We begin by following the same procedure as in the private _refresh method of 
     * Grid::FourFlavorStaggeredEvenEvenPseudoFermionAction (see docs therein), 
     * producing a Gaussian full field eta scaled by sqrt(1/2), with
     * (2) P(eta) ~ exp(-eta^dag eta).
     * From Eqn (4) of the doc directly below the class definition, 
     * (3) D = N + (m_D - m_N) I,
     * so we can express everything in terms of the operator N and the mass difference.
     * As such,
     * (4) Phi = (N^dag)^{-1} (N^dag + (m_D - m_N) I) eta |_{even}
     *         = eta_{even} + (m_D - m_N) (N^dag)^{-1} eta |_{even}.
     * But
     * (5) [(N^dag)^{-1} eta]_{even} = [N (N^dag N)^{-1} eta]_{even}
     *                               = [(N^dag N)^{-1} N eta]_{even},
     * because the staggered Dirac operator is normal. As such, we calculate
     * (6) Psi = (N eta)_{even} = m_N eta_{even} + (K_N)_{eo} eta_{odd},
     * from which, since N^dag N preserves parity,
     * (7) Phi = eta_{even} + (m_D - m_N) (N^dag N)_{ee}^{-1} Psi
     */
    FermionField EtaEven(NumOp.FermionRedBlackGrid());
    FermionField EtaOdd(NumOp.FermionRedBlackGrid());
    FermionField Psi(NumOp.FermionRedBlackGrid());
    SchurStaggeredOperator<FermionOperator<Impl>, FermionField> MdagMOp(NumOp);

    {
      FermionField eta(NumOp.FermionGrid());
      gaussian(pRNG, eta);
      eta *= _scale; // Eqn (2)
      pickCheckerboard(Even, EtaEven, eta);
      pickCheckerboard(Odd,  EtaOdd, eta);
    }

    NumOp.Meooe(EtaOdd, Psi);    // <-+- Eqn (6)
    Psi += NumOp.Mass()*EtaEven; // <-+

    Phi.Checkerboard() = Even;
    Phi = Zero();
    
    ActionSolver(MdagMOp, Psi, Phi);                   // <-+- Eqn (7)
    Phi = EtaEven + (DenOp.Mass() - NumOp.Mass())*Phi; // <-+
  }

  RealD _action() {
    /**
     * @brief Returns pseudofermion action
     * @author Curtis Taylor Peterson
     * @details
     * Because Phi lives only on the even sites and Eqn (4) of the class
     * documentation holds, the action can be expressed as
     * (1) S = Phi^dag Phi + (m_N^2 - m_D^2) Phi^dag (D^dag D)^{-1} Phi.
     * As such, we first obtain
     * (2) Psi = (D^dag D)_{ee}^{-1} Phi.
     * In what I can only describe as an unexpected bit of convenience, Grid provides
     * the innerProductNorm function, which computes Phi^dag Psi and ||Phi||^2
     * in a shared lattice pass and combines their global reductions.
     * We use the real part of the overlap, which is real for an exact solve.
     */
    RealD norm;
    ComplexD overlap;
    FermionField Psi(NumOp.FermionRedBlackGrid());
    SchurStaggeredOperator<FermionOperator<Impl>, FermionField> MdagMOp(DenOp);

    Psi = Zero();
    ActionSolver(MdagMOp, Phi, Psi); // Eqn (2)

    innerProductNorm(overlap, norm, Phi, Psi); // <-+- Eqn (1)
    return norm + _delta*overlap.real();       // <-+
  }

  template <class Derivatives>
  void _deriv(Derivatives& dSdU) {
    /**
     * @brief Wirtinger derivative of pseudofermion action
     * @author Curtis Taylor Peterson
     * @details
     * Due to Eqn (1) of the doc in the _action method of this class, the force
     * comes out to be the force of an ordinary four-flavor staggered pseudofermion
     * action, scaled by the factor m_N^2 - m_D^2. See documentation for the 
     * corresponding force at FourFlavorStaggeredEvenEvenPseudoFermionAction::_deriv
     * for details.
     * 
     * ***Grid::FourFlavorStaggeredEvenEvenPseudoFermionAction::_deriv docs
     */
    FermionField Psi(NumOp.FermionRedBlackGrid());
    FermionField Chi(NumOp.FermionRedBlackGrid());
    SchurStaggeredOperator<FermionOperator<Impl>, FermionField> MdagM(DenOp);

    Psi = Zero();
    DerivativeSolver(MdagM, Phi, Psi); // Eqn (2a)***
    DenOp.Meooe(Psi, Chi);             // Eqn (2b)***

    auto eo = [&](auto& op, auto& out, const auto& in) // Term 1***
    { op.MeoDeriv(out, in, Psi, Chi, DaggerNo); };      
    auto oe = [&](auto& op, auto& out, const auto& in) // Term 2***
    { op.MoeDeriv(out, in, Chi, Psi, DaggerYes); };   

    dSdU.accumulate(DenOp, _delta, eo); // <-+- Eqn (1)***
    dSdU.accumulate(DenOp, _delta, oe); // <-+
  }

public:
  virtual void refresh(const GaugeField& U, GridSerialRNG& sRNG, GridParallelRNG& pRNG)
  { Links.refresh(U, [&]{ _refresh(pRNG); }); }

  virtual RealD S(const GaugeField& U) 
  { return Links.action(U, [&]{ return _action(); }); }

  virtual void deriv(const GaugeField& U, GaugeField& dSdU) 
  { Links.deriv(U, dSdU, [&](auto& deriv){ _deriv(deriv); }); }

  virtual void refresh(
    ConfigurationBase<GaugeField>& U, 
    GridSerialRNG& sRNG, 
    GridParallelRNG& pRNG
  ) { Links.refresh(U, this->is_smeared, [&]{ _refresh(pRNG); }); }

  virtual RealD S(ConfigurationBase<GaugeField>& U) 
  { return Links.action(U, this->is_smeared, [&]{ return _action(); }); }

  virtual RealD Sinitial(ConfigurationBase<GaugeField>& U) { return S(U); }

  virtual void deriv(ConfigurationBase<GaugeField>& U, GaugeField& dSdU) 
  { Links.deriv(U, dSdU, this->is_smeared, [&](auto& deriv){ _deriv(deriv); }); }
};

NAMESPACE_END(Grid);

#endif // QCD_PSEUDOFERMION_FOUR_FLAVOR_STAGGERED_EVEN_EVEN_RATIO_H
