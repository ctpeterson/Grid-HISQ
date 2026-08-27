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
 * @brief Four flavor staggered even-even ratio pseudofermion action
 * @author Curtis Taylor Peterson
 * @details 
 * Implements four-flavor ratio action for even-even (i.e., reduced) staggered
 * pseudofermion fields:
 * (1) S = Phi^dagger N (Ddag D)^-1 Ndag Phi,
 * where Phi is defined only on even sites and
 * (2) N = K + m
 * is the staggered Dirac operator for the numerator and
 * (3) D = K' + m'
 * is the staggered Dirac operator for the denominator with
 * (4) Kdag K = K'dag K'.
 * This last relation is absolutely crucial - see note below. 
 * 
 * Because Phi lives only on the even sites and Eqn (4) holds, the action can be 
 * expressed as
 * (5) S = Phi^dagger Phi + (m^2 - m'^2) Phi^dagger (Ddag D)^-1 Phi.
 * This simplifies the evaluation of the force dramatically, as it allows one
 * to simply reuse the force evaluation machinery from the non-ratio case, but
 * rescaled by a factor of "m^2 - m'^2 := delta"
 * 
 * ***CAUTION***: If you are to use this, you need to take the Eqn (4) condition 
 * seriously. If that is not the case, you should not be using this class.
 */
public: INHERIT_IMPL_TYPES(Impl);

private:
  RealD _scale, _delta;
  FermionOperator<Impl>& NumOp;
  FermionOperator<Impl>& DenOp;
  OperatorFunction<FermionField>& DerivativeSolver;
  OperatorFunction<FermionField>& ActionSolver;

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

private:
  void _refresh(GridParallelRNG& pRNG) {
    FermionField eta(NumOp.FermionGrid());
    FermionField b(NumOp.FermionGrid());
    FermionField be(NumOp.FermionRedBlackGrid());
    FermionField bo(NumOp.FermionRedBlackGrid());
    FermionField src(NumOp.FermionRedBlackGrid());
    FermionField tmp(NumOp.FermionRedBlackGrid());
    SchurStaggeredOperator<FermionOperator<Impl>, FermionField> MdagMOp(NumOp);

    Phi.Checkerboard() = Even;
    Phi = Zero();

    gaussian(pRNG, eta);
    eta *= _scale;
    DenOp.Mdag(eta, b);

    pickCheckerboard(Even, be, b);
    pickCheckerboard(Odd,  bo, b);

    NumOp.Mooee(be, src);
    NumOp.Meooe(bo, tmp);
    src += tmp;

    ActionSolver(MdagMOp, src, Phi);
  }

  RealD _action() {
    FermionField Psi(NumOp.FermionRedBlackGrid());
    FermionField Chi(NumOp.FermionRedBlackGrid());
    SchurStaggeredOperator<FermionOperator<Impl>, FermionField> MdagMOp(DenOp);
    RealD mass2 = DenOp.Mass()*DenOp.Mass();

    Psi = Zero();
    ActionSolver(MdagMOp, Phi, Psi);
    DenOp.Meooe(Psi, Chi);

    return norm2(Phi) + _delta*(mass2*norm2(Psi) + norm2(Chi));
  }

  void _deriv(GaugeField& dSdU) {
    GaugeField ForceE(NumOp.GaugeRedBlackGrid());
    GaugeField ForceO(NumOp.GaugeRedBlackGrid());
    FermionField Psi(NumOp.FermionRedBlackGrid());
    FermionField Chi(NumOp.FermionRedBlackGrid());
    SchurStaggeredOperator<FermionOperator<Impl>, FermionField> MdagM(DenOp);

    ForceE.Checkerboard() = Even;
    ForceO.Checkerboard() = Odd;

    Psi = Zero();
    DerivativeSolver(MdagM, Phi, Psi);
    DenOp.Meooe(Psi, Chi);

    DenOp.MeoDeriv(ForceE, Psi, Chi, DaggerNo);
    DenOp.MoeDeriv(ForceO, Chi, Psi, DaggerYes);

    setCheckerboard(dSdU, ForceE);
    setCheckerboard(dSdU, ForceO);
    dSdU *= -_delta;
  }

public:
  virtual void refresh(const GaugeField& U, GridSerialRNG& sRNG, GridParallelRNG& pRNG) 
  { NumOp.ImportGauge(U); DenOp.ImportGauge(U); _refresh(pRNG); }

  virtual RealD S(const GaugeField& U) { 
    NumOp.ImportGauge(U); 
    DenOp.ImportGauge(U); 
    return _action(); 
  }

  virtual void deriv(const GaugeField& U, GaugeField& dSdU) 
  { NumOp.ImportGauge(U); DenOp.ImportGauge(U); _deriv(dSdU); }

  virtual void refresh(
    ConfigurationBase<GaugeField>& U, 
    GridSerialRNG& sRNG, 
    GridParallelRNG& pRNG
  ) { refresh(U.get_U(this->is_smeared), sRNG, pRNG); }

  virtual RealD S(ConfigurationBase<GaugeField>& U) 
  { return S(U.get_U(this->is_smeared)); }

  virtual RealD Sinitial(ConfigurationBase<GaugeField>& U) { return _action(); }

  virtual void deriv(ConfigurationBase<GaugeField>& U, GaugeField& dSdU) { 
    deriv(U.get_U(this->is_smeared), dSdU); 
    if (this->is_smeared) { U.smeared_force(dSdU); } 
  }
};

NAMESPACE_END(Grid);

#endif // QCD_PSEUDOFERMION_FOUR_FLAVOR_STAGGERED_EVEN_EVEN_RATIO_H