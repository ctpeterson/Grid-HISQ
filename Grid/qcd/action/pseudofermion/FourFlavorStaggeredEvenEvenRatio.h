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

#ifndef QCD_PSEUDOFERMION_STAGGERED_EE_RATIO_H
#define QCD_PSEUDOFERMION_STAGGERED_EE_RATIO_H

NAMESPACE_BEGIN(Grid);

template <class Impl>
class FourFlavorStaggeredEvenEvenRatioPseudoFermionAction: public Action<typename Impl::GaugeField> 
{
public: INHERIT_IMPL_TYPES(Impl);

private:
  RealD _scale;
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
    Phi.Checkerboard() = Even;
    Phi = Zero();
  };

  virtual std::string action_name() 
  { return "FourFlavorStaggeredEvenEvenRatioPseudoFermionAction"; }

  virtual std::string LogParameters() { 
    std::stringstream sstream;
    sstream << GridLogMessage << "["<<action_name()<<"] has no parameters" << std::endl;
    return sstream.str();
  }

private:
  void _refresh(GridParallelRNG& pRNG) {
    FermionField eta(NumOp.FermionGrid());
    FermionField X(NumOp.FermionGrid());
    FermionField Y(NumOp.FermionGrid());
    FermionField phi(NumOp.FermionGrid());
    MdagMLinearOperator<FermionOperator<Impl>, FermionField> MdagMOp(NumOp);

    X = Zero();
    Y = Zero();
    phi = Zero();

    gaussian(pRNG, eta);
    eta = _scale*eta;
    DenOp.Mdag(eta, X);
    ActionSolver(MdagMOp, X, Y);
    NumOp.M(Y, phi);

    pickCheckerboard(Even, Phi, phi);
  }

  RealD _action() {
    FermionField phi(NumOp.FermionGrid());
    FermionField X(NumOp.FermionGrid());
    FermionField Y(NumOp.FermionGrid());
    MdagMLinearOperator<FermionOperator<Impl>, FermionField> MdagMOp(DenOp);

    phi = Zero();
    X = Zero();
    Y = Zero();

    setCheckerboard(phi, Phi);
    NumOp.Mdag(phi, Y);
    ActionSolver(MdagMOp, Y, X);
    DenOp.M(X, Y);

    RealD action = norm2(Y);
    return action;
  }

  void _deriv(GaugeField& dSdU) {
    GaugeField force(NumOp.GaugeGrid());
    FermionField phi(NumOp.FermionGrid());
    FermionField X(NumOp.FermionGrid());
    FermionField Y(NumOp.FermionGrid());
    MdagMLinearOperator<FermionOperator<Impl>, FermionField> MdagMOp(DenOp);

    phi = Zero();
    X = Zero();
    Y = Zero();

    setCheckerboard(phi, Phi);
    NumOp.Mdag(phi, Y);
    DerivativeSolver(MdagMOp, Y, X);    
    DenOp.M(X, Y); 
    NumOp.MDeriv(force, X, phi, DaggerYes); 
    dSdU = force;
    NumOp.MDeriv(force, phi, X, DaggerNo); 
    dSdU += force;
    DenOp.MDeriv(force, Y, X, DaggerNo); 
    dSdU -= force;
    DenOp.MDeriv(force, X, Y, DaggerYes); 
    dSdU -= force;
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
  ) { refresh(U.get_SmearedU(), sRNG, pRNG); }

  virtual RealD S(ConfigurationBase<GaugeField>& U) { return S(U.get_SmearedU()); }

  virtual RealD Sinitial(ConfigurationBase<GaugeField>& U) { return _action(); }

  virtual void deriv(ConfigurationBase<GaugeField>& U, GaugeField& dSdU)
  { deriv(U.get_SmearedU(), dSdU); if (this->is_smeared) { U.smeared_force(dSdU); } }

};

NAMESPACE_END(Grid);

#endif