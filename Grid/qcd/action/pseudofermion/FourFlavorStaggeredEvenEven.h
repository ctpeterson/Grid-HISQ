/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/qcd/action/pseudofermion/FourFlavorStaggeredEvenEven.h

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
  @file FourFlavorStaggeredEvenEven.h
  @author Curtis Taylor Peterson
*/

#pragma once
#include <Grid/Grid.h>

#ifndef QCD_PSEUDOFERMION_STAGGERED_EE_H
#define QCD_PSEUDOFERMION_STAGGERED_EE_H

NAMESPACE_BEGIN(Grid);

template <class Impl>
class FourFlavorStaggeredEvenEvenPseudoFermionAction: public Action<typename Impl::GaugeField> {
public: INHERIT_IMPL_TYPES(Impl);

private:
  RealD _scale;
  FermionOperator<Impl>& FermOp;
  OperatorFunction<FermionField>& DerivativeSolver;
  OperatorFunction<FermionField>& ActionSolver;

public:
  FermionField Phi;

public:
  FourFlavorStaggeredEvenEvenPseudoFermionAction(
    FermionOperator<Impl>& Op,
    OperatorFunction<FermionField>& DS,
    OperatorFunction<FermionField>& AS
  ):DerivativeSolver(DS),
    ActionSolver(AS),
    FermOp(Op),
    Phi(Op.FermionRedBlackGrid()) {
    _scale = std::sqrt(0.5); 
    Phi.Checkerboard() = Even; 
    Phi = Zero();
  }

  virtual std::string action_name() 
  { return "FourFlavorStaggeredEvenEvenPseudoFermionAction"; }

  virtual std::string LogParameters() { 
    std::stringstream sstream;
    sstream << GridLogMessage 
            << "[" <<action_name() << "] has no parameters" 
            << " mass: " << FermOp.Mass()
            << std::endl;
    return sstream.str();
  }

private:
  void _refresh(GridParallelRNG& pRNG) {
    FermionField eta(FermOp.FermionGrid()), phi(FermOp.FermionGrid());
    gaussian(pRNG, eta); 
    eta = _scale*eta; 
    FermOp.Mdag(eta, phi); 
    pickCheckerboard(Even, Phi, phi); 
  }

  RealD _action() {
    FermionField Psi(FermOp.FermionRedBlackGrid());
    FermionField Chi(FermOp.FermionRedBlackGrid());
    SchurStaggeredOperator<FermionOperator<Impl>, FermionField> MdagMOp(FermOp);
    RealD mass2 = FermOp.Mass()*FermOp.Mass();

    Psi = Zero();
    ActionSolver(MdagMOp, Phi, Psi);
    FermOp.Meooe(Psi, Chi);
    return mass2*norm2(Psi) + norm2(Chi);
  }

  void _deriv(GaugeField& dSdU) {
    GaugeField ForceE(FermOp.GaugeRedBlackGrid());
    GaugeField ForceO(FermOp.GaugeRedBlackGrid());
    FermionField Psi(FermOp.FermionRedBlackGrid());
    FermionField Chi(FermOp.FermionRedBlackGrid());
    SchurStaggeredOperator<FermionOperator<Impl>, FermionField> MdagM(FermOp);

    ForceE.Checkerboard() = Even;
    ForceO.Checkerboard() = Odd;

    Psi = Zero();
    DerivativeSolver(MdagM, Phi, Psi);
    FermOp.Meooe(Psi, Chi);

    FermOp.MeoDeriv(ForceE, Psi, Chi, DaggerNo);
    FermOp.MoeDeriv(ForceO, Chi, Psi, DaggerYes);

    setCheckerboard(dSdU, ForceE);
    setCheckerboard(dSdU, ForceO);
    dSdU = -dSdU;
  }

public:
  virtual void refresh(const GaugeField& U, GridSerialRNG& sRNG, GridParallelRNG& pRNG)
  { FermOp.ImportGauge(U); _refresh(pRNG); }

  virtual RealD S(const GaugeField& U) { FermOp.ImportGauge(U); return _action(); }

  virtual void deriv(const GaugeField& U, GaugeField& dSdU) 
  { FermOp.ImportGauge(U); _deriv(dSdU); }

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

#endif