/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/qcd/action/pseudofermion/StaggeredEvenEvenRational.h

Copyright (C) 2015

Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU AnyFlavor Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU AnyFlavor Public License for more details.

You should have received a copy of the GNU AnyFlavor Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

See the full license in the file "LICENSE" in the top level distribution
directory
*************************************************************************************/
/*  END LEGAL */
/**
  @file StaggeredEvenEvenRational.h
  @author Curtis Taylor Peterson
*/

#pragma once
#include <Grid/Grid.h>

#ifndef QCD_PSEUDOFERMION_STAGGERED_EVEN_EVEN_RATIONAL_H
#define QCD_PSEUDOFERMION_STAGGERED_EVEN_EVEN_RATIONAL_H

NAMESPACE_BEGIN(Grid);

template<class Impl>
class StaggeredEvenEvenRational: public Action<typename Impl::GaugeField> {
/**
 * @brief Staggered even-even rational (rooted) pseudofermion action
 * @author Curtis Taylor Peterson
 * @details
 * Implements staggered even-even rational (rooted) staggered pseudofermion action:
 * (1) S = Phi^dagger (M^dagger M)^(-Nf/4) Phi,
 * where Phi is defined on even sites only. Almost everything from the "four flavor"
 * class can be reused with minor modifications for the rooted case.
 */
public: INHERIT_IMPL_TYPES(Impl);

private:
  RealD _scale;
  StaggeredRationalActionParams _params;
  FermionOperator<Impl>& FermOp;

  MultiShiftFunction OneEighthAction;
  MultiShiftFunction NegOneQuarterAction;
  MultiShiftFunction NegOneQuarterDeriv;

public:
  FermionField Phi;

private:
  void _generateRemezApproximation(
    MultiShiftFunction& approx,
    AlgRemez& remez,
    int invPow,
    bool action,
    bool inverse
  ) {
    const int degree = action ? _params.action_degree : _params.md_degree;
    const RealD tolerance = action ? _params.action_tolerance : _params.md_tolerance;
    std::cout << GridLogMessage 
              << "Generating degree " << degree << " approximation for x^("
              << (inverse ? "-" : "") << _params.nf << "/" << invPow << ")"
              << std::endl;
    double error = remez.generateApprox(degree, _params.nf, invPow);
    if (error > tolerance) {
      std::cout << GridLogMessage 
                << "WARNING: Remez approximation has a larger error " 
                << error << " than the CG tolerance " << tolerance 
                << "! Try increasing the number of poles" << std::endl;
    }
    approx.Init(remez, tolerance, inverse);
  }

public:
  StaggeredEvenEvenRational(
    FermionOperator<Impl>& Op,
    const StaggeredRationalActionParams& params
  ):FermOp(Op),
    Phi(Op.FermionRedBlackGrid()),
    _params(params) {
    GRID_ASSERT(_params.lo > 0.0 && "Lower bound of Remez approx must be positive");
    GRID_ASSERT(_params.hi > _params.lo && "Upper bound of Remez approx must be greater than lower bound");

    AlgRemez remez(_params.lo, _params.hi, _params.precision);

    _scale = std::sqrt(0.5);
    Phi.Checkerboard() = Even; 
    Phi = Zero();

    _generateRemezApproximation(OneEighthAction, remez, 8, true, false);
    _generateRemezApproximation(NegOneQuarterAction, remez, 4, true, true);
    _generateRemezApproximation(NegOneQuarterDeriv, remez, 4, false, true);
    
    std::cout << GridLogMessage 
              << action_name() 
              << " initialize: complete" 
              << std::endl;
  }

public:
  virtual std::string action_name() 
  { return "StaggeredEvenEvenRationalPseudoFermionAction"; }

  virtual std::string LogParameters() {
    std::stringstream sstream;
    sstream << GridLogMessage 
            << "[" << action_name() << "] mass: " << FermOp.Mass()
            << std::endl;
    sstream << GridLogMessage << "[" << action_name() 
            << "] Power              : " << _params.nf << "/4" << std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] Low                :"
            << _params.lo <<  std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] High               :"
            << _params.hi <<  std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] Max iterations     :"
            << _params.MaxIter <<  std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] Tolerance (Action) :"
            << _params.action_tolerance <<  std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] Degree (Action)    :"
            << _params.action_degree <<  std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] Tolerance (MD)     :"
            << _params.md_tolerance <<  std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] Degree (MD)        :"
            << _params.md_degree <<  std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] Precision          :"
            << _params.precision <<  std::endl;
    return sstream.str();
  }

private:
  // pole solutions only
  void _multiShiftSolve(
    const MultiShiftFunction& approx,
    const FermionField& in,
    std::vector<FermionField>& outs
  ) {
    SchurStaggeredOperator<FermionOperator<Impl>, FermionField> MdagM(FermOp);
    ConjugateGradientMultiShift<FermionField> solver(_params.MaxIter, approx);
    for (int i = 0; i < outs.size(); ++i) 
    { outs[i].Checkerboard() = in.Checkerboard(); }
    solver(MdagM, in, outs);
  }

  // summed rational function
  void _multiShiftSolve(
    const MultiShiftFunction& approx,
    const FermionField& in,
    FermionField& out
  ) {
    SchurStaggeredOperator<FermionOperator<Impl>, FermionField> MdagM(FermOp);
    ConjugateGradientMultiShift<FermionField> solver(_params.MaxIter, approx);
    std::vector<FermionField> outs(approx.poles.size(), FermOp.FermionRedBlackGrid());
    for (int i = 0; i < outs.size(); ++i) 
    { outs[i].Checkerboard() = in.Checkerboard(); }
    out.Checkerboard() = in.Checkerboard();
    solver(MdagM, in, outs, out);
  }

private:
  void _refresh(GridParallelRNG& pRNG) {
    FermionField eta(FermOp.FermionGrid());
    FermionField etae(FermOp.FermionRedBlackGrid());
    gaussian(pRNG, eta);
    eta *= _scale;
    pickCheckerboard(Even, etae, eta);
    _multiShiftSolve(OneEighthAction, etae, Phi);
  }

  RealD _action() {
    FermionField Psi(FermOp.FermionRedBlackGrid());
    _multiShiftSolve(NegOneQuarterAction, Phi, Psi);
    return innerProduct(Phi, Psi).real();
  }

  void _deriv(GaugeField& dSdU) {
    const int numPoles = NegOneQuarterDeriv.poles.size();
    std::vector<FermionField> Psis(numPoles, FermOp.FermionRedBlackGrid());
    GaugeField Force(FermOp.GaugeGrid());
    GaugeField ForceE(FermOp.GaugeRedBlackGrid());
    GaugeField ForceO(FermOp.GaugeRedBlackGrid());
    FermionField Chi(FermOp.FermionRedBlackGrid());

    ForceE.Checkerboard() = Even;
    ForceO.Checkerboard() = Odd;

    _multiShiftSolve(NegOneQuarterDeriv, Phi, Psis);
    
    dSdU = Zero();
    for (int i = 0; i < numPoles; ++i) {
      FermOp.Meooe(Psis[i], Chi);
      FermOp.MeoDeriv(ForceE, Psis[i], Chi, DaggerNo);
      FermOp.MoeDeriv(ForceO, Chi, Psis[i], DaggerYes);

      setCheckerboard(Force, ForceE);
      setCheckerboard(Force, ForceO);

      dSdU -= NegOneQuarterDeriv.residues[i]*Force;
    }
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

#endif // QCD_PSEUDOFERMION_STAGGERED_EVEN_EVEN_RATIONAL_H