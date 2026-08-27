/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/qcd/action/pseudofermion/StaggeredEvenEvenRatioRational.h

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
  @file StaggeredEvenEvenRatioRational.h
  @author Curtis Taylor Peterson
*/

#pragma once
#include <Grid/Grid.h>

#ifndef QCD_PSEUDOFERMION_STAGGERED_EVEN_EVEN_RATIO_RATIONAL_H
#define QCD_PSEUDOFERMION_STAGGERED_EVEN_EVEN_RATIO_RATIONAL_H

NAMESPACE_BEGIN(Grid);

template<class Impl>
class StaggeredEvenEvenRatioRational: public Action<typename Impl::GaugeField> {
/**
 * @brief Staggered even-even ratio rational (rooted) pseudofermion action
 * @author Curtis Taylor Peterson
 * @details
 * Implements the staggered even-even ratio rational (rooted) pseudofermion action
 * (1) S = Phi^dagger (Ndag N)^(Nf/8) (Ddag D)^(-Nf/4) (Ndag N)^(Nf/8) Phi,
 * where Phi lives on even sites only.
 */
public: INHERIT_IMPL_TYPES(Impl);

private:
  RealD _scale;
  StaggeredRationalActionParams _numParams;
  StaggeredRationalActionParams _denParams;
  FermionOperator<Impl>& NumOp;
  FermionOperator<Impl>& DenOp;

  MultiShiftFunction OneEighthNumAction;     // <---+--- action
  MultiShiftFunction OneEighthDenAction;     //     |
  MultiShiftFunction NegOneEighthNumAction;  //     |
  MultiShiftFunction NegOneQuarterDenAction; // <---+
  MultiShiftFunction OneEighthNumDeriv;      // <---+--- force
  MultiShiftFunction NegOneQuarterDenDeriv;  // <---+

private:
  static constexpr bool Num = true;
  static constexpr bool Den = false;

public:
  FermionField Phi;

private:
  void _generateRemezApproximation(
    MultiShiftFunction& approx,
    AlgRemez& remez,
    const StaggeredRationalActionParams& params,
    int invPow,
    bool action,
    bool inverse
  ) {
    const int degree = action ? params.action_degree : params.md_degree;
    const RealD tolerance = action ? params.action_tolerance : params.md_tolerance;
    std::cout << GridLogMessage 
              << "Generating degree " << degree << " approximation for x^("
              << (inverse ? "-" : "") << params.nf << "/" << invPow << ")"
              << std::endl;
    double error = remez.generateApprox(degree, params.nf, invPow);
    if (error > tolerance) {
      std::cout << GridLogMessage 
                << "WARNING: Remez approximation has a larger error " 
                << error << " than the CG tolerance " << tolerance 
                << "! Try increasing the number of poles" << std::endl;
    }
    approx.Init(remez, tolerance, inverse);
  }

public:
  StaggeredEvenEvenRatioRational(
    FermionOperator<Impl>& _numOp,
    FermionOperator<Impl>& _denOp,
    const StaggeredRationalActionParams& numParams,
    const StaggeredRationalActionParams& denParams
  ):_numParams(numParams),
    _denParams(denParams),
    NumOp(_numOp),
    DenOp(_denOp),
    Phi(_numOp.FermionRedBlackGrid()) {
    GRID_ASSERT(_numParams.nf == _denParams.nf && "Nf must agree between num and den");
    GRID_ASSERT(_numParams.lo > 0.0 && "Num lower bound of Remez approx must be positive");
    GRID_ASSERT(_denParams.lo > 0.0 && "Den lower bound of Remez approx must be positive");
    GRID_ASSERT(_numParams.hi > _numParams.lo && "Num upper bound of Remez approx must be greater than lower bound");
    GRID_ASSERT(_denParams.hi > _denParams.lo && "Den upper bound of Remez approx must be greater than lower bound");
    conformable(_numOp.FermionRedBlackGrid(), _denOp.FermionRedBlackGrid());

    AlgRemez remezNum(_numParams.lo, _numParams.hi, _numParams.precision);
    AlgRemez remezDen(_denParams.lo, _denParams.hi, _denParams.precision);

    _scale = std::sqrt(0.5);
    Phi.Checkerboard() = Even;
    Phi = Zero();

    _generateRemezApproximation(OneEighthNumAction, remezNum, _numParams, 8, true, false);
    _generateRemezApproximation(OneEighthDenAction, remezDen, _denParams, 8, true, false);
    _generateRemezApproximation(NegOneEighthNumAction, remezNum, _numParams, 8, true, true);
    _generateRemezApproximation(NegOneQuarterDenAction, remezDen, _denParams, 4, true, true);
    _generateRemezApproximation(OneEighthNumDeriv, remezNum, _numParams, 8, false, false);
    _generateRemezApproximation(NegOneQuarterDenDeriv, remezDen, _denParams, 4, false, true);

    std::cout << GridLogMessage 
              << action_name() 
              << " initialize: complete" 
              << std::endl;
  }

public:
  virtual std::string action_name()
  { return "StaggeredEvenEvenRatioRationalPseudoFermionAction"; }

  virtual std::string LogParameters() { // !!! CHECK, CHECK, CHECK !!!
    std::stringstream sstream;
    sstream << GridLogMessage
            << "[" << action_name() << "] m_num: " << NumOp.Mass()
            << " m_den: " << DenOp.Mass() << std::endl;
    sstream << GridLogMessage << "[" << action_name()
            << "] Power                  : " << _numParams.nf << "/4" << std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] Low  (num/den)         :"
            << _numParams.lo << " " << _denParams.lo << std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] High (num/den)         :"
            << _numParams.hi << " " << _denParams.hi << std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] Max iterations         :"
            << _numParams.MaxIter << " " << _denParams.MaxIter << std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] Tolerance (Action)     :"
            << _numParams.action_tolerance << " " << _denParams.action_tolerance << std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] Degree (Action)        :"
            << _numParams.action_degree << " " << _denParams.action_degree << std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] Tolerance (MD)         :"
            << _numParams.md_tolerance << " " << _denParams.md_tolerance << std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] Degree (MD)            :"
            << _numParams.md_degree << " " << _denParams.md_degree << std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] Precision              :"
            << _numParams.precision << " " << _denParams.precision << std::endl;
    return sstream.str();
  }

private:
  void _multiShiftSolve(
    bool numerator,
    const MultiShiftFunction& approx,
    const FermionField& in,
    std::vector<FermionField>& outs,
    FermionField& out
  ) {
    const Integer maxIter = numerator ? _numParams.MaxIter : _denParams.MaxIter;
    SchurStaggeredOperator<FermionOperator<Impl>, FermionField> MdagM(numerator ? NumOp : DenOp);
    ConjugateGradientMultiShift<FermionField> solver(maxIter, approx);
    for (int i = 0; i < outs.size(); ++i) 
    { outs[i].Checkerboard() = in.Checkerboard(); }
    out.Checkerboard() = in.Checkerboard();
    solver(MdagM, in, outs, out);
  }

  void _multiShiftSolve(
    bool numerator,
    const MultiShiftFunction& approx,
    const FermionField& in,
    std::vector<FermionField>& outs
  ) {
    FermionField out(NumOp.FermionRedBlackGrid());
    _multiShiftSolve(numerator, approx, in, outs, out);
  }

  void _multiShiftSolve(
    bool numerator,
    const MultiShiftFunction& approx,
    const FermionField& in,
    FermionField& out
  ) { 
    std::vector<FermionField> outs(approx.poles.size(), NumOp.FermionRedBlackGrid());
    _multiShiftSolve(numerator, approx, in, outs, out);
  }

private:
  void _refresh(GridParallelRNG& pRNG) { 
    FermionField eta(NumOp.FermionGrid());
    FermionField etae(NumOp.FermionRedBlackGrid());
    FermionField tmp(NumOp.FermionRedBlackGrid());

    gaussian(pRNG, eta);
    eta *= _scale;
    pickCheckerboard(Even, etae, eta);

    _multiShiftSolve(Den, OneEighthDenAction, etae, tmp);
    _multiShiftSolve(Num, NegOneEighthNumAction, tmp, Phi);
  }

  RealD _action() {
    FermionField X(NumOp.FermionRedBlackGrid());
    FermionField Y(NumOp.FermionRedBlackGrid());
    _multiShiftSolve(Num, OneEighthNumAction, Phi, X);
    _multiShiftSolve(Den, NegOneQuarterDenAction, X, Y);
    return innerProduct(X, Y).real();
  }

  void _deriv(GaugeField& dSdU) {
    const int numPoles = OneEighthNumDeriv.poles.size();
    const int denPoles = NegOneQuarterDenDeriv.poles.size();

    std::vector<FermionField> Xs(numPoles, NumOp.FermionRedBlackGrid());
    std::vector<FermionField> Ys(denPoles, NumOp.FermionRedBlackGrid());
    std::vector<FermionField> Zs(numPoles, NumOp.FermionRedBlackGrid());
    
    FermionField X(NumOp.FermionRedBlackGrid());
    FermionField Y(NumOp.FermionRedBlackGrid());
    FermionField ChiL(NumOp.FermionRedBlackGrid());
    FermionField ChiR(NumOp.FermionRedBlackGrid());

    GaugeField Force(NumOp.GaugeGrid());
    GaugeField ForceE(NumOp.GaugeRedBlackGrid());
    GaugeField ForceO(NumOp.GaugeRedBlackGrid());

    ForceE.Checkerboard() = Even;
    ForceO.Checkerboard() = Odd;

    _multiShiftSolve(Num, OneEighthNumDeriv, Phi, Xs, X);
    _multiShiftSolve(Den, NegOneQuarterDenDeriv, X, Ys, Y);
    _multiShiftSolve(Num, OneEighthNumDeriv, Y, Zs);

    dSdU = Zero();
 
    for (int k = 0; k < denPoles; ++k) {
      DenOp.Meooe(Ys[k], ChiL);

      DenOp.MeoDeriv(ForceE, Ys[k], ChiL, DaggerNo);
      DenOp.MoeDeriv(ForceO, ChiL, Ys[k], DaggerYes);

      setCheckerboard(Force, ForceE);
      setCheckerboard(Force, ForceO);

      dSdU -= NegOneQuarterDenDeriv.residues[k]*Force;
    }

    for (int j = 0; j < numPoles; ++j) {
      const RealD rj = OneEighthNumDeriv.residues[j];

      NumOp.Meooe(Zs[j], ChiL);
      NumOp.Meooe(Xs[j], ChiR);

      NumOp.MeoDeriv(ForceE, Zs[j], ChiR, DaggerNo);
      NumOp.MoeDeriv(ForceO, ChiL, Xs[j], DaggerYes);
      
      setCheckerboard(Force, ForceE);
      setCheckerboard(Force, ForceO);

      dSdU -= rj*Force;

      NumOp.MeoDeriv(ForceE, Xs[j], ChiL, DaggerNo);
      NumOp.MoeDeriv(ForceO, ChiR, Zs[j], DaggerYes);
      
      setCheckerboard(Force, ForceE);
      setCheckerboard(Force, ForceO);

      dSdU -= rj*Force;
    }
  }

public:
  virtual void refresh(const GaugeField& U, GridSerialRNG& sRNG, GridParallelRNG& pRNG)
  { NumOp.ImportGauge(U); DenOp.ImportGauge(U); _refresh(pRNG); }

  virtual RealD S(const GaugeField& U) 
  { NumOp.ImportGauge(U); DenOp.ImportGauge(U); return _action(); }

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

#endif // QCD_PSEUDOFERMION_STAGGERED_EVEN_EVEN_RATIO_RATIONAL_H