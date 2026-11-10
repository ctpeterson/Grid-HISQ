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
 * @file StaggeredEvenEvenRatioRational.h
 * @author Curtis Taylor Peterson
 */

#pragma once

#include <Grid/Grid.h>

#ifndef QCD_PSEUDOFERMION_STAGGERED_EVEN_EVEN_RATIO_RATIONAL_H
#define QCD_PSEUDOFERMION_STAGGERED_EVEN_EVEN_RATIO_RATIONAL_H

NAMESPACE_BEGIN(Grid);

template<class Impl>
class StaggeredEvenEvenRatioRational: public Action<typename Impl::GaugeField> {
/**
 * @class Grid::StaggeredEvenEvenRatioRational
 * @brief Staggered even-even ratio rational (rooted) pseudofermion action
 * @author Curtis Taylor Peterson
 * @details
 * Implements the staggered even-even (i.e., reduced) ratio rational (i.e., rooted) 
 * pseudofermion action
 * (1) S = Phi^dagger (Ndag N)^(Nf/8) (Ddag D)^(-Nf/4) (Ndag N)^(Nf/8) Phi,
 * where Phi lives on even sites only.
 */
public: 
  INHERIT_IMPL_TYPES(Impl);

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

  ActionContract<GaugeField> _numContract, _denContract;

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

  StaggeredEvenEvenRatioRational(
    FermionOperator<Impl>& _numOp,
    FermionOperator<Impl>& _denOp,
    const StaggeredRationalActionParams& numParams,
    const StaggeredRationalActionParams& denParams,
    const MultiShiftFunction& numHeatBath,
    const MultiShiftFunction& numAction,
    const MultiShiftFunction& numDeriv,
    const MultiShiftFunction& denHeatBath,
    const MultiShiftFunction& denAction,
    const MultiShiftFunction& denDeriv
  ):_numParams(numParams),
    _denParams(denParams),
    NumOp(_numOp),
    DenOp(_denOp),
    OneEighthNumAction(numAction),
    OneEighthDenAction(denHeatBath),
    NegOneEighthNumAction(numHeatBath),
    NegOneQuarterDenAction(denAction),
    OneEighthNumDeriv(numDeriv),
    NegOneQuarterDenDeriv(denDeriv),
    Phi(_numOp.FermionRedBlackGrid()) {
    conformable(_numOp.FermionRedBlackGrid(), _denOp.FermionRedBlackGrid());

    _scale = std::sqrt(0.5);
    Phi.Checkerboard() = Even;
    Phi = Zero();

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

public: // opt-in link interface: needs documentation
  void signContract(Primals<GaugeField> numPrimals, Primals<GaugeField> denPrimals) {
    _numContract = Action<GaugeField>::signContract(NumOp, numPrimals);
    _denContract = Action<GaugeField>::signContract(DenOp, denPrimals);
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
    /**
     * @brief Pseudofermion heatbath
     * @author Curtis Taylor Peterson
     * @details
     * Given the action of Eqn (1) in the class documentation, one wishes to generate 
     * a pseudofermion field Phi as
     * (1) Phi = (N^dag N)^{-Nf/8} (D^dag D)^{Nf/8} Eta.
     * We start off by producing a Gaussian field Eta on the even sites,
     * scaled by sqrt(1/2), with
     * (2) P(Eta) ~ exp(-Eta^dag Eta),
     * then obtain Phi using two multi-shift solves for the rational approximations:
     * first the positive denominator power, then the negative numerator power.
     */
    FermionField EtaEven(NumOp.FermionRedBlackGrid());
    FermionField tmp(NumOp.FermionRedBlackGrid());

    EtaEven.Checkerboard() = Even;
    gaussian(pRNG, EtaEven);
    EtaEven *= _scale; // Eqn (2)

    _multiShiftSolve(Den, OneEighthDenAction, EtaEven, tmp); // <-+- Eqn (1)
    _multiShiftSolve(Num, NegOneEighthNumAction, tmp, Phi);  // <-+
  }

  RealD _action() {
    /**
     * @brief Pseudofermion action
     * @author Curtis Taylor Peterson
     * @details
     * Computes the pseudofermion action S defined in Eqn (1) of the class
     * documentation. The calculation is straightforward. Define
     * (1a) X = (N^dag N)^{Nf/8} Phi
     * (1b) Y = (D^dag D)^{-Nf/4} X
     * and calculate S as
     * (2) S = X^dagger Y.
     * The powers are evaluated on the even checkerboard using the action rational
     * approximations. We take the real part of the inner product numerically.
     */
    FermionField X(NumOp.FermionRedBlackGrid());
    FermionField Y(NumOp.FermionRedBlackGrid());

    _multiShiftSolve(Num, OneEighthNumAction, Phi, X);   // Eqn (1a) 
    _multiShiftSolve(Den, NegOneQuarterDenAction, X, Y); // Eqn (1b)
    
    return innerProduct(X, Y).real(); // Eqn (2)
  }

  template <class Derivative>
  void _deriv(Derivative& dSdU) {
    /**
     * @brief Wirtinger derivative of pseudofermion action
     * @author Curtis Taylor Peterson
     * @details
     * Calculates the Wirtinger derivative of the pseudofermion action defined in
     * Eqn (1) of the docs directly below the class definition. As in docs under 
     * _action, we start by defining
     * (1a) X = (N^dag N)^{Nf/8} Phi,
     * (1b) Y = (D^dag D)^{-Nf/4} X,
     * so that S = X^dag Y. Writing the numerator and denominator poles as betaN_j 
     * and betaD_k, the multi-shift solves give X and Y, along with
     * (2a) X_j = (N^dag N + betaN_j)^{-1} Phi,
     * (2b) Y_k = (D^dag D + betaD_k)^{-1} X,
     * (2c) Z_j = (N^dag N + betaN_j)^{-1} Y.
     *
     * Applying Eqn (1)*** to each inverse while holding Phi fixed, one has
     * (3) -dS = sum_{d} r_d Y_d^dag d(D^dag D) Y_d
     *         + sum_{n} r_n [ Z_n^dag d(N^dag N) X_n
     *                       + X_n^dag d(N^dag N) Z_n ],
     * where r_d and r_n are the denominator and numerator residues, respectively.
     * The first sum is the force of an ordinary four-flavor staggered pseudofermion
     * action for each Ys_d, scaled by r_d. The numerator appears on both sides of
     * the action, so differentiating it gives two terms for each pole. 
     *
     * As described in FourFlavorStaggeredEvenEvenPseudoFermionAction::_deriv,
     * we can work fully within the checkerboards. X_n, Y_d, and Z_n live
     * on the even checkerboard. Applying Meooe gives the odd fields
     * (4a) ChiL_d = D_{oe} Ys_d,                      (denominator)
     * (4b) ChiL_n = N_{oe} Z_n,                        (numerator)
     * (4c) ChiR_n = N_{oe} X_n.                        (numerator)
     * As such, the force calculation follows the same procedure as in the
     * four-flavor class, with the fields and residues given above. See the
     * documentation for the corresponding force at
     * FourFlavorStaggeredEvenEvenPseudoFermionAction::_deriv for details.
     *
     * As in StaggeredEvenEvenRational::_deriv, a checkerboarded Term 1*** (left field
     * on the even checkerboard) and Term 2*** (left field on the odd checkerboard,
     * DaggerYes) combine into the derivative of a single full-grid bilinear. Because
     * the mass term drops out of the derivative and M^dag = -M(-m) for staggered
     * operators,
     * (5) d[ L^dag M R ] = d[ A^dag M E ] + d[ B^dag M^dag C ]
     * for
     * (6) L = A (even) + B (odd),  R = -C (even) + E (odd).
     * Each denominator pole gives one such bilinear,
     * (7) A = C = Y_d,  B = E = ChiL_d,
     * and each numerator pole gives two,
     * (8a) A = Z_n, B = ChiL_n, C = X_n, E = ChiR_n,
     * (8b) A = X_n, B = ChiR_n, C = Z_n, E = ChiL_n,
     * weighted by the residues of Eqn (3). The bilinears of each operator are passed
     * to it as one weighted sum, so that field-independent work in its derivative
     * (e.g., the three-hop recursion of the improved staggered operator) can be done
     * once per operator rather than once per term.
     *
     * ***Grid::FourFlavorStaggeredEvenEvenPseudoFermionAction::_deriv docs
     */
    const int numPoles = OneEighthNumDeriv.poles.size();
    const int denPoles = NegOneQuarterDenDeriv.poles.size();

    std::vector<FermionField> Xs(numPoles, NumOp.FermionRedBlackGrid());
    std::vector<FermionField> Ys(denPoles, NumOp.FermionRedBlackGrid());
    std::vector<FermionField> Zs(numPoles, NumOp.FermionRedBlackGrid());
    
    FermionField X(NumOp.FermionRedBlackGrid());
    FermionField Y(NumOp.FermionRedBlackGrid());
    FermionField ChiL(NumOp.FermionRedBlackGrid());
    FermionField ChiR(NumOp.FermionRedBlackGrid());

    std::vector<FermionField> denLefts, denRights, numLefts, numRights; // Eqn (6)
    std::vector<RealD> denWeights, numWeights;                          // Eqn (3)

    _multiShiftSolve(Num, OneEighthNumDeriv, Phi, Xs, X);   // Eqns (1a) & (2a)
    _multiShiftSolve(Den, NegOneQuarterDenDeriv, X, Ys, Y); // Eqns (1b) & (2b)
    _multiShiftSolve(Num, OneEighthNumDeriv, Y, Zs);        // Eqn (2c)

    denLefts.reserve(denPoles);
    denRights.reserve(denPoles);
    numLefts.reserve(2*numPoles);
    numRights.reserve(2*numPoles);
 
    for (int d = 0; d < denPoles; ++d) {
      DenOp.Meooe(Ys[d], ChiL); // Eqn (4a)

      _appendBilinear(DenOp, denLefts, denRights, Ys[d], ChiL, Ys[d], ChiL); // Eqn (7)
      denWeights.push_back(NegOneQuarterDenDeriv.residues[d]);
    }

    for (int n = 0; n < numPoles; ++n) {
      NumOp.Meooe(Zs[n], ChiL); // Eqn (4b)
      NumOp.Meooe(Xs[n], ChiR); // Eqn (4c)

      _appendBilinear(NumOp, numLefts, numRights, Zs[n], ChiL, Xs[n], ChiR); // Eqn (8a)
      _appendBilinear(NumOp, numLefts, numRights, Xs[n], ChiR, Zs[n], ChiL); // Eqn (8b)
      numWeights.insert(numWeights.end(), 2, OneEighthNumDeriv.residues[n]);
    }

    auto den = [&](auto& out) // Terms 1 and 2*** of every denominator pole; Eqn (5)
    { DenOp.MDeriv(out, denLefts, denRights, denWeights, DaggerNo); };
    auto num = [&](auto& out) // Terms 1 and 2*** of every numerator pole; Eqn (5)
    { NumOp.MDeriv(out, numLefts, numRights, numWeights, DaggerNo); };

    dSdU = Zero();
    accumulate(dSdU, _denContract, den); // <-+- First sum in Eqn (3)
    accumulate(dSdU, _numContract, num); // <-+- Second sum in Eqn (3)
  }

  void _appendBilinear(
    FermionOperator<Impl>& op,
    std::vector<FermionField>& lefts,
    std::vector<FermionField>& rights,
    const FermionField& A, // even
    const FermionField& B, // odd
    const FermionField& C, // even
    const FermionField& E  // odd
  ) { // Eqn (6) of _deriv
    FermionField minusC(C.Grid());
    minusC = -C;

    lefts.emplace_back(op.FermionGrid());
    rights.emplace_back(op.FermionGrid());

    setCheckerboard(lefts.back(), A);
    setCheckerboard(lefts.back(), B);
    setCheckerboard(rights.back(), minusC);
    setCheckerboard(rights.back(), E);
  }

public:
  virtual void refresh(const GaugeField& U, GridSerialRNG& sRNG, GridParallelRNG& pRNG)
  { NumOp.ImportGauge(U); DenOp.ImportGauge(U); _refresh(pRNG); }

  virtual RealD S(const GaugeField& U) 
  { NumOp.ImportGauge(U); DenOp.ImportGauge(U); return _action(); }

  virtual void deriv(const GaugeField& U, GaugeField& UdSdU)
  { NumOp.ImportGauge(U); DenOp.ImportGauge(U); _deriv(UdSdU); }

  virtual void refresh(
    ConfigurationBase<GaugeField>& U, 
    GridSerialRNG& sRNG, 
    GridParallelRNG& pRNG
  ) {
    if (this->hasOptedIn()) {
      NumOp.ImportGauge(_numContract); 
      DenOp.ImportGauge(_denContract); 
      _refresh(pRNG);
    } else { refresh(U.get_U(this->is_smeared), sRNG, pRNG); return; }
  }

  virtual RealD S(ConfigurationBase<GaugeField>& U) {
    if (this->hasOptedIn()) {
      NumOp.ImportGauge(_numContract); 
      DenOp.ImportGauge(_denContract); 
      return _action();
    } else { return S(U.get_U(this->is_smeared)); }
  }

  virtual RealD Sinitial(ConfigurationBase<GaugeField>& U) { return S(U); }

  virtual void deriv(ConfigurationBase<GaugeField>& U, GaugeField& UdSdU) {
    if (this->hasOptedIn()) {
      PrimalCotangentPairs<GaugeField> dSdU;
      NumOp.ImportGauge(_numContract); 
      DenOp.ImportGauge(_denContract);
      _deriv(dSdU);
      U.pullback(UdSdU, dSdU);
    } else {
      deriv(U.get_U(this->is_smeared), UdSdU);
      if (this->is_smeared) { U.smeared_force(UdSdU); }
    }
  }
};

NAMESPACE_END(Grid);

#endif // QCD_PSEUDOFERMION_STAGGERED_EVEN_EVEN_RATIO_RATIONAL_H
