/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/qcd/action/pseudofermion/TwoPlusOneFlavorStaggeredEvenEvenRatioRational.h

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
 * @file TwoPlusOneFlavorStaggeredEvenEvenRatioRational.h
 * @author Curtis Taylor Peterson
 */

#pragma once

#include <Grid/Grid.h>

#ifndef QCD_PSEUDOFERMION_TWO_PLUS_ONE_FLAVOR_STAGGERED_EVEN_EVEN_RATIO_RATIONAL_H
#define QCD_PSEUDOFERMION_TWO_PLUS_ONE_FLAVOR_STAGGERED_EVEN_EVEN_RATIO_RATIONAL_H

NAMESPACE_BEGIN(Grid);

template<class Impl>
class TwoPlusOneFlavorStaggeredEvenEvenRatioRational: public Action<typename Impl::GaugeField> {
/**
 * @class Grid::TwoPlusOneFlavorStaggeredEvenEvenRatioRational
 * @brief Two-plus-one flavor staggered even-even ratio rational action
 * @author Curtis Taylor Peterson
 * @details
 * Let Q_n, Q_1, and Q_2 denote M^dag M on even sites for the numerator and
 * two denominator operators, with flavor counts N_n = 3, N_1 = 2, N_2 = 1. One
 * pseudofermion implements the determinant weight
 * (1) det(Q_1)^(N_1/4) det(Q_2)^(N_2/4) det(Q_n)^(-N_n/4),
 * using the positive action
 * (2) S = Phi^dag Q_n^(N_n/8) Q_2^(-N_2/8) Q_1^(-N_1/4)
 *                Q_2^(-N_2/8) Q_n^(N_n/8) Phi.
 * Num/Den refer to their positions in the pseudofermion action, as in
 * StaggeredEvenEvenRatioRational; the determinant weight has inverse powers.
 * For equal kinetic operators these matrix functions commute. In MILC's
 * HISQ factorization, Den1Op is light (N_1 = 2), Den2Op is strange (N_2 = 1),
 * and NumOp is the regulator (N_n = 3). Three separate one-flavor regulator
 * actions cancel det(Q_n)^(-3/4); the charm action is also separate.
 * Reference: https://doi.org/10.1103/PhysRevD.82.074501, Sec. II.
 */
public:
  INHERIT_IMPL_TYPES(Impl);

private:
  RealD _scale;

  StaggeredRationalActionParams _numParams;
  StaggeredRationalActionParams _den1Params;
  StaggeredRationalActionParams _den2Params;

  FermionOperator<Impl>& NumOp;
  FermionOperator<Impl>& Den1Op;
  FermionOperator<Impl>& Den2Op;

  MultiShiftFunction OneEighthNumAction;       // <---+--- action
  MultiShiftFunction NegOneEighthNumAction;    //     |
  MultiShiftFunction OneEighthDen1Action;      //     |
  MultiShiftFunction NegOneQuarterDen1Action;  //     |
  MultiShiftFunction OneEighthDen2Action;      //     |
  MultiShiftFunction NegOneEighthDen2Action;   // <---+
  MultiShiftFunction OneEighthNumDeriv;        // <---+--- force
  MultiShiftFunction NegOneQuarterDen1Deriv;   //     |
  MultiShiftFunction NegOneEighthDen2Deriv;    // <---+

  ActionContract<GaugeField> _numContract, _den1Contract, _den2Contract;

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
  TwoPlusOneFlavorStaggeredEvenEvenRatioRational(
    FermionOperator<Impl>& _numOp,
    FermionOperator<Impl>& _den1Op,
    FermionOperator<Impl>& _den2Op,
    const StaggeredRationalActionParams& numParams,
    const StaggeredRationalActionParams& den1Params,
    const StaggeredRationalActionParams& den2Params
  ):_numParams(numParams),
    _den1Params(den1Params),
    _den2Params(den2Params),
    NumOp(_numOp),
    Den1Op(_den1Op),
    Den2Op(_den2Op),
    Phi(_numOp.FermionRedBlackGrid()) {
    GRID_ASSERT(_den1Params.nf == 2 && "First denominator must have two flavors");
    GRID_ASSERT(_den2Params.nf == 1 && "Second denominator must have one flavor");
    GRID_ASSERT(_numParams.nf == 3 && "Numerator must have three regulator flavors");
    GRID_ASSERT(_den1Params.lo > 0.0 && "Den1 lower bound of Remez approx must be positive");
    GRID_ASSERT(_den1Params.hi > _den1Params.lo && "Den1 upper bound of Remez approx must be greater than lower bound");
    GRID_ASSERT(_den2Params.lo > 0.0 && "Den2 lower bound of Remez approx must be positive");
    GRID_ASSERT(_den2Params.hi > _den2Params.lo && "Den2 upper bound of Remez approx must be greater than lower bound");
    GRID_ASSERT(_numParams.lo > 0.0 && "Num lower bound of Remez approx must be positive");
    GRID_ASSERT(_numParams.hi > _numParams.lo && "Num upper bound of Remez approx must be greater than lower bound");
    conformable(Den1Op.FermionRedBlackGrid(), Den2Op.FermionRedBlackGrid());
    conformable(Den1Op.FermionRedBlackGrid(), NumOp.FermionRedBlackGrid());

    AlgRemez remezDen1(_den1Params.lo, _den1Params.hi, _den1Params.precision);
    AlgRemez remezDen2(_den2Params.lo, _den2Params.hi, _den2Params.precision);
    AlgRemez remezNum(_numParams.lo, _numParams.hi, _numParams.precision);

    _scale = std::sqrt(0.5);
    Phi.Checkerboard() = Even;
    Phi = Zero();

    _generateRemezApproximation(OneEighthDen1Action, remezDen1, _den1Params, 8, true, false);
    _generateRemezApproximation(NegOneQuarterDen1Action, remezDen1, _den1Params, 4, true, true);
    _generateRemezApproximation(OneEighthDen2Action, remezDen2, _den2Params, 8, true, false);
    _generateRemezApproximation(NegOneEighthDen2Action, remezDen2, _den2Params, 8, true, true);
    _generateRemezApproximation(OneEighthNumAction, remezNum, _numParams, 8, true, false);
    _generateRemezApproximation(NegOneEighthNumAction, remezNum, _numParams, 8, true, true);
    _generateRemezApproximation(NegOneQuarterDen1Deriv, remezDen1, _den1Params, 4, false, true);
    _generateRemezApproximation(NegOneEighthDen2Deriv, remezDen2, _den2Params, 8, false, true);
    _generateRemezApproximation(OneEighthNumDeriv, remezNum, _numParams, 8, false, false);

    std::cout << GridLogMessage
              << action_name()
              << " initialize: complete"
              << std::endl;
  }

public:
  virtual std::string action_name()
  { return "TwoPlusOneFlavorStaggeredEvenEvenRatioRationalPseudoFermionAction"; }

  virtual std::string LogParameters() {
    std::stringstream sstream;
    sstream << GridLogMessage
            << "[" << action_name() << "] m_num: " << NumOp.Mass()
            << " m_den1: " << Den1Op.Mass()
            << " m_den2: " << Den2Op.Mass() << std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] Powers (num/den1/den2) :"
            << _numParams.nf << "/4 " << _den1Params.nf << "/4 "
            << _den2Params.nf << "/4" << std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] Low (num/den1/den2)    :"
            << _numParams.lo << " " << _den1Params.lo
            << " " << _den2Params.lo << std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] High (num/den1/den2)   :"
            << _numParams.hi << " " << _den1Params.hi
            << " " << _den2Params.hi << std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] Max iterations         :"
            << _numParams.MaxIter << " " << _den1Params.MaxIter
            << " " << _den2Params.MaxIter << std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] Tolerance (Action)     :"
            << _numParams.action_tolerance << " " << _den1Params.action_tolerance
            << " " << _den2Params.action_tolerance << std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] Degree (Action)        :"
            << _numParams.action_degree << " " << _den1Params.action_degree
            << " " << _den2Params.action_degree << std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] Tolerance (MD)         :"
            << _numParams.md_tolerance << " " << _den1Params.md_tolerance
            << " " << _den2Params.md_tolerance << std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] Degree (MD)            :"
            << _numParams.md_degree << " " << _den1Params.md_degree
            << " " << _den2Params.md_degree << std::endl;
    sstream << GridLogMessage << "[" << action_name() << "] Precision              :"
            << _numParams.precision << " " << _den1Params.precision
            << " " << _den2Params.precision << std::endl;
    return sstream.str();
  }

public: // opt-in link interface
  void contract(
    Primals<GaugeField> numPrimals,
    Primals<GaugeField> den1Primals,
    Primals<GaugeField> den2Primals
  ) {
    GRID_ASSERT(!numPrimals.empty() && !den1Primals.empty() && !den2Primals.empty());
    this->initializeContracts();
    _numContract = ActionContract<GaugeField>(NumOp.identity(), numPrimals);
    _den1Contract = ActionContract<GaugeField>(Den1Op.identity(), den1Primals);
    _den2Contract = ActionContract<GaugeField>(Den2Op.identity(), den2Primals);
    this->finalizeContracts();
  }

private:
  void _multiShiftSolve(
    FermionOperator<Impl>& op,
    const StaggeredRationalActionParams& params,
    const MultiShiftFunction& approx,
    const FermionField& in,
    std::vector<FermionField>& outs,
    FermionField& out
  ) {
    SchurStaggeredOperator<FermionOperator<Impl>, FermionField> MdagM(op);
    ConjugateGradientMultiShift<FermionField> solver(params.MaxIter, approx);
    for (int i = 0; i < outs.size(); ++i)
    { outs[i].Checkerboard() = in.Checkerboard(); }
    out.Checkerboard() = in.Checkerboard();
    solver(MdagM, in, outs, out);
  }

  void _multiShiftSolve(
    FermionOperator<Impl>& op,
    const StaggeredRationalActionParams& params,
    const MultiShiftFunction& approx,
    const FermionField& in,
    std::vector<FermionField>& outs
  ) {
    FermionField out(op.FermionRedBlackGrid());
    _multiShiftSolve(op, params, approx, in, outs, out);
  }

  void _multiShiftSolve(
    FermionOperator<Impl>& op,
    const StaggeredRationalActionParams& params,
    const MultiShiftFunction& approx,
    const FermionField& in,
    FermionField& out
  ) {
    std::vector<FermionField> outs(approx.poles.size(), op.FermionRedBlackGrid());
    _multiShiftSolve(op, params, approx, in, outs, out);
  }

private:
  void _refresh(GridParallelRNG& pRNG) {
    /**
     * @brief Pseudofermion heatbath
     * @author Curtis Taylor Peterson
     * @details
     * For Eqn (2) in the class documentation, generate
     * (1) Phi = Q_n^(-N_n/8) Q_2^(N_2/8) Q_1^(N_1/8) eta |_{even},
     * where P(eta) ~ exp(-eta^dag eta). The order inverts the factors of
     * the action without assuming commutation. As in the four-flavor action,
     * generate full-field Gaussian noise, scale by sqrt(1/2), and select even.
     */
    FermionField eta(Den1Op.FermionGrid());
    FermionField EtaEven(Den1Op.FermionRedBlackGrid());
    FermionField X(Den1Op.FermionRedBlackGrid());
    FermionField Y(Den1Op.FermionRedBlackGrid());

    gaussian(pRNG, eta);
    eta *= _scale;
    pickCheckerboard(Even, EtaEven, eta);

    _multiShiftSolve(Den1Op, _den1Params, OneEighthDen1Action, EtaEven, X);
    _multiShiftSolve(Den2Op, _den2Params, OneEighthDen2Action, X, Y);
    _multiShiftSolve(NumOp, _numParams, NegOneEighthNumAction, Y, Phi);
  }

  RealD _action() {
    /**
     * @brief Pseudofermion action
     * @author Curtis Taylor Peterson
     * @details
     * Evaluate Eqn (2) in the class documentation using
     * (1a) X = Q_n^(N_n/8) Phi,
     * (1b) Y = Q_2^(-N_2/8) X,
     * (1c) Z = Q_1^(-N_1/4) Y,
     * then S = Y^dag Z. All fields live on even sites.
     */
    FermionField X(Den1Op.FermionRedBlackGrid());
    FermionField Y(Den1Op.FermionRedBlackGrid());
    FermionField Z(Den1Op.FermionRedBlackGrid());

    _multiShiftSolve(NumOp, _numParams, OneEighthNumAction, Phi, X);
    _multiShiftSolve(Den2Op, _den2Params, NegOneEighthDen2Action, X, Y);
    _multiShiftSolve(Den1Op, _den1Params, NegOneQuarterDen1Action, Y, Z);

    return innerProduct(Y, Z).real();
  }

  template <class Derivative>
  void _deriv(Derivative& dSdU) {
    /**
     * @brief Wirtinger derivative of pseudofermion action
     * @author Curtis Taylor Peterson
     * @details
     * Use X, Y, Z from _action and T = Q_2^(-N_2/8) Z, evaluated with the MD
     * rational approximations. For pole beta of each corresponding function,
     * (1a) YsDen1 = (Q_1 + beta)^(-1) Y,
     * (1b) XsDen2 = (Q_2 + beta)^(-1) X,
     * (1c) ZsDen2 = (Q_2 + beta)^(-1) Z,
     * (1d) XsNum  = (Q_n + beta)^(-1) Phi,
     * (1e) TsNum  = (Q_n + beta)^(-1) T.
     * Differentiating each inverse gives
     * (2) -dS = sum_l r_l YsDen1^dag dQ_1 YsDen1
     *         + sum_s r_s [ ZsDen2^dag dQ_2 XsDen2 + h.c. ]
     *         + sum_r r_r [ TsNum^dag dQ_n XsNum + h.c. ].
     * The first denominator term is the ordinary rooted-action derivative.
     * The second denominator and numerator each appear on both sides, as in
     * StaggeredEvenEvenRatioRational.
     * Apply Meooe to form the odd fields and accumulate the MeoDeriv/MoeDeriv
     * terms as in FourFlavorStaggeredEvenEvenPseudoFermionAction::_deriv.
     */
    const int den1Poles = NegOneQuarterDen1Deriv.poles.size();
    const int den2Poles = NegOneEighthDen2Deriv.poles.size();
    const int numPoles = OneEighthNumDeriv.poles.size();

    std::vector<FermionField> YsDen1(den1Poles, Den1Op.FermionRedBlackGrid());
    std::vector<FermionField> XsDen2(den2Poles, Den1Op.FermionRedBlackGrid());
    std::vector<FermionField> ZsDen2(den2Poles, Den1Op.FermionRedBlackGrid());
    std::vector<FermionField> XsNum(numPoles, Den1Op.FermionRedBlackGrid());
    std::vector<FermionField> TsNum(numPoles, Den1Op.FermionRedBlackGrid());

    FermionField X(Den1Op.FermionRedBlackGrid());
    FermionField Y(Den1Op.FermionRedBlackGrid());
    FermionField Z(Den1Op.FermionRedBlackGrid());
    FermionField T(Den1Op.FermionRedBlackGrid());
    FermionField ChiL(Den1Op.FermionRedBlackGrid());
    FermionField ChiR(Den1Op.FermionRedBlackGrid());

    _multiShiftSolve(NumOp, _numParams, OneEighthNumDeriv, Phi, XsNum, X);
    _multiShiftSolve(Den2Op, _den2Params, NegOneEighthDen2Deriv, X, XsDen2, Y);
    _multiShiftSolve(Den1Op, _den1Params, NegOneQuarterDen1Deriv, Y, YsDen1, Z);
    _multiShiftSolve(Den2Op, _den2Params, NegOneEighthDen2Deriv, Z, ZsDen2, T);
    _multiShiftSolve(NumOp, _numParams, OneEighthNumDeriv, T, TsNum);

    dSdU = Zero();

    for (int d = 0; d < den1Poles; ++d) {
      const RealD rd = NegOneQuarterDen1Deriv.residues[d]; // Eqn (2)

      Den1Op.Meooe(YsDen1[d], ChiL); // Eqn (1a)

      auto eo = [&](auto& out)
      { Den1Op.MeoDeriv(out, YsDen1[d], ChiL, DaggerNo); };
      auto oe = [&](auto& out)
      { Den1Op.MoeDeriv(out, ChiL, YsDen1[d], DaggerYes); };

      accumulate(dSdU, _den1Contract, rd, eo); // <-+- Den1 sum in Eqn (2)
      accumulate(dSdU, _den1Contract, rd, oe); // <-+
    }

    for (int n = 0; n < den2Poles; ++n) {
      const RealD rn = NegOneEighthDen2Deriv.residues[n]; // Eqn (2)

      Den2Op.Meooe(ZsDen2[n], ChiL);
      Den2Op.Meooe(XsDen2[n], ChiR);

      // Meo contribution
      auto eoA = [&](auto& out)
      { Den2Op.MeoDeriv(out, ZsDen2[n], ChiR, DaggerNo); };
      auto eoB = [&](auto& out)
      { Den2Op.MeoDeriv(out, XsDen2[n], ChiL, DaggerNo); };

      // Moe contribution
      auto oeA = [&](auto& out)
      { Den2Op.MoeDeriv(out, ChiL, XsDen2[n], DaggerYes); };
      auto oeB = [&](auto& out)
      { Den2Op.MoeDeriv(out, ChiR, ZsDen2[n], DaggerYes); };

      accumulate(dSdU, _den2Contract, rn, eoA); // <-+- Den2 sum in Eqn (2)
      accumulate(dSdU, _den2Contract, rn, eoB); //   |
      accumulate(dSdU, _den2Contract, rn, oeA); //   |
      accumulate(dSdU, _den2Contract, rn, oeB); // <-+
    }

    for (int n = 0; n < numPoles; ++n) {
      const RealD rn = OneEighthNumDeriv.residues[n]; // Eqn (2)

      NumOp.Meooe(TsNum[n], ChiL);
      NumOp.Meooe(XsNum[n], ChiR);

      // Meo contribution
      auto eoA = [&](auto& out)
      { NumOp.MeoDeriv(out, TsNum[n], ChiR, DaggerNo); };
      auto eoB = [&](auto& out)
      { NumOp.MeoDeriv(out, XsNum[n], ChiL, DaggerNo); };

      // Moe contribution
      auto oeA = [&](auto& out)
      { NumOp.MoeDeriv(out, ChiL, XsNum[n], DaggerYes); };
      auto oeB = [&](auto& out)
      { NumOp.MoeDeriv(out, ChiR, TsNum[n], DaggerYes); };

      accumulate(dSdU, _numContract, rn, eoA); // <-+- Num sum in Eqn (2)
      accumulate(dSdU, _numContract, rn, eoB); //   |
      accumulate(dSdU, _numContract, rn, oeA); //   |
      accumulate(dSdU, _numContract, rn, oeB); // <-+
    }
  }

public:
  virtual void refresh(const GaugeField& U, GridSerialRNG& sRNG, GridParallelRNG& pRNG)
  { NumOp.ImportGauge(U); Den1Op.ImportGauge(U); Den2Op.ImportGauge(U); _refresh(pRNG); }

  virtual RealD S(const GaugeField& U)
  { NumOp.ImportGauge(U); Den1Op.ImportGauge(U); Den2Op.ImportGauge(U); return _action(); }

  virtual void deriv(const GaugeField& U, GaugeField& UdSdU)
  { NumOp.ImportGauge(U); Den1Op.ImportGauge(U); Den2Op.ImportGauge(U); _deriv(UdSdU); }

  virtual void refresh(
    ConfigurationBase<GaugeField>& U,
    GridSerialRNG& sRNG,
    GridParallelRNG& pRNG
  ) {
    if (this->hasOptedIn()) {
      NumOp.ImportGauge(_numContract); 
      Den1Op.ImportGauge(_den1Contract); 
      Den2Op.ImportGauge(_den2Contract);
      _refresh(pRNG);
    } else { refresh(U.get_U(this->is_smeared), sRNG, pRNG); return; }
  }

  virtual RealD S(ConfigurationBase<GaugeField>& U) {
    if (this->hasOptedIn()) {
      NumOp.ImportGauge(_numContract); 
      Den1Op.ImportGauge(_den1Contract); 
      Den2Op.ImportGauge(_den2Contract);
      return _action();
    } else { return S(U.get_U(this->is_smeared)); }
  }

  virtual RealD Sinitial(ConfigurationBase<GaugeField>& U) { return S(U); }

  virtual void deriv(ConfigurationBase<GaugeField>& U, GaugeField& UdSdU) {
    if (this->hasOptedIn()) {
      PrimalCotangentPairs<GaugeField> dSdU;
      NumOp.ImportGauge(_numContract); 
      Den1Op.ImportGauge(_den1Contract); 
      Den2Op.ImportGauge(_den2Contract);
      _deriv(dSdU);
      U.pullback(UdSdU, dSdU);
    } else {
      deriv(U.get_U(this->is_smeared), UdSdU);
      if (this->is_smeared) { U.smeared_force(UdSdU); }
    }
  }
};

NAMESPACE_END(Grid);

#endif // QCD_PSEUDOFERMION_TWO_PLUS_ONE_FLAVOR_STAGGERED_EVEN_EVEN_RATIO_RATIONAL_H
