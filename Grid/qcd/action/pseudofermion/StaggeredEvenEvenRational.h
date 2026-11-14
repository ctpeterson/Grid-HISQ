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
 * @file StaggeredEvenEvenRational.h
 * @author Curtis Taylor Peterson
 */

#pragma once

#include <Grid/Grid.h>

#ifndef QCD_PSEUDOFERMION_STAGGERED_EVEN_EVEN_RATIONAL_H
#define QCD_PSEUDOFERMION_STAGGERED_EVEN_EVEN_RATIONAL_H

NAMESPACE_BEGIN(Grid);

template<class Impl>
class StaggeredEvenEvenRational: public Action<typename Impl::GaugeField> {
/**
 * @class Grid::StaggeredEvenEvenRational
 * @brief Staggered even-even rational (rooted) pseudofermion action
 * @author Curtis Taylor Peterson
 * @details
 * Implements staggered even-even (i.e., reduced) rational (i.e., rooted) staggered 
 * pseudofermion action:
 * (1) S = Phi^dagger (M^dagger M)^(-Nf/4) Phi,
 * where Phi is defined on even sites only. Almost everything from the "four flavor"
 * class can be reused with minor modifications for the rooted case. As in MILC, the action
 * is evaluated as the norm of (M^dagger M)^(-Nf/8) Phi, so the action rational approximates
 * that half power (MILC's FA function).
 */
public: 
  INHERIT_IMPL_TYPES(Impl);

private:
  RealD _scale;

  StaggeredRationalActionParams _params;
  
  FermionOperator<Impl>& FermOp;

  MultiShiftFunction OneEighthAction;    // heatbath, (M^dag M)^(Nf/8)
  MultiShiftFunction NegOneEighthAction; // action, (M^dag M)^(-Nf/8)
  MultiShiftFunction NegOneQuarterDeriv; // force, (M^dag M)^(-Nf/4)

  ActionContract<GaugeField> _contract;

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
  ):_params(params),
    FermOp(Op),
    Phi(Op.FermionRedBlackGrid()) {
    GRID_ASSERT(_params.lo > 0.0 && "Lower bound of Remez approx must be positive");
    GRID_ASSERT(_params.hi > _params.lo && "Upper bound of Remez approx must be greater than lower bound");

    AlgRemez remez(_params.lo, _params.hi, _params.precision);

    _scale = std::sqrt(0.5);
    Phi.Checkerboard() = Even; 
    Phi = Zero();

    _generateRemezApproximation(OneEighthAction, remez, 8, true, false);
    _generateRemezApproximation(NegOneEighthAction, remez, 8, true, true);
    _generateRemezApproximation(NegOneQuarterDeriv, remez, 4, false, true);
    
    std::cout << GridLogMessage 
              << action_name() 
              << " initialize: complete" 
              << std::endl;
  }

  StaggeredEvenEvenRational(
    FermionOperator<Impl>& Op,
    const StaggeredRationalActionParams& params,
    const MultiShiftFunction& heatBath, // (M^dag M)^(Nf/8)
    const MultiShiftFunction& action,   // (M^dag M)^(-Nf/8)
    const MultiShiftFunction& deriv     // (M^dag M)^(-Nf/4)
  ):_params(params),
    FermOp(Op),
    OneEighthAction(heatBath),
    NegOneEighthAction(action),
    NegOneQuarterDeriv(deriv),
    Phi(Op.FermionRedBlackGrid()) {
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

public: // opt-in link interface: needs documentation
  void signContract(Primals<GaugeField> primals)
  { _contract = Action<GaugeField>::signContract(FermOp, primals); }

public: // roles of the multi-shift solves (MILC's GR, FA and MD solves)
  enum class SolveRole { HeatBath, Action, Force };

protected:
  /**
   * @brief Multi-shift solves of M^dag M on the pseudofermion checkerboard
   * @author Curtis Taylor Peterson
   * @details
   * The default solves in the precision of Impl. Derived classes can override these
   * solves, e.g. to solve in mixed precision; role identifies the solve (heatbath, action
   * or force) and maxIter is its iteration limit.
   */
  virtual void _multiShiftSolve(
    SolveRole role,
    int maxIter,
    const MultiShiftFunction& approx,
    const FermionField& in,
    std::vector<FermionField>& outs
  ) {
    SchurStaggeredOperator<FermionOperator<Impl>, FermionField> MdagM(FermOp);
    ConjugateGradientMultiShift<FermionField> solver(maxIter, approx);
    
    for (int i = 0; i < outs.size(); ++i) 
    { outs[i].Checkerboard() = in.Checkerboard(); }
    solver(MdagM, in, outs);
  }

  virtual void _multiShiftSolve(
    SolveRole role,
    int maxIter,
    const MultiShiftFunction& approx,
    const FermionField& in,
    FermionField& out
  ) {
    SchurStaggeredOperator<FermionOperator<Impl>, FermionField> MdagM(FermOp);
    ConjugateGradientMultiShift<FermionField> solver(maxIter, approx);
    std::vector<FermionField> outs(approx.poles.size(), FermOp.FermionRedBlackGrid());
    
    for (int i = 0; i < outs.size(); ++i) 
    { outs[i].Checkerboard() = in.Checkerboard(); }
    out.Checkerboard() = in.Checkerboard();
    solver(MdagM, in, outs, out);
  }

private:
  void _refresh(GridParallelRNG& pRNG) {
    /**
     * @brief Pseudofermion heatbath
     * @author Curtis Taylor Peterson
     * @details
     * Given the action of Eqn (1) in the class documentation, one wishes to generate 
     * a pseudofermion field Phi as
     * (1) Phi = (M^dag M)^{Nf/8} Eta.
     * We start off by producing a Gaussian field Eta on the even sites,
     * scaled by sqrt(1/2), with
     * (2) P(Eta) ~ exp(-Eta^dag Eta),
     * to which we obtain Phi as in Eqn (1) from a multi-shift solve.
     */
    FermionField EtaEven(FermOp.FermionRedBlackGrid());

    EtaEven.Checkerboard() = Even;
    gaussian(pRNG, EtaEven); // Eqn (2)
    EtaEven *= _scale;
    _multiShiftSolve(SolveRole::HeatBath, _params.MaxIter, OneEighthAction, EtaEven, Phi); // Eqn (1)
  }

  RealD _action() {
    /**
     * @brief Pseudofermion action
     * @author Curtis Taylor Peterson
     * @details
     * Computes the pseudofermion action S defined in Eqn (1) of the class
     * documentation as MILC does. Define
     * (1) Psi = (M^dag M)^{-Nf/8} Phi
     * and calculate S as
     * (2) S = Psi^dagger Psi,
     * which equals Phi^dagger (M^dag M)^{-Nf/4} Phi.
     */
    FermionField Psi(FermOp.FermionRedBlackGrid());
    
    _multiShiftSolve(SolveRole::Action, _params.MaxIter, NegOneEighthAction, Phi, Psi); // Eqn (1)
    return norm2(Psi); // Eqn (2)
  }

  template <class Derivative>
  void _deriv(Derivative& dSdU) {
    /**
     * @brief Wirtinger derivative of pseudofermion action
     * @author Curtis Taylor Peterson
     * @details
     * Because we are using a rational approximation of the staggered Dirac
     * operator of the form
     * (1) (M^dag M)^{-Nf/4} = alpha_0 + sum_{i=1}^{N} alpha_i / (M^dag M + beta_i),
     * the force for the pseudofermion action ends up being nothing more than a 
     * sum unrooted forces over each multi-shift solution. See documentation for the 
     * corresponding force at FourFlavorStaggeredEvenEvenPseudoFermionAction::_deriv
     * for details.
     * 
     * For each multi-shift solution Psi_i, the two terms of the unrooted force*** are
     * the derivatives of the bilinears Psi_i^dag M Chi_i (Term 1***) and
     * Chi_i^dag M^dag Psi_i (Term 2***), in the convention of the operator's
     * DerivInternal. Because the mass term drops out of the derivative and
     * M^dag = -M(-m) for staggered operators, they combine into the derivative of a
     * single full-grid bilinear:
     * (2) Term 1*** + Term 2*** = d[ Psi_i^dag M Chi_i - Chi_i^dag M Psi_i ]
     *                           = d[ L_i^dag M R_i ],
     * where
     * (3) L_i = Psi_i (even) + Chi_i (odd)
     * and
     * (4) R_i = -Psi_i (even) + Chi_i (odd).
     * The bilinears of all solutions are passed to the operator as one weighted sum
     * with the residues alpha_i as weights, so that field-independent work in the
     * operator's derivative (e.g., the three-hop recursion of the improved staggered
     * operator) can be done once per force rather than twice per pole.
     *
     * ***Grid::FourFlavorStaggeredEvenEvenPseudoFermionAction::_deriv docs
     */
    const int numPoles = NegOneQuarterDeriv.poles.size();
    std::vector<FermionField> Psis(numPoles, FermOp.FermionRedBlackGrid());
    std::vector<FermionField> lefts(numPoles, FermOp.FermionGrid());
    std::vector<FermionField> rights(numPoles, FermOp.FermionGrid());
    FermionField Chi(FermOp.FermionRedBlackGrid());
    FermionField minusPsi(FermOp.FermionRedBlackGrid());

    _multiShiftSolve(SolveRole::Force, _params.MaxIter, NegOneQuarterDeriv, Phi, Psis); // Eqn (2a)***

    for (int i = 0; i < numPoles; ++i) {
      FermOp.Meooe(Psis[i], Chi); // Eqn (2b)***
      minusPsi = -Psis[i];

      setCheckerboard(lefts[i], Psis[i]);   // <-+- Eqn (3)
      setCheckerboard(lefts[i], Chi);       // <-+
      setCheckerboard(rights[i], minusPsi); // <-+- Eqn (4)
      setCheckerboard(rights[i], Chi);      // <-+
    }

    auto poles = [&](auto& out) // Terms 1 and 2*** of every solution; Eqn (2)
    { FermOp.MDeriv(out, lefts, rights, NegOneQuarterDeriv.residues, DaggerNo); };

    dSdU = Zero();
    accumulate(dSdU, _contract, poles); // Eqn (1), each term as in Eqn (1)***
  }

public:
  virtual void refresh(const GaugeField& U, GridSerialRNG& sRNG, GridParallelRNG& pRNG)
  { FermOp.ImportGauge(U); _refresh(pRNG); }

  virtual RealD S(const GaugeField& U) 
  { FermOp.ImportGauge(U); return _action(); }

  virtual void deriv(const GaugeField& U, GaugeField& UdSdU)
  { FermOp.ImportGauge(U); _deriv(UdSdU); }

  virtual void refresh(
    ConfigurationBase<GaugeField>& U, 
    GridSerialRNG& sRNG, 
    GridParallelRNG& pRNG
  ) {
    if (this->hasOptedIn()) { FermOp.ImportGauge(_contract); _refresh(pRNG); }
    else { refresh(U.get_U(this->is_smeared), sRNG, pRNG); return; }
  }

  virtual RealD S(ConfigurationBase<GaugeField>& U) {
    if (this->hasOptedIn()) { FermOp.ImportGauge(_contract); return _action(); }
    else { return S(U.get_U(this->is_smeared)); }
  }

  virtual RealD Sinitial(ConfigurationBase<GaugeField>& U) { return S(U); }

  virtual void deriv(ConfigurationBase<GaugeField>& U, GaugeField& UdSdU) {
    if (this->hasOptedIn()) {
      PrimalCotangentPairs<GaugeField> dSdU;
      FermOp.ImportGauge(_contract);
      _deriv(dSdU);
      U.pullback(UdSdU, dSdU);
    } else {
      deriv(U.get_U(this->is_smeared), UdSdU);
      if (this->is_smeared) { U.smeared_force(UdSdU); }
    }
  }
};

NAMESPACE_END(Grid);

#endif // QCD_PSEUDOFERMION_STAGGERED_EVEN_EVEN_RATIONAL_H
