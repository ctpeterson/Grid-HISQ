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

#ifndef QCD_PSEUDOFERMION_FOUR_FLAVOR_STAGGERED_EVEN_EVEN_H
#define QCD_PSEUDOFERMION_FOUR_FLAVOR_STAGGERED_EVEN_EVEN_H

NAMESPACE_BEGIN(Grid);

template <class Impl>
class FourFlavorStaggeredEvenEvenPseudoFermionAction: public Action<typename Impl::GaugeField> {
/**
 * @class Grid::FourFlavorStaggeredEvenEvenPseudoFermionAction
 * @brief Staggered even-even pseudofermion action
 * @author Curtis Taylor Peterson
 * @details
 * Implements the staggered even-even (i.e., reduced) pseudofermion action:
 * (1) S = Phi^dag (M^dag M)^{-1} Phi,
 * where Phi is defined on even sites only. The reduction acts to halve the number of 
 * continuum Dirac fermions; you can see this by taking the determinant of the
 * Schur-decomposed staggered Dirac operator in the even/odd "basis".
 */
public: 
  INHERIT_IMPL_TYPES(Impl);
  using Action<GaugeField>::bindLinks;

private:
  RealD _scale;

  FermionOperator<Impl>& FermOp;
  
  OperatorFunction<FermionField>& DerivativeSolver;
  OperatorFunction<FermionField>& ActionSolver;

  LinkCoordinator<Impl> Links{&FermOp};

public:
  FermionField Phi;

public:
  FourFlavorStaggeredEvenEvenPseudoFermionAction(
    FermionOperator<Impl>& Op,
    OperatorFunction<FermionField>& DS,
    OperatorFunction<FermionField>& AS
  ):FermOp(Op),
    DerivativeSolver(DS),
    ActionSolver(AS),
    Phi(Op.FermionRedBlackGrid()) {
    _scale = std::sqrt(0.5); 
    Phi.Checkerboard() = Even; 
    Phi = Zero();
  }

public:
  virtual std::string action_name() 
  { return "FourFlavorStaggeredEvenEvenPseudoFermionAction"; }

  virtual std::string LogParameters() { 
    std::stringstream sstream;
    sstream << GridLogMessage 
            << "[" << action_name() << "] mass: " << FermOp.Mass()
            << std::endl;
    return sstream.str();
  }

public: // opt-in link interface
  void useLinkMap() { Links.useLinkMap(); }

  void bindLinks(const void* op, const LinkBinding<GaugeField>& bind)
  { Links.select(op, bind); }

  void bindLinks(const LinkBinding<GaugeField>& binding)
  { Links.select(FermOp.linkIdentity(), binding); }

private:
  void _refresh(GridParallelRNG& pRNG) {
    /**
     * @brief Pseudofermion heatbath
     * @author Curtis Taylor Peterson
     * @details
     * The even-even pseudofermion fields are distributed as
     * (1) P(Phi) ~ exp(-S),
     * with S defined in Eqn (1) of the class documentation. We draw
     * ***full*** fields eta from 
     * (2) P(eta) ~ exp(-0.5 eta^dag eta)
     * using some Gaussian sampling algorithm and take
     * (3) Phi = sqrt(1/2) M^dag eta |_{even}.
     * The Jacobian of the transformation eta -> Phi is constant.
     */
    FermionField eta(FermOp.FermionGrid()), phi(FermOp.FermionGrid());
    gaussian(pRNG, eta);              // Eqn (2)
    eta *= _scale;                    // <-+- Eqn (3)
    FermOp.Mdag(eta, phi);            //   |
    pickCheckerboard(Even, Phi, phi); // <-+
  }

  RealD _action() {
    /**
     * @brief Returns the pseudofermion action
     * @author Curtis Taylor Peterson
     * @details
     * Computes the pseudofermion action S defined in Eqn (1) of the class
     * documentation. The calculation is straightforward. Define
     * (1) Psi = (M^dag M)^{-1} Phi
     * and calculate S as
     * (2) S = Phi^dagger Psi.
     * 
     * Alternatively, we could have taken
     * (3) Psi = M (M^dag M)^{-1} Phi
     * and calculated S as
     * (4) S = Psi^dag Psi.
     * Many codebases (such as Quantum EXpressions) do this; however, this method
     * avoids an additional application of the M operator and works fully within the 
     * even checkerboard.
     */
    FermionField Psi(FermOp.FermionRedBlackGrid());
    SchurStaggeredOperator<FermionOperator<Impl>, FermionField> MdagMOp(FermOp);
    Psi = Zero();
    ActionSolver(MdagMOp, Phi, Psi);      // Eqn (1)
    return innerProduct(Phi, Psi).real(); // Eqn (2)
  }

  template <class Derivatives>
  void _deriv(Derivatives& dSdU) {
    /**
     * @brief Pseudofermion action Wirtinger derivative
     * @author Curtis Taylor Peterson
     * @details
     * Calculates the Wirtinger derivative of the pseudofermion action defined in Eqn (1)
     * of the class documentation. One has
     * (1) -dS = Phi^dag  (M^dag M)^-1 [ M^dag dM + dM^dag M ] (M^dag M)^-1 Phi
     *         = Chi^dag dM Psi + Psi^dag dM^dag Chi,
     * where I've defined
     * (2a) Psi = (M^dag M)^{-1} Phi
     * and
     * (2b) Chi = M Psi.
     * This is all fine and dandy, but you may be scratching your head a bit upon 
     * further inspection. Let's carefully walk through the checkerboarding to see
     * how the algebra translates into method calls. 
     * 
     * We can work fully within the checkerboards by thinking through the fields that
     * we're working with and the calls that we're making. Start by recalling that our
     * frozen pseudofermion field Phi lives on the even checkerboard. Because M^dag M 
     * connects checkerboards to themselves, Psi remains even. To get Chi, we apply M;
     * however, you'll immediately note that M Psi has support on both checkerboards,
     * yet we only keep the odd part (which amounts to applying Moe to Psi;
     * Meooe selects this block because Psi is even). To understand why, let's consider
     * the two terms in Eqn (1).
     * 
     * Term 1: dM is essentially the massless Dirac operator if we think in terms of 
     *         the unprojected left-trivialized derivative (U times the forward 
     *         Wirtinger derivative). As such, applying dM to Psi amounts to applying 
     *         just Moe to Psi, which results in an odd field. As such, taking the 
     *         inner product with Chi then knocks out any would-be even component in 
     *         Chi had we computed it using the full M operator instead.
     * Term 2: dM^dag is again the conjugate of the massless Dirac operator. Psi^dag 
     *         is going to select out the even part of whatever M^dag Chi is. If we 
     *         had kept the full M^dag Chi, the odd part would have just been Moe 
     *         multiplied by m Psi. As such, we can again safely restrict our attention
     *         to the odd checkerboard of Chi, which then gets mapped into the even
     *         checkerboard that is selected out by Psi^dag upon the application of 
     *         M^dag to Chi.
     * 
     * If you want to learn more about how the derivatives of the kinetic bilinears 
     * work, please refer to the documentation of the DerivInternal methods of the 
     * relevant staggered fermion operator classes. 
     */
    FermionField Psi(FermOp.FermionRedBlackGrid());
    FermionField Chi(FermOp.FermionRedBlackGrid());
    SchurStaggeredOperator<FermionOperator<Impl>, FermionField> MdagM(FermOp);

    Psi = Zero();
    DerivativeSolver(MdagM, Phi, Psi); // Eqn (2a)
    FermOp.Meooe(Psi, Chi);            // Eqn (2b)

    auto eo = [&](auto& op, auto& out, const auto& in)  // Term 1
    { op.MeoDeriv(out, in, Psi, Chi, DaggerNo); };      
    auto oe = [&](auto& op, auto& out, const auto& in)  // Term 2
    { op.MoeDeriv(out, in, Chi, Psi, DaggerYes); };      

    dSdU.accumulate(FermOp, eo); // <-+- Eqn (1)
    dSdU.accumulate(FermOp, oe); // <-+
  }

public:
  virtual void refresh(const GaugeField& U, GridSerialRNG& sRNG, GridParallelRNG& pRNG)
  { Links.refresh(U, [&]{ _refresh(pRNG); }); }

  virtual RealD S(const GaugeField& U) 
  { return Links.action(U, [&]{ return _action(); }); }

  virtual void deriv(const GaugeField& U, GaugeField& dSdU) 
  { Links.deriv(U, dSdU, [&](auto& deriv) { _deriv(deriv); }); }

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

#endif // QCD_PSEUDOFERMION_FOUR_FLAVOR_STAGGERED_EVEN_EVEN_H
