/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/qcd/action/pseudofermion/FourFlavorStaggeredPauliVillars.h

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
 * @file FourFlavorStaggeredPauliVillars.h
 * @author Curtis Taylor Peterson
 */

#pragma once

#include <Grid/Grid.h>

#ifndef QCD_PSEUDOFERMION_FOUR_FLAVOR_STAGGERED_PAULI_VILLARS_H
#define QCD_PSEUDOFERMION_FOUR_FLAVOR_STAGGERED_PAULI_VILLARS_H

NAMESPACE_BEGIN(Grid);

template <class Impl>
class FourFlavorStaggeredPauliVillarsPseudoFermionAction: public Action<typename Impl::GaugeField> {
/**
 * @class Grid::FourFlavorStaggeredPauliVillarsPseudoFermionAction
 * @brief Four-flavor staggered Pauli-Villars action
 * @author Curtis Taylor Peterson
 * @details
 * Implements the staggered even-even Pauli-Villars action:
 * (1) S = Phi^dag (M^dag M) Phi,
 * where Phi is defined on even sites only. Integrating over Phi gives
 * det[(M^dag M)_{ee}]^{-1}, the inverse of the determinant represented by
 * FourFlavorStaggeredEvenEvenPseudoFermionAction at the same mass.
 */
public:
  INHERIT_IMPL_TYPES(Impl);

private:
  RealD _scale;

  FermionOperator<Impl>& FermOp;

  OperatorFunction<FermionField>& ActionSolver;

  ActionContract<GaugeField> _contract;

public:
  FermionField Phi;

public:
  FourFlavorStaggeredPauliVillarsPseudoFermionAction(
    FermionOperator<Impl>& Op,
    OperatorFunction<FermionField>& AS
  ):FermOp(Op),
    ActionSolver(AS),
    Phi(Op.FermionRedBlackGrid()) {
    _scale = std::sqrt(0.5);
    Phi.Checkerboard() = Even;
    Phi = Zero();
  }

public:
  virtual std::string action_name()
  { return "FourFlavorStaggeredPauliVillarsPseudoFermionAction"; }

  virtual std::string LogParameters() {
    std::stringstream sstream;
    sstream << GridLogMessage
            << "[" << action_name() << "] mass: " << FermOp.Mass()
            << std::endl;
    return sstream.str();
  }

public: // opt-in link interface: needs documentation
  void signContract(Primals<GaugeField> primals)
  { _contract = Action<GaugeField>::signContract(FermOp, primals); }

private:
  void _refresh(GridParallelRNG& pRNG) {
    /**
     * @brief Pauli-Villars heatbath
     * @author Curtis Taylor Peterson
     * @details
     * Draw the same full Gaussian noise as the four-flavor pseudofermion
     * heatbath, scaled by sqrt(1/2). With A = (M^dag M)_{ee}, form
     * (1) Psi = (M^dag eta)_{even}
     *         = m eta_{even} + (K^dag)_{eo} eta_{odd},
     * and solve
     * (2) Phi = A^{-1} Psi.
     * Since Psi has covariance A, Phi has covariance A^{-1}, as required
     * by exp(-S). This is the only solve needed by the PV action.
     */
    FermionField etaEven(FermOp.FermionRedBlackGrid());
    FermionField etaOdd(FermOp.FermionRedBlackGrid());
    FermionField Psi(FermOp.FermionRedBlackGrid());
    SchurStaggeredOperator<FermionOperator<Impl>, FermionField> MdagMOp(FermOp);

    {
      FermionField eta(FermOp.FermionGrid());
      gaussian(pRNG, eta);
      eta *= _scale;
      pickCheckerboard(Even, etaEven, eta);
      pickCheckerboard(Odd, etaOdd, eta);
    }

    FermOp.MeooeDag(etaOdd, Psi); // <-+- Eqn (1)
    Psi += FermOp.Mass()*etaEven; // <-+
    Phi = Zero();
    ActionSolver(MdagMOp, Psi, Phi); // Eqn (2)
  }

  RealD _action() {
    /**
     * @brief Returns Pauli-Villars action
     * @author Curtis Taylor Peterson
     * @details
     * With Phi supported only on even sites, M Phi has the even component
     * m Phi and the odd component Chi = Moe Phi. Consequently,
     * (1) S = ||M Phi||^2 = m^2 ||Phi||^2 + ||Chi||^2.
     */
    FermionField Chi(FermOp.FermionRedBlackGrid());

    FermOp.Meooe(Phi, Chi);
    return FermOp.Mass()*FermOp.Mass()*norm2(Phi) + norm2(Chi); // Eqn (1)
  }

  template <class Derivative>
  void _deriv(Derivative& dSdU) {
    /**
     * @brief Wirtinger derivative of Pauli-Villars action
     * @author Curtis Taylor Peterson
     * @details
     * The field Phi is held fixed along the trajectory. Defining
     * (1) Chi = Moe Phi,
     * the derivative is
     * (2) dS = Chi^dag dM Phi + Phi^dag dM^dag Chi.
     * The checkerboard bilinears are the same as in the four-flavor
     * pseudofermion action, with Psi replaced by Phi and the opposite
     * sign because Eqn (1) of the class documentation contains M^dag M
     * rather than its inverse. The weighted accumulations form the
     * algebraic derivative; the public interface converts it to the
     * HMC force convention.
     */
    FermionField Chi(FermOp.FermionRedBlackGrid());

    FermOp.Meooe(Phi, Chi); // Eqn (1)

    auto eo = [&](auto& out) // Term 1
    { FermOp.MeoDeriv(out, Phi, Chi, DaggerNo); };
    auto oe = [&](auto& out) // Term 2
    { FermOp.MoeDeriv(out, Chi, Phi, DaggerYes); };

    dSdU = Zero();
    accumulate(dSdU, _contract, -1.0, eo); // <-+- Eqn (2)
    accumulate(dSdU, _contract, -1.0, oe); // <-+
  }

public:
  virtual void refresh(const GaugeField& U, GridSerialRNG& sRNG, GridParallelRNG& pRNG)
  { FermOp.ImportGauge(U); _refresh(pRNG); }

  virtual RealD S(const GaugeField& U)
  { FermOp.ImportGauge(U); return _action(); }

  virtual void deriv(const GaugeField& U, GaugeField& UdSdU)
  {
    FermOp.ImportGauge(U);
    _deriv(UdSdU);
    // Legacy operator derivatives already contain the link multiplication.
    // HMC needs dS/dt = -2 Re Tr(P UdSdU) for U(t) = exp(t P) U.
    // The opt-in path applies this minus sign in the configuration pullback.
    UdSdU = -UdSdU;
  }

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

#endif // QCD_PSEUDOFERMION_FOUR_FLAVOR_STAGGERED_PAULI_VILLARS_H
