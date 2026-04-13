/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/qcd/action/pseudofermion/StaggeredEE.h

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
  @file StaggeredEE.h
  @author Curtis Taylor Peterson
*/

#pragma once
#include <Grid/Grid.h>

#ifndef QCD_PSEUDOFERMION_STAGGERED_EE_H
#define QCD_PSEUDOFERMION_STAGGERED_EE_H

NAMESPACE_BEGIN(Grid);

enum Solver {ActionSolve,DerivativeSolve};

template <class Impl>
class FourFlavorStaggeredEvenEvenPseudoFermionAction: public Action<typename Impl::GaugeField> 
{
public: INHERIT_IMPL_TYPES(Impl);

private:
  RealD _scale;
  FermionOperator<Impl> &FermOp;
  OperatorFunction<FermionField> &DerivativeSolver;
  OperatorFunction<FermionField> &ActionSolver;

public:
  RealD mass;
  FermionField Phi;

public:
  FourFlavorStaggeredEvenEvenPseudoFermionAction(
    FermionOperator<Impl> &Op,
    OperatorFunction<FermionField> &DS,
    OperatorFunction<FermionField> &AS
  ):DerivativeSolver(DS),
    ActionSolver(AS),
    FermOp(Op),
    Phi(Op.FermionRedBlackGrid()) {
    mass = Op.Mass(), 
    _scale = std::sqrt(0.5), 
    Phi.Checkerboard() = Even, 
    Phi = Zero();
  }

  virtual std::string action_name(){ return "FourFlavorStaggeredEvenEvenPseudoFermionAction"; }

  virtual std::string LogParameters()
  {return "type: staggered \"half\" pseudofermion mass: " + std::to_string(mass);}

private:
  void _solve(FermionField &psi, Solver solver)
  {
    /**
     * @brief Fermion solve
     * @details
     * If doing action solve, gives
     * (1) psi = Mdag^-1 phi_e [action].
     * If doing derivative (force) solve, gives
     * (2) psi = M(m_s=1) (MdagM)^-1 phi_e,
     * where M(m_s=1) denotes the Dirac operator with unit mass
     * and (MdagM)^-1 is the inverse of MdagM with the appropriate
     * bare fermion mass. See notes in "deriv" method for description
     * of why this is useful for the derivative. The difference
     * amounts to whether or not (MdagM)^-1 phi_e is multiplied by
     * the bare fermion mass and inserted in the even "slot" of
     * psi or is kept "as is" (i.e., not multiplied by the bare
     * fermion mass). See notes in NaiveStaggeredFermion 
     * and ImprovedStaggeredFermion DerivInternal methods for further
     * simplifications that work for both full and half fields.
     * @param FermionField Solution
     * @param Solver Solver (ActionSolve or DerivativeSolve; see enum)
     */
    SchurStaggeredOperator<FermionOperator<Impl>, FermionField> MdagM(FermOp);
    FermionField PsiE(FermOp.FermionRedBlackGrid());
    FermionField PsiO(FermOp.FermionRedBlackGrid());

    PsiE.Checkerboard() = Even, PsiO.Checkerboard() = Odd;
    psi = Zero(), PsiE = Zero(), PsiO = Zero();

    switch (solver) {
      case ActionSolve: ActionSolver(MdagM,Phi,PsiE); break;
      case DerivativeSolve: DerivativeSolver(MdagM,Phi,PsiE); break;
    }
    FermOp.Meooe(PsiE,PsiO);
    switch (solver) {
      case ActionSolve: 
        if (mass != 0.0) {PsiE = mass*PsiE; setCheckerboard(psi, PsiE);} break;
      case DerivativeSolve: setCheckerboard(psi, PsiE); break;
    }
    setCheckerboard(psi,PsiO);
  };

  void _refresh(GridParallelRNG &pRNG)
  {
    /**  
	   * @brief Pseudofermion heatbath
	   * @details 
	   * "Half" pseudofermion fields distributed as
	   * (1) Pr(phi_e) = exp{- phi_e^dag (MdagM)^-1 phi_e} = exp(- eta^dag eta).
	   * Hence, pseudofermion heatbath performed by first drawing eta from
	   * (2) Gaussian(eta) = exp(- 2 eta^dag eta / 2),
	   * then taking
	   * (3) phi_e = Mdag eta|_{even sites}.
	   * @param GaugeField Grid Gauge field type
	   * @param GridSerialRNG Grid serial random number generator type; not used
	   * @param GridParallelRNG Grid parallel random number generator type
	  */
    FermionField eta(FermOp.FermionGrid()), phi(FermOp.FermionGrid());
    
    gaussian(pRNG,eta); // (2.a)
    eta = _scale*eta; // (2.b)
    FermOp.Mdag(eta,phi); // (3.a)
    pickCheckerboard(Even,Phi,phi); // (3.b)
  }

  RealD _action()
  {
    /**
	   * @brief Pseudofermion action
	   * @details
	   * Pseudofermion action for staggered half field is
	   * (1) S(phi_e) = - phi_e^dag (MdagM)^-1 phi_e.
	   * According to the even/odd split in the staggered
	   * Dirac operator, we have
	   * (2) S = -eta^dag eta,
	   * where
	   * (3) eta = M | (DDdag + m^2) phi_e, 0 |^T
	   * @param GaugeField Input gauge field
	  */
    FermionField psi(FermOp.FermionGrid());
    _solve(psi,ActionSolve); // (3)
    return norm2(psi); // (2)
  }

  void _deriv(GaugeField &dSdU) {
    /** 
     * @brief Derivative of fermion action
     * @details
     * Let D_mu^i(m) be the link derivative; i.e.,
     * (1) D_mu^i(m) U_nu(n) = T^i U_nu(m) delta_{mu,nu} delta_{m,n}
     * with {T^i} the SU(N) Lie algebra basis conventionally used in
     * high-energy physics. Moreover, define
     * (2) eta_e = (MdagM)^-1 phi_e
     * and
     * (3) eta = (Mdag)^-1 phi_e = M eta_e.
     * Then, after a decent bit of algebra, the link derivative of 
     * the action is
     * (4) D_mu^i(m) S
     *      = Tr[ (D_mu^i(m) M) eta_e eta^dag ] if m is even
     *      = -Tr[ (D_mu^i(m) M) eta eta_e^dag ] if m is odd.
     * This can be simplified quite a bit. First, define
     * (5) p(n) = (-1)^{n_1+n_2+...+Nd},
     * where n = (n_1,n_2,...,Nd) and Nd is the number of dimensions.
     * Then, already, we can express the force as
     * (6) D_mu^i(m) S = p(m) Tr[ (D_mu^i(m) M) eta_e eta^dag ].
     * Now define
     * (7) psi = M(m_s=1) eta_e,
     * where M(m_s=1) is the Dirac operator with unit mass and eta_e 
     * is defined as in Eqn. (2) using the correct bare fermion mass.
     * Then the in Eqn. (6) reduces to
     * (8) D_mu^i(m) S = p(m) Tr[ (D_mu^i(m) M) psi psi^dag ].
     * As such, we only need to calculate the outer product once, which
     * is not the case when we use full fields (defined on even & odd sites).
     * @param GaugeField Input gauge field
     * @param GaugeField Action derivative
    */
    FermionField psi(FermOp.FermionGrid());
    std::vector<FermionField> psiv(Nd,FermOp.FermionGrid());
    Lattice<iScalar<vInteger>> 
      x(FermOp.FermionGrid()), 
      y(FermOp.FermionGrid()), 
      z(FermOp.FermionGrid()), 
      t(FermOp.FermionGrid()), 
      xyzt(FermOp.FermionGrid());

    _solve(psi,DerivativeSolve);
    FermOp.MDeriv(dSdU, psi, psi, DaggerNo);

    // rephase odd sites
    LatticeCoordinate(x, 0);
    LatticeCoordinate(y, 1);
    LatticeCoordinate(z, 2);
    LatticeCoordinate(t, 3);
    xyzt = x + y + z + t;
    for (int mu = 0; mu < Nd; ++mu) {
      GaugeLinkField dSdUmu = PeekIndex<LorentzIndex>(dSdU, mu);
      PokeIndex<LorentzIndex>(dSdU, where(mod(xyzt, 2) != (Integer)0, dSdUmu, -dSdUmu), mu);
    }
  }

public:
  virtual void refresh(const GaugeField &U, GridSerialRNG &sRNG, GridParallelRNG &pRNG)
  { FermOp.ImportGauge(U); _refresh(pRNG); }

  virtual RealD S(const GaugeField &U) { FermOp.ImportGauge(U); return _action(); }

  virtual void deriv(const GaugeField &U, GaugeField &dSdU) 
  { FermOp.ImportGauge(U); _deriv(dSdU); }

  virtual void refresh(
    ConfigurationBase<GaugeField> &U, 
    GridSerialRNG &sRNG, 
    GridParallelRNG &pRNG
  ) { refresh(U.get_SmearedU(), sRNG, pRNG); }

  virtual RealD S(ConfigurationBase<GaugeField> &U) { return S(U.get_SmearedU()); }

  virtual RealD Sinitial(ConfigurationBase<GaugeField> &U) { return _action(); }

  virtual void deriv(ConfigurationBase<GaugeField> &U, GaugeField &dSdU)
  { deriv(U.get_SmearedU(), dSdU); if (this->is_smeared) { U.smeared_force(dSdU); } }
};

NAMESPACE_END(Grid);

#endif