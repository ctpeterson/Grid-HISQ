/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/qcd/action/fermion/ImprovedStaggeredFermion.cc

Copyright (C) 2015

Author: Azusa Yamaguchi, Peter Boyle, Curtis Taylor Peterson

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
#include <Grid/Grid.h>

#pragma once 

NAMESPACE_BEGIN(Grid);

/////////////////////////////////
// Constructor and gauge import
/////////////////////////////////

template <class Impl>
ImprovedStaggeredFermion<Impl>::ImprovedStaggeredFermion(GridCartesian &Fgrid, GridRedBlackCartesian &Hgrid, 
							 RealD _mass,
							 RealD _c1, RealD _c2,RealD _u0,
							 const ImplParams &p)
  : Kernels(p),
    _grid(&Fgrid),
    _cbgrid(&Hgrid),
    Stencil(&Fgrid, npoint, Even, directions, displacements,p),
    StencilEven(&Hgrid, npoint, Even, directions, displacements,p),  // source is Even
    StencilOdd(&Hgrid, npoint, Odd, directions, displacements,p),  // source is Odd
    mass(_mass),
    Umu(&Fgrid),
    UmuEven(&Hgrid),
    UmuOdd(&Hgrid),
    UUUmu(&Fgrid),
    UUUmuEven(&Hgrid),
    UUUmuOdd(&Hgrid),
    _tmp(&Hgrid),
    Dirichlet(0)
{
  GRID_ASSERT(_u0 != 0);

  int vol4;
  int LLs=1;
  c1=_c1;
  c2=_c2;
  u0=_u0;
  vol4= _grid->oSites();
  Stencil.BuildSurfaceList(LLs,vol4);
  vol4= _cbgrid->oSites();
  StencilEven.BuildSurfaceList(LLs,vol4);
  StencilOdd.BuildSurfaceList(LLs,vol4);

  // Dirichlet boundary conditions
  if (p.dirichlet.size() == Nd) {
    Coordinate block = p.dirichlet;
    if (block[0] or block[1] or block[2] or block[3]) {
      GRID_ASSERT(p.partialDirichlet == 0);
      std::cout << GridLogMessage 
                << " non-trivial Dirichlet boundary condition " 
                << block 
                << std::endl;
      Coordinate block = p.dirichlet;
      Dirichlet = 1;
      Block = block;
    }
  } else { Coordinate block(Nd, 0); Block = block; }
}

template <class Impl>
ImprovedStaggeredFermion<Impl>::ImprovedStaggeredFermion(GaugeField &_Uthin, GaugeField &_Ufat, GridCartesian &Fgrid,
							 GridRedBlackCartesian &Hgrid, RealD _mass,
							 RealD _c1, RealD _c2,RealD _u0,
							 const ImplParams &p)
  : ImprovedStaggeredFermion(Fgrid,Hgrid,_mass,_c1,_c2,_u0,p)
{ ImportGauge(_Uthin,_Ufat); }

////////////////////////////////////////////////////////////
// Momentum space propagator should be 
// https://arxiv.org/pdf/hep-lat/9712010.pdf
//
// mom space action.
//   gamma_mu i ( c1 sin pmu + c2 sin 3 pmu ) + m
//
// must track through staggered flavour/spin reduction in literature to 
// turn to free propagator for the one component chi field, a la page 4/5
// of above link to implmement fourier based solver.
////////////////////////////////////////////////////////////
template <class Impl>
void ImprovedStaggeredFermion<Impl>::ImportGaugeSimple(const GaugeField &_Utriple,const GaugeField &_Ufat) 
{
  /////////////////////////////////////////////////////////////////
  // Trivial import; phases and fattening and such like preapplied
  /////////////////////////////////////////////////////////////////
  GaugeLinkField U(GaugeGrid());

  for (int mu = 0; mu < Nd; mu++) {
    U = PeekIndex<LorentzIndex>(_Utriple, mu);
    PokeIndex<LorentzIndex>(UUUmu, U, mu );

    U = adj( Cshift(U, mu, -3));
    PokeIndex<LorentzIndex>(UUUmu, -U, mu+4 );

    U = PeekIndex<LorentzIndex>(_Ufat, mu);
    PokeIndex<LorentzIndex>(Umu, U, mu);

    U = adj( Cshift(U, mu, -1));
    PokeIndex<LorentzIndex>(Umu, -U, mu+4);
  }
  CopyGaugeCheckerboards();
}
template <class Impl>
void ImprovedStaggeredFermion<Impl>::ImportGaugeSimple(const DoubledGaugeField &_UUU,const DoubledGaugeField &_U) 
{
  Umu   = _U;
  UUUmu = _UUU;
  CopyGaugeCheckerboards();
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::CopyGaugeCheckerboards(void)
{
  pickCheckerboard(Even, UmuEven,  Umu);
  pickCheckerboard(Odd,  UmuOdd ,  Umu);
  pickCheckerboard(Even, UUUmuEven,UUUmu);
  pickCheckerboard(Odd,  UUUmuOdd, UUUmu);
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::ImportGauge(
  const GaugeField& _Ut, // three-hop port  <-+- ordering confusing for user
  const GaugeField& _Uf  // one-hop port    <-+
) {
  GaugeField _Uthin = _Ut;
  GaugeField _Ufat  = _Uf;
  GaugeLinkField U(GaugeGrid());

  ////////////////////////////////
  // Dirichlet boundary conditions
  ////////////////////////////////
  if (Dirichlet) {
    // Note: due to how we've implemented the Dirichlet boundary conditions 
    // (applying the Dirichlet mask to the input links, then using them to construct
    // the sparse matrix representation of the "improved" staggered operator), the 
    // depth of the zeroed-out region is one for the one-link ("fat") term and three
    // for the three-link ("thin"/"Naik") term
    this->validateDirichletBlock(GaugeGrid(), Block);
    this->applyDirichletMasks(_Uthin, Block);
    this->applyDirichletMasks(_Ufat, Block);
  }

  ////////////////////////////////////////////////////////
  // Double Store should take two fields for Naik and one hop separately.
  ////////////////////////////////////////////////////////
  Impl::ImprovedDoubleStore(GaugeGrid(), UUUmu, Umu, _Uthin, _Ufat);

  ////////////////////////////////////////////////////////
  // Apply scale factors to get the right fermion Kinetic term
  // Could pass coeffs into the double store to save work.
  // 0.5 ( U p(x+mu) - Udag(x-mu) p(x-mu) ) 
  ////////////////////////////////////////////////////////
  for (int mu = 0; mu < Nd; mu++) {
    U = PeekIndex<LorentzIndex>(Umu, mu);
    PokeIndex<LorentzIndex>(Umu, U*( 0.5*c1/u0), mu );
    
    U = PeekIndex<LorentzIndex>(Umu, mu+4);
    PokeIndex<LorentzIndex>(Umu, U*(-0.5*c1/u0), mu+4);

    U = PeekIndex<LorentzIndex>(UUUmu, mu);
    PokeIndex<LorentzIndex>(UUUmu, U*( 0.5*c2/u0/u0/u0), mu );
    
    U = PeekIndex<LorentzIndex>(UUUmu, mu+4);
    PokeIndex<LorentzIndex>(UUUmu, U*(-0.5*c2/u0/u0/u0), mu+4);
  }

  CopyGaugeCheckerboards();
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::ImportGauge(const ActionContract<GaugeField>& contract) {
  GRID_ASSERT(contract.first == this->identity());
  const auto& in = contract.second;
  conformable(in, _grid);
  if (in.size() == 1) { ImportGauge(in[0].resolve()); }
  else if (in.size() == 2) { ImportGauge(in[0].resolve(), in[1].resolve()); }
  else { GRID_ASSERT(0 && "invalid port count"); }
}

/////////////////////////////
// Implement the interface
/////////////////////////////

template <class Impl>
void ImprovedStaggeredFermion<Impl>::M(const FermionField &in, FermionField &out) 
{
  out.Checkerboard() = in.Checkerboard();
  Dhop(in, out, DaggerNo);
  axpy(out, mass, in, out);
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::Mdag(const FermionField &in, FermionField &out) 
{
  out.Checkerboard() = in.Checkerboard();
  Dhop(in, out, DaggerYes);
  axpy(out, mass, in, out);
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::Meooe(const FermionField &in, FermionField &out) 
{
  if (in.Checkerboard() == Odd) {
    DhopEO(in, out, DaggerNo);
  } else {
    DhopOE(in, out, DaggerNo);
  }
}
template <class Impl>
void ImprovedStaggeredFermion<Impl>::MeooeDag(const FermionField &in, FermionField &out) 
{
  if (in.Checkerboard() == Odd) {
    DhopEO(in, out, DaggerYes);
  } else {
    DhopOE(in, out, DaggerYes);
  }
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::Mooee(const FermionField &in, FermionField &out) 
{
  out.Checkerboard() = in.Checkerboard();
  typename FermionField::scalar_type scal(mass);
  out = scal * in;
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::MooeeDag(const FermionField &in, FermionField &out) 
{
  out.Checkerboard() = in.Checkerboard();
  Mooee(in, out);
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::MooeeInv(const FermionField &in, FermionField &out) 
{
  out.Checkerboard() = in.Checkerboard();
  out = (1.0 / (mass)) * in;
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::MooeeInvDag(const FermionField &in,FermionField &out) 
{
  out.Checkerboard() = in.Checkerboard();
  MooeeInv(in, out);
}

///////////////////////////////////
// Internal
///////////////////////////////////

template <class Impl>
void ImprovedStaggeredFermion<Impl>::DerivInternal(
  StencilImpl& st, 
  DoubledGaugeField& U, 
  DoubledGaugeField& UUU, 
	GaugeField& mat,
	const FermionField& A, 
  const FermionField& B, 
  int dag
) {
  /**
   * @brief Unprojected left-trivialized derivative of kinetic fermion bilinears
   * @author Curtis Taylor Peterson
   * @details
   * This method calculates the unprojected left-trivialized derivative of the kinetic
   * fermion bilinears directly from the doubled stores U and UUU, accumulating the
   * one-hop and three-hop contributions in mat.
   * 
   * See the tagged-field DerivInternal overload for the
   * shared notation and derivative conventions.
   *
   * link properties
   * ---------------
   *
   * The legacy path has the following restrictions:
   * - When the three-hop term is active, the elementary links entering both terms must
   *   be the same unitary field. This method recovers W from the doubled one-hop store
   *   U by removing its coefficient and phases. Recovering W requires c1 != 0 when 
   *   c2 != 0.
   * - The legacy DhopDerivEO and DhopDerivOE wrappers reject calls. A three-hop
   *   derivative contributes to both link checkerboards, which their single
   *   checkerboard output cannot represent. This implementation serves the full-grid
   *   legacy derivative; the opt-in path handles checkerboarded fermion arguments.
   */
  GRID_ASSERT(dag == DaggerNo or dag == DaggerYes);
  GRID_ASSERT((c1 != 0.0 or c2 == 0.0) && "need c1 != 0 or c2 == 0 to recover input links");

  mat = Zero();

  GridBase* GaugeGrid = U.Grid();
  FermionField Btilde(B.Grid());
  
  Compressor compressor;

  st.HaloExchange(B, compressor);

  if (c1 != 0.0) { // one-hop contribution; assumes unitary links
    for (int mu = 0; mu < Nd; ++mu) {
      Kernels::DhopDirForward(st, U, B, Btilde, mu, 0);
      pokeLorentz(mat, this->outer(Btilde, A), mu);
  } }

  if (c2 != 0.0) { // three-hop contribution; assumes unitary links
    GaugeLinkField w(GaugeGrid), cmp(GaugeGrid), deriv3(GaugeGrid);
    std::vector<ComplexField> eta(Nd, GaugeGrid);
    std::vector<ComplexField> bcs(Nd, GaugeGrid);
    
    this->setStaggeredPhases(eta);
    this->setBoundaryPhases(bcs);

    for (int mu = 0; mu < Nd; ++mu) {
      w = peekLorentz(U, mu) / (0.5*c1/u0*eta[mu]*bcs[mu]);

      Kernels::DhopDirForward(st, UUU, B, Btilde, mu, 1);
      deriv3 = this->outer(Btilde, A);

      cmp = Cshift(adj(w)*deriv3*w, mu, -1);
      deriv3 += cmp;

      cmp = Cshift(adj(w)*cmp*w, mu, -1);
      deriv3 += cmp;

      pokeLorentz(mat, peekLorentz(mat, mu) + deriv3, mu);
  } }

  if (dag == DaggerYes) { mat = -mat; }
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::DerivInternal(
  StencilImpl& stencil,
  PrimalCotangentPairs<GaugeField>& derivs,
  const DoubledGaugeField& U, // never used for computation
  const DoubledGaugeField& UUU,
  const FermionField& left, 
  const FermionField& right, 
  int dag
) {
  /**
   * @brief Forward Wirtinger derivative of kinetic fermion bilinears
   * @author Curtis Taylor Peterson
   * @details
   * This method calculates the forward Wirtinger derivative of fermion bilinears
   * (1) phi_{left}^dag M(m) phi_{right}
   * with m the bare fermion mass and
   * (2) M(m) = k1 D_{one-hop} + k2 D_{three-hop} + m,
   * where k1, k2 are prefactors multiplying the one-hop and three-hop contributions to
   * the "improved" staggered fermion operator. Recalling that the full pulled back force
   * is schematically of the form
   * (3) F = project_{Lie algebra}(-U dS/dU),
   * what is meant by the "forward Wirtinger derivative" is the dS/dU factor, which is a
   * matrix derivative with respect to the forward links, each component of which is an
   * ordinary Wirtinger derivative. This method calculates the forward Wirtinger
   * derivative with respect to the links that enter the one-hop (X) and three-hop (W)
   * terms of the improved staggered fermion operator. These derivatives are eventually
   * pulled back to the fundamental gauge field U (please excuse the confusing names given
   * to the one- and three-hop terms in the method signature) outside of this class.
   *
   * link properties
   * ---------------
   *
   * Note some important assumptions are made in this method that reflect both how this
   * operator is used in real QCD simulations and some simplifications that help deal with
   * Dirichlet boundary conditions:
   * - The links entering the one-hop term may be non-unitary. This does not mean that
   *   them being unitary will break anything. In practice, this doesn't actually matter
   *   (see below), but it is the reason why the opt-in interface works with Wirtinger
   *   derivatives as opposed to unprojected left-trivialized derivatives, which require
   *   removing the link factor to recover the raw derivative for the chain rule (an
   *   adjoint for unitary links; otherwise one has to swallow a numerically estimated
   *   inverse, perhaps by some Cayley-Hamilton procedure).
   * - The elementary links entering the three-hop term are assumed to be unitary. This is
   *   purely for convenience, as it allows us to propagate the Dirichlet boundary
   *   conditions already applied to "UUU" (i.e., "WWW") via successive Cartesian shifts
   *   and group conjugation operations (see below). The stored UUU includes coefficients,
   *   phases, and Dirichlet masks.
   *
   * one-hop derivative
   * ------------------
   *
   * The Wirtinger derivative with respect to the one-hop links is of the form
   * (4) d_{one-hop,mu}(n) = k1 m_{1-hop,mu}(n) phi_{right}(n + mu) phi_{left}^dag(n),
   * where m_{1-hop,mu}(n) contains the staggered phases, global boundary phases (which
   * take care of the periodic/anti-periodic boundary conditions), and possible Dirichlet
   * masks (which take care of Dirichlet boundary conditions).
   *
   * three-hop derivative
   * --------------------
   *
   * The forward three-hop contribution to the improved staggered operator is composed of
   * three links, which are assumed to be unitary:
   * (5) D_{three-hop,mu}(n -> n + 3 mu) = m_{3-hop,mu}(n)
   *                                       W_{mu}(n) W_{mu}(n + mu) W_{mu}(n + 2 mu),
   * with m_{3-hop,mu}(n) containing the staggered phases, global boundary phases, and
   * possible Dirichlet masks. The width of the three-hop Dirichlet mask is 3 lattice
   * units; see the two-port ImportGauge method for details. The three-hop derivative is 
   * composed of three parts by the product rule. The unitarity of the links entering the 
   * three-hop term simplifies two parts of the calculation. First, it allows me to 
   * calculate the unprojected left-trivialized derivative, then convert it to the 
   * Wirtinger derivative by multiplying on the left by W^dag. This is nice because, 
   * second, each contribution K_{mu}^{(j)} to the three-hop derivative can be obtained 
   * from K_{mu}^{(j-1)} as
   * (6) K_{mu}^{(j)} = BackwardCartesianShift_{mu}[ W_{mu}^{dag} K_{mu}^{(j-1)} W_{mu} ]
   * with
   * (7) K_{mu}^{(0)}(n) = m_{3-hop,mu}(n)
   *                       W_{mu}(n) W_{mu}(n + mu) W_{mu}(n + 2 mu)
   *                       phi_{right}(n + 3 mu) phi_{left}^dag(n).
   * The full three-hop derivative in direction mu is then
   * (8) d_{three-hop,mu}(n) = k2 W_{mu}^{dag}(n) sum_{j=0}^{2} K_{mu}^{(j)}(n).
   * There is one very important benefit that this approach provides outside of it just
   * being cute: it allows us to propagate the Dirichlet masks and global boundary phases
   * that are already folded into the first four components of the DoubledGaugeField
   * constituting the sparse matrix representation of the three-hop term into each
   * derivative term without having to explicitly apply them manually, like we do for the
   * one-hop term; the staggered phases also conveniently propagate through the shifts.
   * Quite neat, isn't it?
   *
   * derivative of conjugate operator
   * --------------------------------
   *
   * One has
   * (9) M^dag(m) = -M(-m)
   * for staggered Dirac operators. As such, the forward derivative of M^dag can be 
   * obtained from the forward derivative of M by multiplying by -1 (the derivative 
   * knocks out the constant mass term).
   */
  GRID_ASSERT(dag == DaggerNo || dag == DaggerYes);
  GRID_ASSERT(derivs.size() == 1 || derivs.size() == 2);
  
  conformable(derivs, _grid);

  const std::size_t oneHopPort = derivs.size() - 1; // single-port branch compatibility
  const std::size_t threeHopPort = 0;

  GridBase* GaugeGrid = U.Grid();
  GridBase* FermionGrid = left.Grid();
  
  FermionField leftTilde(FermionGrid);
  FermionField rightTilde = right;

  Compressor compressor;

  stencil.HaloExchange(right, compressor);

  derivs = Zero();

  if (c1 != 0.0) {
    GaugeField X(GaugeGrid);

    // staggered and boundary phases folded in m_{1-hop,mu}(n); Eqn (4)
    X.Checkerboard() = U.Checkerboard();
    X = 0.5*c1/u0;
    this->rephase(_grid, X);

    // one-hop Wirtinger derivative; Eqn (4)
    for (int mu = 0; mu < Nd; ++mu) {
      Kernels::DhopDirForward(stencil, X, right, rightTilde, mu, 0);
      pokeLorentz(derivs[oneHopPort], this->outer(rightTilde, left), mu);
    }

    // reimposition of Dirichlet masks contributing to m_{1-hop,mu}(n); Eqn (4)
    if (Dirichlet) { this->applyDirichletMasks(derivs[oneHopPort], Block); }
  }

  if (c2 != 0.0) {
    const GaugeField& W = derivs[threeHopPort].gauge();
    GaugeLinkField cmp(_grid);
    PrimalCotangentPair<GaugeField> threeHop(derivs[threeHopPort].primal());

    for (int mu = 0; mu < Nd; ++mu) {
      GaugeLinkField w = peekLorentz(W, mu);

      // first term - same structure as one-hop contribution; Eqn (7)
      Kernels::DhopDirForward(stencil, UUU, right, rightTilde, mu, 1);
      pokeLorentz(threeHop, this->outer(rightTilde, left), mu);
      GaugeLinkField component = peekLorentz(threeHop, mu);

      // second term - requires communication, but necessary; Eqn (6)
      cmp = Cshift(adj(w)*component*w, mu, -1);
      component += cmp;

      // third term - requires communication, but necessary; Eqn (6)
      cmp = Cshift(adj(w)*cmp*w, mu, -1);
      component += cmp;

      pokeLorentz(threeHop, adj(w)*component, mu);
    }
    // accumulate three-hop contributions into the main derivative object; Eqn (8)
    derivs[threeHopPort] += threeHop;
  }

  if (dag == DaggerYes) { derivs *= -1; }
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::DhopDeriv(GaugeField &mat, const FermionField &U, const FermionField &V, int dag) 
{
  conformable(U.Grid(), _grid);
  conformable(U.Grid(), V.Grid());
  conformable(U.Grid(), mat.Grid());

  mat.Checkerboard() = U.Checkerboard();

  DerivInternal(Stencil, Umu, UUUmu, mat, U, V, dag);
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::DhopDeriv(
  PrimalCotangentPairs<GaugeField>& derivs,
  const FermionField& left,
  const FermionField& right,
  int dag
) {
  conformable(left.Grid(), _grid);
  conformable(left.Grid(), right.Grid());
  DerivInternal(Stencil, derivs, Umu, UUUmu, left, right, dag);
} 

template <class Impl>
void ImprovedStaggeredFermion<Impl>::DhopDerivOE(
  GaugeField& mat, 
  const FermionField& U, 
  const FermionField& V, 
  int dag
) {
  // This path is rejected because "mat" is restricted to a single checkerboard by 
  // construction, but the three-hop derivative contributes to both checkerboards
  GRID_ASSERT(0 && "strict checkerboarded derivative not defined for three-hop term"); 
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::DhopDerivOE(
  PrimalCotangentPairs<GaugeField>& derivs,
  const FermionField& left,
  const FermionField& right,
  int dag
) {
  // We do not reject this path because opt-in interface is not strictly checkerboarded,
  // unlike the legacy interface; though input fields can occupy single checkerboard, 
  // the derivative is accumulated into a full field. As such, the three-hop derivative 
  // occupying multiple checkerboards is handled correctly.
  GRID_ASSERT(left.Checkerboard() == Odd);
  GRID_ASSERT(right.Checkerboard() == Even);

  conformable(left.Grid(), _cbgrid);
  conformable(left.Grid(), right.Grid());
  DerivInternal(StencilEven, derivs, UmuOdd, UUUmuOdd, left, right, dag);
} 

template <class Impl>
void ImprovedStaggeredFermion<Impl>::DhopDerivEO(
  GaugeField& mat, 
  const FermionField& U, 
  const FermionField& V, 
  int dag
) { 
  // This path is rejected because "mat" is restricted to a single checkerboard by 
  // construction, but the three-hop derivative contributes to both checkerboards
  GRID_ASSERT(0 && "strict checkerboarded derivative not defined for three-hop term"); 
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::DhopDerivEO(
  PrimalCotangentPairs<GaugeField>& derivs,
  const FermionField& left,
  const FermionField& right,
  int dag
) {
  // We do not reject this path because opt-in interface is not strictly checkerboarded,
  // unlike the legacy interface; though input fields can occupy single checkerboard, 
  // the derivative is accumulated into a full field. As such, the three-hop derivative 
  // occupying multiple checkerboards is handled correctly.
  GRID_ASSERT(left.Checkerboard() == Even);
  GRID_ASSERT(right.Checkerboard() == Odd);

  conformable(left.Grid(), _cbgrid);
  conformable(left.Grid(), right.Grid());
  DerivInternal(StencilOdd, derivs, UmuEven, UUUmuEven, left, right, dag);
} 

template <class Impl>
void ImprovedStaggeredFermion<Impl>::Dhop(const FermionField &in, FermionField &out, int dag) 
{
  conformable(in.Grid(), _grid);  // verifies full grid
  conformable(in.Grid(), out.Grid());

  out.Checkerboard() = in.Checkerboard();

  DhopInternal(Stencil, Umu, UUUmu, in, out, dag);
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::DhopOE(const FermionField &in, FermionField &out, int dag) 
{
  conformable(in.Grid(), _cbgrid);    // verifies half grid
  conformable(in.Grid(), out.Grid());  // drops the cb check

  GRID_ASSERT(in.Checkerboard() == Even);
  out.Checkerboard() = Odd;

  DhopInternal(StencilEven, UmuOdd, UUUmuOdd, in, out, dag);
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::DhopEO(const FermionField &in, FermionField &out, int dag) 
{
  conformable(in.Grid(), _cbgrid);    // verifies half grid
  conformable(in.Grid(), out.Grid());  // drops the cb check

  GRID_ASSERT(in.Checkerboard() == Odd);
  out.Checkerboard() = Even;

  DhopInternal(StencilOdd, UmuEven, UUUmuEven, in, out, dag);
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::Mdir(const FermionField &in, FermionField &out, int dir, int disp) 
{
  DhopDir(in, out, dir, disp);
}
template <class Impl>
void ImprovedStaggeredFermion<Impl>::MdirAll(const FermionField &in, std::vector<FermionField> &out) 
{
  GRID_ASSERT(0); // Not implemented yet
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::DhopDir(const FermionField &in, FermionField &out, int dir, int disp) 
{

  Compressor compressor;
  Stencil.HaloExchange(in, compressor);
  autoView( Umu_v   ,   Umu, CpuRead);
  autoView( UUUmu_v , UUUmu, CpuRead);
  autoView( in_v    ,  in, CpuRead);
  autoView( out_v   , out, CpuWrite);
  /*
  thread_for( sss, in.Grid()->oSites(),{
    Kernels::DhopDirKernel(Stencil, Umu_v, UUUmu_v, Stencil.CommBuf(), sss, sss, in_v, out_v, dir, disp);
  });
  */
};


template <class Impl>
void ImprovedStaggeredFermion<Impl>::DhopInternal(StencilImpl &st, 
						  DoubledGaugeField &U,
						  DoubledGaugeField &UUU,
						  const FermionField &in,
						  FermionField &out, int dag) 
{
  if ( StaggeredKernelsStatic::Comms == StaggeredKernelsStatic::CommsAndCompute )
    DhopInternalOverlappedComms(st,U,UUU,in,out,dag);
  else
    DhopInternalSerialComms(st,U,UUU,in,out,dag);
}
template <class Impl>
void ImprovedStaggeredFermion<Impl>::DhopInternalOverlappedComms(StencilImpl &st, 
								 DoubledGaugeField &U,
								 DoubledGaugeField &UUU,
								 const FermionField &in,
								 FermionField &out, int dag) 
{
  Compressor compressor; 
  int len =  U.Grid()->oSites();

  st.Prepare();
  st.HaloGather(in,compressor);

  std::vector<std::vector<CommsRequest_t> > requests;
  st.CommunicateBegin(requests);

  st.CommsMergeSHM(compressor);

  //////////////////////////////////////////////////////////////////////////////////////////////////////
  // Removed explicit thread comms
  //////////////////////////////////////////////////////////////////////////////////////////////////////
  {
    int interior=1;
    int exterior=0;
    Kernels::DhopImproved(st,U,UUU,in,out,dag,interior,exterior);
  }

  st.CommunicateComplete(requests);

  // First to enter, last to leave timing
  st.CommsMerge(compressor);

  {
    int interior=0;
    int exterior=1;
    Kernels::DhopImproved(st,U,UUU,in,out,dag,interior,exterior);
  }
}


template <class Impl>
void ImprovedStaggeredFermion<Impl>::DhopInternalSerialComms(StencilImpl &st, 
							     DoubledGaugeField &U,
							     DoubledGaugeField &UUU,
							     const FermionField &in,
							     FermionField &out, int dag) 
{
  GRID_ASSERT((dag == DaggerNo) || (dag == DaggerYes));

  Compressor compressor;
  st.HaloExchange(in, compressor);

  {
    int interior=1;
    int exterior=1;
    Kernels::DhopImproved(st,U,UUU,in,out,dag,interior,exterior);
  }
};

//////////////////////////////////////////////////////// 
// Conserved current - not yet implemented.
////////////////////////////////////////////////////////
template <class Impl>
void ImprovedStaggeredFermion<Impl>::ContractConservedCurrent(PropagatorField &q_in_1,
							      PropagatorField &q_in_2,
							      PropagatorField &q_out,
							      PropagatorField &src,
							      Current curr_type,
							      unsigned int mu)
{
  GRID_ASSERT(0);
}

template <class Impl>
void ImprovedStaggeredFermion<Impl>::SeqConservedCurrent(PropagatorField &q_in,
                                                         PropagatorField &q_out,
                                                         PropagatorField &src,
                                                         Current curr_type,
                                                         unsigned int mu, 
                                                         unsigned int tmin,
                                              unsigned int tmax,
					      ComplexField &lattice_cmplx)
{
  GRID_ASSERT(0);

}

NAMESPACE_END(Grid);
