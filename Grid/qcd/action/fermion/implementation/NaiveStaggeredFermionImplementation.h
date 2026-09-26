/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/qcd/action/fermion/NaiveStaggeredFermion.cc

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
NaiveStaggeredFermion<Impl>::NaiveStaggeredFermion(GridCartesian &Fgrid, GridRedBlackCartesian &Hgrid, 
						   RealD _mass,
						   RealD _c1, RealD _u0,
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
    _tmp(&Hgrid),
    Dirichlet(0)
{
  int vol4;
  int LLs=1;
  c1=_c1;
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
      std::cout << GridLogMessage << " NaiveStaggeredFermion: non-trivial Dirichlet boundary condition " << block << std::endl;
      Dirichlet = 1;
      Block = block;
    }
  } else { Coordinate block(Nd, 0); Block = block; }
}

template <class Impl>
NaiveStaggeredFermion<Impl>::NaiveStaggeredFermion(GaugeField &_U, GridCartesian &Fgrid,
						   GridRedBlackCartesian &Hgrid, RealD _mass,
						   RealD _c1, RealD _u0,
						   const ImplParams &p)
  : NaiveStaggeredFermion(Fgrid,Hgrid,_mass,_c1,_u0,p)
{
  ImportGauge(_U);
}

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
void NaiveStaggeredFermion<Impl>::CopyGaugeCheckerboards(void)
{
  pickCheckerboard(Even, UmuEven,  Umu);
  pickCheckerboard(Odd,  UmuOdd ,  Umu);
}
template <class Impl>
void NaiveStaggeredFermion<Impl>::ImportGauge(const GaugeField &Uin) 
{
  GaugeField _U = Uin;
  GaugeLinkField U(GaugeGrid());
  DoubledGaugeField _UUU(GaugeGrid());

  ////////////////////////////////
  // Dirichlet boundary conditions
  ////////////////////////////////
  if (Dirichlet) { 
    this->validateDirichletBlock(GaugeGrid(), Block); 
    this->applyDirichletMasks(_U, Block);
  }

  ////////////////////////////////////////////////////////
  // Double Store should take two fields for Naik and one hop separately.
  // Discard teh Naik as Naive
  ////////////////////////////////////////////////////////
  Impl::NaiveDoubleStore(GaugeGrid(), Umu, _U);

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

  }

  CopyGaugeCheckerboards();
}

template <class Impl>
void NaiveStaggeredFermion<Impl>::ImportGauge(const ActionContract<GaugeField>& contract) {
  GRID_ASSERT(contract.first == this->identity());
  const auto& in = contract.second;
  conformable(in, _grid);
  if (in.size() == 1) { ImportGauge(in[0].resolve()); }
  else { GRID_ASSERT(0 && "invalid port count"); }
}


/////////////////////////////
// Implement the interface
/////////////////////////////

template <class Impl>
void NaiveStaggeredFermion<Impl>::M(const FermionField &in, FermionField &out) {
  out.Checkerboard() = in.Checkerboard();
  Dhop(in, out, DaggerNo);
  axpy(out, mass, in, out);
}

template <class Impl>
void NaiveStaggeredFermion<Impl>::Mdag(const FermionField &in, FermionField &out) {
  out.Checkerboard() = in.Checkerboard();
  Dhop(in, out, DaggerYes);
  axpy(out, mass, in, out);
}

template <class Impl>
void NaiveStaggeredFermion<Impl>::Meooe(const FermionField &in, FermionField &out) {
  if (in.Checkerboard() == Odd) {
    DhopEO(in, out, DaggerNo);
  } else {
    DhopOE(in, out, DaggerNo);
  }
}
template <class Impl>
void NaiveStaggeredFermion<Impl>::MeooeDag(const FermionField &in, FermionField &out) {
  if (in.Checkerboard() == Odd) {
    DhopEO(in, out, DaggerYes);
  } else {
    DhopOE(in, out, DaggerYes);
  }
}

template <class Impl>
void NaiveStaggeredFermion<Impl>::Mooee(const FermionField &in, FermionField &out) {
  out.Checkerboard() = in.Checkerboard();
  typename FermionField::scalar_type scal(mass);
  out = scal * in;
}

template <class Impl>
void NaiveStaggeredFermion<Impl>::MooeeDag(const FermionField &in, FermionField &out) {
  out.Checkerboard() = in.Checkerboard();
  Mooee(in, out);
}

template <class Impl>
void NaiveStaggeredFermion<Impl>::MooeeInv(const FermionField &in, FermionField &out) {
  out.Checkerboard() = in.Checkerboard();
  out = (1.0 / (mass)) * in;
}

template <class Impl>
void NaiveStaggeredFermion<Impl>::MooeeInvDag(const FermionField &in, FermionField &out) 
{
  out.Checkerboard() = in.Checkerboard();
  MooeeInv(in, out);
}

///////////////////////////////////
// Internal
///////////////////////////////////

template <class Impl>
void NaiveStaggeredFermion<Impl>::DerivInternal(
  StencilImpl& st, 
  DoubledGaugeField& U,
	GaugeField& mat,
	const FermionField& A, 
  const FermionField& B, 
  int dag
) {
  /**
   * @brief Unprojected left-trivialized derivative of kinetic fermion bilinears
   * @author Curtis Taylor Peterson
   * @details See the tagged-field DerivInternal overload below
   */
  GRID_ASSERT((dag == DaggerNo) || (dag == DaggerYes));

  FermionField Btilde(B.Grid());
  FermionField Atilde = A;

  Compressor compressor;

  st.HaloExchange(B, compressor);

  for (int mu = 0; mu < Nd; ++mu) {
    Kernels::DhopDirForward(st, U, B, Btilde, mu, 0);
    pokeLorentz(mat, this->outer(Btilde, Atilde), mu);
  }

  if (dag) { mat = -mat; }
}

template <class Impl>
void NaiveStaggeredFermion<Impl>::DerivInternal(
  StencilImpl& stencil,
  PrimalCotangentPairs<GaugeField>& derivs,
  const DoubledGaugeField& U, // never used for computation
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
   * (2) M(m) = k1 D + m,
   * where k1 is the prefactor multiplying the massless naive staggered fermion operator. 
   * Recalling that the full pulled back force is schematically of the form
   * (3) F = project_{Lie algebra}(-U dS/dU),
   * what is meant by the "forward Wirtinger derivative" is the dS/dU factor, which is a
   * matrix derivative with respect to the forward links, each component of which is an
   * ordinary Wirtinger derivative. The Wirtinger derivative is eventually pulled back to 
   * the fundamental gauge field U outside of this class.
   *
   * link properties
   * ---------------
   *
   * The links entering the one-hop term may be non-unitary. This does not mean that
   * them being unitary will break anything. In practice, this doesn't actually matter
   * (see below), but it is the reason why the opt-in interface works with Wirtinger
   * derivatives as opposed to unprojected left-trivialized derivatives, which require
   * removing the link factor to recover the raw derivative for the chain rule.
   *
   * derivative
   * ----------
   *
   * The Wirtinger derivative with respect to the one-hop links is of the form
   * (4) d_{mu}(n) = k1 m_{mu}(n) phi_{right}(n + mu) phi_{left}^dag(n),
   * where m_{mu}(n) contains the staggered phases, global boundary phases (which
   * take care of the periodic/anti-periodic boundary conditions), and possible Dirichlet
   * masks (which take care of Dirichlet boundary conditions).
   *
   * derivative of conjugate operator
   * --------------------------------
   *
   * One has
   * (5) M^dag(m) = -M(-m)
   * for staggered Dirac operators. As such, the forward derivative of M^dag can be
   * obtained from the forward derivative of M by multiplying by -1 (the derivative
   * knocks out the constant mass term).
   */
  GRID_ASSERT(dag == DaggerNo || dag == DaggerYes);
  GRID_ASSERT(derivs.size() == 1);
  
  conformable(derivs, _grid);

  GridBase* GaugeGrid = U.Grid();
  GridBase* FermionGrid = left.Grid();
  
  GaugeField X(GaugeGrid);
  FermionField leftTilde(FermionGrid);
  FermionField rightTilde = right;

  Compressor compressor;

  stencil.HaloExchange(right, compressor);

  derivs = Zero();

  X.Checkerboard() = U.Checkerboard();
  X = 0.5*c1/u0;
  this->rephase(_grid, X);

  // Wirtinger derivative; Eqn (4)
  for (int mu = 0; mu < Nd; ++mu) {
    Kernels::DhopDirForward(stencil, X, right, rightTilde, mu, 0);
    pokeLorentz(derivs[0], this->outer(rightTilde, left), mu);
  }

  // finish off by applying Dirichlet masks and daggering if requested
  if (Dirichlet) { this->applyDirichletMasks(derivs[0], Block); }
  if (dag == DaggerYes) { derivs *= -1; }
}

template <class Impl>
void NaiveStaggeredFermion<Impl>::DhopDeriv(
  GaugeField &mat, 
  const FermionField &U, 
  const FermionField &V, 
  int dag
) {
  conformable(U.Grid(), _grid);
  conformable(U.Grid(), V.Grid());
  conformable(U.Grid(), mat.Grid());

  mat.Checkerboard() = U.Checkerboard();

  DerivInternal(Stencil, Umu, mat, U, V, dag);
}

template <class Impl>
void NaiveStaggeredFermion<Impl>::DhopDeriv(
  PrimalCotangentPairs<GaugeField>& derivs,
  const FermionField& left,
  const FermionField& right,
  int dag
) {
  conformable(left.Grid(), _grid);
  conformable(left.Grid(), right.Grid());
  DerivInternal(Stencil, derivs, Umu, left, right, dag);
}

template <class Impl>
void NaiveStaggeredFermion<Impl>::DhopDerivOE(GaugeField &mat, const FermionField &U, const FermionField &V, int dag) {

  conformable(U.Grid(), _cbgrid);
  conformable(U.Grid(), V.Grid());
  mat.reset(_cbgrid);

  GRID_ASSERT(V.Checkerboard() == Even);
  GRID_ASSERT(U.Checkerboard() == Odd);
  mat.Checkerboard() = Odd;

  DerivInternal(StencilEven, UmuOdd, mat, U, V, dag);
}

template <class Impl>
void NaiveStaggeredFermion<Impl>::DhopDerivOE(
  PrimalCotangentPairs<GaugeField>& derivs,
  const FermionField& left,
  const FermionField& right,
  int dag
) {
  GRID_ASSERT(left.Checkerboard() == Odd);
  GRID_ASSERT(right.Checkerboard() == Even);

  conformable(left.Grid(), _cbgrid);
  conformable(left.Grid(), right.Grid());
  DerivInternal(StencilEven, derivs, UmuOdd, left, right, dag);
} 

template <class Impl>
void NaiveStaggeredFermion<Impl>::DhopDerivEO(GaugeField &mat, const FermionField &U, const FermionField &V, int dag) {

  conformable(U.Grid(), _cbgrid);
  conformable(U.Grid(), V.Grid());
  mat.reset(_cbgrid);

  GRID_ASSERT(V.Checkerboard() == Odd);
  GRID_ASSERT(U.Checkerboard() == Even);
  mat.Checkerboard() = Even;

  DerivInternal(StencilOdd, UmuEven, mat, U, V, dag);
}

template <class Impl>
void NaiveStaggeredFermion<Impl>::DhopDerivEO(
  PrimalCotangentPairs<GaugeField>& derivs,
  const FermionField& left,
  const FermionField& right,
  int dag
) {
  GRID_ASSERT(left.Checkerboard() == Even);
  GRID_ASSERT(right.Checkerboard() == Odd);

  conformable(left.Grid(), _cbgrid);
  conformable(left.Grid(), right.Grid());
  DerivInternal(StencilOdd, derivs, UmuEven, left, right, dag);
} 

template <class Impl>
void NaiveStaggeredFermion<Impl>::Dhop(const FermionField &in, FermionField &out, int dag) 
{
  conformable(in.Grid(), _grid);  // verifies full grid
  conformable(in.Grid(), out.Grid());

  out.Checkerboard() = in.Checkerboard();

  DhopInternal(Stencil, Umu, in, out, dag);
}

template <class Impl>
void NaiveStaggeredFermion<Impl>::DhopOE(const FermionField &in, FermionField &out, int dag) 
{
  conformable(in.Grid(), _cbgrid);    // verifies half grid
  conformable(in.Grid(), out.Grid());  // drops the cb check

  GRID_ASSERT(in.Checkerboard() == Even);
  out.Checkerboard() = Odd;

  DhopInternal(StencilEven, UmuOdd, in, out, dag);
}

template <class Impl>
void NaiveStaggeredFermion<Impl>::DhopEO(const FermionField &in, FermionField &out, int dag) 
{
  conformable(in.Grid(), _cbgrid);    // verifies half grid
  conformable(in.Grid(), out.Grid());  // drops the cb check

  GRID_ASSERT(in.Checkerboard() == Odd);
  out.Checkerboard() = Even;

  DhopInternal(StencilOdd, UmuEven, in, out, dag);
}

template <class Impl>
void NaiveStaggeredFermion<Impl>::Mdir(const FermionField &in, FermionField &out, int dir, int disp) 
{
  DhopDir(in, out, dir, disp);
}
template <class Impl>
void NaiveStaggeredFermion<Impl>::MdirAll(const FermionField &in, std::vector<FermionField> &out) 
{
  GRID_ASSERT(0); // Not implemented yet
}

template <class Impl>
void NaiveStaggeredFermion<Impl>::DhopDir(const FermionField &in, FermionField &out, int dir, int disp) 
{

  Compressor compressor;
  Stencil.HaloExchange(in, compressor);
  autoView( Umu_v   ,  Umu, CpuRead);
  autoView( in_v    ,  in, CpuRead);
  autoView( out_v   , out, CpuWrite);
  //  thread_for( sss, in.Grid()->oSites(),{
  //    Kernels::DhopDirKernel(Stencil, Umu_v, Stencil.CommBuf(), sss, sss, in_v, out_v, dir, disp);
  //  });
  GRID_ASSERT(0);
};


template <class Impl>
void NaiveStaggeredFermion<Impl>::DhopInternal(StencilImpl &st,
					       DoubledGaugeField &U,
					       const FermionField &in,
					       FermionField &out, int dag) 
{
  if ( StaggeredKernelsStatic::Comms == StaggeredKernelsStatic::CommsAndCompute )
    DhopInternalOverlappedComms(st,U,in,out,dag);
  else
    DhopInternalSerialComms(st,U,in,out,dag);
}
template <class Impl>
void NaiveStaggeredFermion<Impl>::DhopInternalOverlappedComms(StencilImpl &st,
							      DoubledGaugeField &U,
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
    Kernels::DhopNaive(st,U,in,out,dag,interior,exterior);
  }

  st.CommunicateComplete(requests);

  // First to enter, last to leave timing
  st.CommsMerge(compressor);

  {
    int interior=0;
    int exterior=1;
    Kernels::DhopNaive(st,U,in,out,dag,interior,exterior);
  }
}

template <class Impl>
void NaiveStaggeredFermion<Impl>::DhopInternalSerialComms(StencilImpl &st,
							  DoubledGaugeField &U,
							  const FermionField &in,
							  FermionField &out, int dag) 
{
  GRID_ASSERT((dag == DaggerNo) || (dag == DaggerYes));

  Compressor compressor;
  st.HaloExchange(in, compressor);

  {
    int interior=1;
    int exterior=1;
    Kernels::DhopNaive(st,U,in,out,dag,interior,exterior);
  }
};

//////////////////////////////////////////////////////// 
// Conserved current - not yet implemented.
////////////////////////////////////////////////////////
template <class Impl>
void NaiveStaggeredFermion<Impl>::ContractConservedCurrent(PropagatorField &q_in_1,
							      PropagatorField &q_in_2,
							      PropagatorField &q_out,
							      PropagatorField &src,
							      Current curr_type,
							      unsigned int mu)
{
  GRID_ASSERT(0);
}

template <class Impl>
void NaiveStaggeredFermion<Impl>::SeqConservedCurrent(PropagatorField &q_in,
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
