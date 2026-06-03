/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/qcd/action/fermion/FermionOperatorImpl.h

Copyright (C) 2015

Author: Peter Boyle <pabobyle@ph.ed.ac.uk>

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
#pragma once

NAMESPACE_BEGIN(Grid);

template <class S, class Representation = FundamentalRepresentation >
class StaggeredImpl : public PeriodicGaugeImpl<GaugeImplTypes<S, Representation::Dimension > > 
{

public:

  typedef RealD  _Coeff_t ;
  static const int Dimension = Representation::Dimension;
  static const bool isFundamental = Representation::isFundamental;
  static const bool LsVectorised=false;
  typedef PeriodicGaugeImpl<GaugeImplTypes<S, Dimension > > Gimpl;
      
  //Necessary?
  constexpr bool is_fundamental() const{return Dimension == Nc ? 1 : 0;}
    
  typedef _Coeff_t Coeff_t;

  INHERIT_GIMPL_TYPES(Gimpl);
      
  template <typename vtype> using iImplSpinor            = iScalar<iScalar<iVector<vtype, Dimension> > >;
  template <typename vtype> using iImplHalfSpinor        = iScalar<iScalar<iVector<vtype, Dimension> > >;
  template <typename vtype> using iImplDoubledGaugeField = iVector<iScalar<iMatrix<vtype, Dimension> >, Nds>;
  template <typename vtype> using iImplPropagator        = iScalar<iScalar<iMatrix<vtype, Dimension> > >;
    
  typedef iImplSpinor<Simd>            SiteSpinor;
  typedef iImplHalfSpinor<Simd>        SiteHalfSpinor;
  typedef iImplDoubledGaugeField<Simd> SiteDoubledGaugeField;
  typedef iImplPropagator<Simd>        SitePropagator;
    
  typedef Lattice<SiteSpinor>            FermionField;
  typedef Lattice<SiteDoubledGaugeField> DoubledGaugeField;
  typedef Lattice<SitePropagator> PropagatorField;
    
  typedef StaggeredImplParams ImplParams;
  typedef SimpleCompressor<SiteSpinor> Compressor;
  typedef CartesianStencil<SiteSpinor, SiteSpinor, ImplParams> StencilImpl;
  typedef typename StencilImpl::View_type StencilView;

  ImplParams Params;
    
  StaggeredImpl(const ImplParams &p = ImplParams()) : Params(p){};
      
  template<class _Spinor>
  static accelerator_inline void multLink(_Spinor &phi,
		       const SiteDoubledGaugeField &U,
		       const _Spinor &chi,
		       int mu)
  {
    auto UU = coalescedRead(U(mu));
    mult(&phi(), &UU, &chi());
  }
  template<class _Spinor>
  static accelerator_inline void multLinkAdd(_Spinor &phi,
			  const SiteDoubledGaugeField &U,
			  const _Spinor &chi,
			  int mu)
  {
    auto UU = coalescedRead(U(mu));
    mac(&phi(), &UU, &chi());
  }
      
  template <class ref>
  static accelerator_inline void loadLinkElement(Simd &reg, ref &memory) 
  {
    reg = memory;
  }
      
    inline void InsertGaugeField(DoubledGaugeField &U_ds,
				 const GaugeLinkField &U,int mu)
    {
      PokeIndex<LorentzIndex>(U_ds, U, mu);
    }
  inline void DoubleStore(
    GridBase *GaugeGrid,
		DoubledGaugeField &UUUds, // for Naik term
		DoubledGaugeField &Uds,
		const GaugeField &Uthin,
		const GaugeField &Ufat
  ) {
    /**
     * @brief Staggered double store
     * @author Curtis Taylor Peterson, Peter Boyle
     * @details
     * This code follows the "MILC convention", which treats the fourth 
     * direction as the "time" coordinate:
     * (2a) eta_0 = (-1)^{x3}       <---| 
     * (2b) eta_1 = (-1)^{x3+x0}        | convention in
     * (2c) eta_2 = (-1)^{x3+x0+x1}     | this code
     * (2d) eta_3 = 1               <---|
     * Though awkard, this convention follows that of most texts on
     * relativity. This is opposed to the convention that one often finds in 
     * lattice gauge theory textbooks, where the "time" direction is x0:
     * (3a) eta_0 = 1
     * (3b) eta_1 = (-1)^{x0}
     * (3c) eta_2 = (-1)^{x0+x1}
     * (3d) eta_3 = (-1)^{x0+x1+x2}
     * Boundary conditions are imposed by "rephasing" the links 
     * on the boundary, with "1" for periodic and "-1" for anti-periodic.
     */
    typedef typename Simd::scalar_type GridScalar;

    conformable(Uds.Grid(), GaugeGrid);
    conformable(Uthin.Grid(), GaugeGrid);
    conformable(Ufat.Grid(), GaugeGrid);

    GaugeLinkField U(GaugeGrid);
    GaugeLinkField UU(GaugeGrid);
    GaugeLinkField UUU(GaugeGrid);
    GaugeLinkField Udag(GaugeGrid);
    GaugeLinkField UUUdag(GaugeGrid);

    Lattice<iScalar<vInteger>> x(GaugeGrid), y(GaugeGrid), t(GaugeGrid);
    Lattice<iScalar<vInteger>> tx(GaugeGrid), txy(GaugeGrid), xyzt(GaugeGrid);

    LatticeCoordinate(x, 0);
    LatticeCoordinate(y, 1);
    LatticeCoordinate(t, 3);
    tx = t + x;
    txy = tx + y;

    for (int mu = 0; mu < Nd; mu++) {
      /*
      // Staggered Phase.
      Lattice<iScalar<vInteger> > coor(GaugeGrid);
      Lattice<iScalar<vInteger> > x(GaugeGrid); LatticeCoordinate(x,0);
      Lattice<iScalar<vInteger> > y(GaugeGrid); LatticeCoordinate(y,1);
      Lattice<iScalar<vInteger> > z(GaugeGrid); LatticeCoordinate(z,2);
      Lattice<iScalar<vInteger> > t(GaugeGrid); LatticeCoordinate(t,3);

      Lattice<iScalar<vInteger> > lin_z(GaugeGrid); lin_z=x+y;
      Lattice<iScalar<vInteger> > lin_t(GaugeGrid); lin_t=x+y+z;

      ComplexField phases(GaugeGrid);	phases=1.0;

      if ( mu == 1 ) phases = where( mod(x    ,2)==(Integer)0, phases,-phases);
      if ( mu == 2 ) phases = where( mod(lin_z,2)==(Integer)0, phases,-phases);
      if ( mu == 3 ) phases = where( mod(lin_t,2)==(Integer)0, phases,-phases);
      */

      ComplexField bcs(GaugeGrid), phases(GaugeGrid);
      Lattice<iScalar<vInteger>> coor(GaugeGrid);
      int N = GaugeGrid->GlobalDimensions()[mu] - 1;
      auto bpha = Params.boundary_phases[mu];
      GridScalar dirichlet(real(bpha), imag(bpha));

      // staggered phases
      phases = 1.0;
      if (mu == 0) { phases = where(mod(t, 2) == (Integer)0,   phases, -phases); }
      if (mu == 1) { phases = where(mod(tx, 2) == (Integer)0,  phases, -phases); }
      if (mu == 2) { phases = where(mod(txy, 2) == (Integer)0, phases, -phases); }

      // boundary conditions
      bcs = 1.0;
      LatticeCoordinate(coor, mu);
      bcs = where(coor == (Integer)N, dirichlet*bcs, bcs);

      // 1 hop based on fat links
      //U      = PeekIndex<LorentzIndex>(Ufat, mu);
      U = bcs*PeekIndex<LorentzIndex>(Ufat, mu);
      Udag = adj(Cshift(U, mu, -1));

      U = U*phases;
      Udag = Udag*phases;

	    InsertGaugeField(Uds, U, mu);
	    InsertGaugeField(Uds, Udag, mu+4);

      // 3 hop based on thin links. Crazy huh?
      //U = PeekIndex<LorentzIndex>(Uthin, mu);
      U = bcs*PeekIndex<LorentzIndex>(Uthin, mu);
      UU = Gimpl::CovShiftForward(U, mu, U);
      UUU = Gimpl::CovShiftForward(U, mu, UU);
	
      UUUdag = adj( Cshift(UUU, mu, -3));

      UUU = UUU*phases;
      UUUdag = UUUdag*phases;

	    InsertGaugeField(UUUds, UUU, mu);
	    InsertGaugeField(UUUds, UUUdag, mu+4);
    }
  }

  inline void InsertForce4D(GaugeField &mat, FermionField &Btilde, FermionField &A,int mu){
    GaugeLinkField link(mat.Grid());
    link = TraceIndex<SpinIndex>(outerProduct(Btilde,A)); 
    PokeIndex<LorentzIndex>(mat,link,mu);
  }   
      
  inline void InsertForce5D(GaugeField &mat, FermionField &Btilde, FermionField &Atilde,int mu){
    GRID_ASSERT (0); 
    // Must never hit
  }
};
typedef StaggeredImpl<vComplex,  FundamentalRepresentation > StaggeredImplR;   // Real.. whichever prec
typedef StaggeredImpl<vComplexF, FundamentalRepresentation > StaggeredImplF;  // Float
typedef StaggeredImpl<vComplexD, FundamentalRepresentation > StaggeredImplD;  // Double

NAMESPACE_END(Grid);
