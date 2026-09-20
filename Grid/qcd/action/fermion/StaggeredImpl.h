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
      
  template<class _Spinor, class _Gauge>
  static accelerator_inline void multLink(_Spinor &phi,
		       const _Gauge &U,
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

  inline void setStaggeredPhases(std::vector<ComplexField>& eta) {
    /**
     * @brief Calculates standard staggered phases eta_mu(x)
     * @author Curtis Taylor Peterson, Peter Boyle
     * @brief
     * Calculates phases with the "MILC convention", which treats the fourth 
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
     */
    GRID_ASSERT(!eta[0].Grid()->_isCheckerBoarded);

    GridBase* grid = eta[0].Grid();
    ComplexField phase(grid);
    Lattice<iScalar<vInteger>> x(grid), y(grid), t(grid);
    Lattice<iScalar<vInteger>> tx(grid), txy(grid), xyzt(grid);

    LatticeCoordinate(x, 0);
    LatticeCoordinate(y, 1);
    LatticeCoordinate(t, 3);
    
    tx = t + x;
    txy = tx + y;

    for (int mu = 0; mu < Nd; mu++) {
      phase = 1.0;
      if (mu == 0) { phase = where(mod(t, 2) == (Integer)0,   phase, -phase); }
      if (mu == 1) { phase = where(mod(tx, 2) == (Integer)0,  phase, -phase); }
      if (mu == 2) { phase = where(mod(txy, 2) == (Integer)0, phase, -phase); }
      eta[mu] = phase;
    }
  }

  /** @brief overload for checkerboarded grids; (marginally) wasteful */
  inline void setStaggeredPhases(GridBase* grid, std::vector<ComplexField>& eta) { 
    GRID_ASSERT(!grid->_isCheckerBoarded);
    
    std::vector<ComplexField> _eta(Nd, grid);
    
    setStaggeredPhases(_eta);
    
    for (int mu = 0; mu < Nd; ++mu) {
      if (eta[mu].Grid()->_isCheckerBoarded) { 
        GRID_ASSERT(eta[mu].Grid()->_fdimensions == grid->_fdimensions);
        GRID_ASSERT(eta[mu].Grid()->_processors == grid->_processors);
        GRID_ASSERT(eta[mu].Grid()->_processor_coor == grid->_processor_coor);
        GRID_ASSERT(eta[mu].Grid()->_simd_layout == grid->_simd_layout);

        int cb = eta[mu].Checkerboard();
        int dim = eta[mu].Grid()->_checker_dim;

        acceleratorPickCheckerboard(cb, eta[mu], _eta[mu], dim); 
      } else { conformable(eta[mu].Grid(), grid); eta[mu] = _eta[mu]; }
    }
  }

  inline void setBoundaryPhases(std::vector<ComplexField>& bcs) {
    /**
     * @brief Sets the boundary phases for the gauge links
     * @author Curtis Taylor Peterson, Peter Boyle
     * @details
     * Boundary conditions are imposed by "rephasing" the links on the boundary, 
     * with "1" for periodic and "-1" for anti-periodic.
     */
    GRID_ASSERT(!bcs[0].Grid()->_isCheckerBoarded);

    typedef typename Simd::scalar_type GridScalar;
    GridBase* grid = bcs[0].Grid();
    Lattice<iScalar<vInteger>> coor(grid);

    for (int mu = 0; mu < Nd; ++mu) {
      int N = grid->GlobalDimensions()[mu] - 1;
      auto bpha = Params.boundary_phases[mu];
      ComplexField phase(grid);
      GridScalar dirichlet(real(bpha), imag(bpha));

      phase = 1.0;
      LatticeCoordinate(coor, mu);
      bcs[mu] = where(coor == (Integer)N, dirichlet*phase, phase);
    }
  }

  /** @brief overload for checkerboarded grids; (marginally) wasteful */
  inline void setBoundaryPhases(GridBase* grid, std::vector<ComplexField>& bcs) {
    GRID_ASSERT(!grid->_isCheckerBoarded);
    
    std::vector<ComplexField> _bcs(Nd, grid);

    setBoundaryPhases(_bcs);

    for (int mu = 0; mu < Nd; ++mu) {
      if (bcs[mu].Grid()->_isCheckerBoarded) { 
        GRID_ASSERT(bcs[mu].Grid()->_fdimensions == grid->_fdimensions);
        GRID_ASSERT(bcs[mu].Grid()->_processors == grid->_processors);
        GRID_ASSERT(bcs[mu].Grid()->_processor_coor == grid->_processor_coor);
        GRID_ASSERT(bcs[mu].Grid()->_simd_layout == grid->_simd_layout);

        int cb = bcs[mu].Checkerboard();
        int dim = bcs[mu].Grid()->_checker_dim;

        acceleratorPickCheckerboard(cb, bcs[mu], _bcs[mu], dim); 
      } else { conformable(bcs[mu].Grid(), grid); bcs[mu] = _bcs[mu]; }
    }
  }

  inline void validateDirichletBlock(GridBase* grid, const Coordinate& Block) {
    std::cout << GridLogMessage << " FULL Dirichlet BCs " << Block << std::endl;
    for (int mu = 0; mu < Nd; ++mu) {
      int ldim = grid->LocalDimensions()[mu];
      if (Block[mu]) {
        std::string err = "block size must be multiple of local dimensions";
        GRID_ASSERT((Block[mu] % ldim) == 0 && err.c_str());
    } }
    std::cout << " Dirichlet filtering gauge field BCs block " << Block << std::endl;
  }

  inline void applyDirichletMasks(GaugeField& U, const Coordinate& Block)
  { DirichletFilter<GaugeField> filter(Block); filter.applyFilter(U); }

  void rephase(GridBase* _grid, GaugeField& U) {
    GridBase* grid = U.Grid();
    std::vector<ComplexField> eta(Nd, grid);
    std::vector<ComplexField> bcs(Nd, grid);

    for (int mu = 0; mu < Nd; ++mu) {
      eta[mu].Checkerboard() = U.Checkerboard();
      bcs[mu].Checkerboard() = U.Checkerboard();
    }
    setStaggeredPhases(_grid, eta); // <-+- manage checkerboard internally
    setBoundaryPhases(_grid, bcs);  // <-+
    
    for (int mu = 0; mu < Nd; ++mu) 
    { pokeLorentz(U, eta[mu]*bcs[mu]*peekLorentz(U, mu), mu); }
  }

  inline void NaiveDoubleStore(
    GridBase* grid,
    DoubledGaugeField& Uds,
    const GaugeField& Uin
  ) {
    /**
     * @brief One-link staggered double store; includes staggered and boundary phases
     * @author Curtis Taylor Peterson, Peter Boyle
     */
    conformable(Uds.Grid(), grid);
    conformable(Uin.Grid(), grid);

    GaugeLinkField U(grid);
    GaugeLinkField UU(grid);
    GaugeLinkField UUU(grid);
    GaugeLinkField Udag(grid);
    GaugeLinkField UUUdag(grid);

    std::vector<ComplexField> eta(Nd, grid);
    std::vector<ComplexField> bcs(Nd, grid);

    setStaggeredPhases(eta);
    setBoundaryPhases(bcs);

    for (int mu = 0; mu < Nd; mu++) {
      U = bcs[mu]*PeekIndex<LorentzIndex>(Uin, mu);
      Udag = adj(Cshift(U, mu, -1));

      U = U*eta[mu];
      Udag = Udag*eta[mu];

	    InsertGaugeField(Uds, U, mu);
	    InsertGaugeField(Uds, Udag, mu+4);
    }
  }

  inline void ImprovedDoubleStore(
    GridBase *GaugeGrid,
		DoubledGaugeField &UUUds, // for Naik term
		DoubledGaugeField &Uds,
		const GaugeField &Uthin,
		const GaugeField &Ufat
  ) {
    /**
     * @brief Staggered double store; includes staggered and boundary phases
     * @author Curtis Taylor Peterson, Peter Boyle
     */
    conformable(Uds.Grid(), GaugeGrid);
    conformable(Uthin.Grid(), GaugeGrid);
    conformable(Ufat.Grid(), GaugeGrid);

    GaugeLinkField U(GaugeGrid);
    GaugeLinkField UU(GaugeGrid);
    GaugeLinkField UUU(GaugeGrid);
    GaugeLinkField Udag(GaugeGrid);
    GaugeLinkField UUUdag(GaugeGrid);

    std::vector<ComplexField> eta(Nd, GaugeGrid);
    std::vector<ComplexField> bcs(Nd, GaugeGrid);

    setStaggeredPhases(eta);
    setBoundaryPhases(bcs);

    for (int mu = 0; mu < Nd; mu++) {
      // 1 hop based on fat links
      U = bcs[mu]*PeekIndex<LorentzIndex>(Ufat, mu);
      Udag = adj(Cshift(U, mu, -1));

      U = U*eta[mu];
      Udag = Udag*eta[mu];

	    InsertGaugeField(Uds, U, mu);
	    InsertGaugeField(Uds, Udag, mu+4);

      // 3 hop based on thin links. Crazy huh?
      U = bcs[mu]*PeekIndex<LorentzIndex>(Uthin, mu);
      UU = Gimpl::CovShiftForward(U, mu, U);
      UUU = Gimpl::CovShiftForward(U, mu, UU);
	
      UUUdag = adj( Cshift(UUU, mu, -3));

      UUU = UUU*eta[mu];
      UUUdag = UUUdag*eta[mu];

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

  GaugeLinkField outer(const FermionField &A, const FermionField &B) {
    GaugeLinkField result = Grid::outerProduct(A, B);
    result.Checkerboard() = B.Checkerboard();
    return result;
  }
};

typedef StaggeredImpl<vComplex,  FundamentalRepresentation > StaggeredImplR;   // Real.. whichever prec
typedef StaggeredImpl<vComplexF, FundamentalRepresentation > StaggeredImplF;  // Float
typedef StaggeredImpl<vComplexD, FundamentalRepresentation > StaggeredImplD;  // Double

NAMESPACE_END(Grid);
