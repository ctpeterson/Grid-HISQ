/*************************************************************************************
Grid physics library, www.github.com/paboyle/Grid
Source file: ./lib/qcd/utils/Transporters.h
Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

Copyright (C) 2023 

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
 * @file PeriodicTranspoerters.h
 * @brief Interface for periodic gauge transporters
 * @author Curtis Taylor Peterson
 * @details
 * ...
 */

#pragma once 

#ifndef QCD_UTILS_PERIODIC_TRANSPORTERS_H
#define QCD_UTILS_PERIODIC_TRANSPORTERS_H

NAMESPACE_BEGIN(Grid);

namespace PeriodicBC { 

//
// macros
//

// make scope more explicit
#define ACCELERATOR_SCOPE(exec) { exec; } \

// shorten call to get stencil entry in declaration
#define STENCIL_ENTRY(se, st, mu, n)              \
  GeneralStencilEntry const *se = st.GetEntry(mu, n); \

// shorten call to coalesced read in function
#define ACCREAD(u, se)                                          \
  coalescedReadGeneralPermute(u[se->_offset], se->_permute, Nd) \

// shorten call to coalesced write
#define ACCWRITE(wu, u) coalescedWrite(wu, u) \

// shorted gauge loop construction
#define GAUGELOOP1(ua, ub) ua * adj(ub) \

// shorted gauge loop construction
#define GAUGELOOP2(ua, ub) adj(ua) * ub \

// shorten out-of-class implementation definitions
#define IMPL(x) template <class Gimpl>                  \
inline const typename x<Gimpl>::GaugeLinkField x<Gimpl> \

//
// convenient data structures
//

enum TransportHeading {FORWARD = 0, BACKWARD = Nd};

//
// periodic transporter definition
//

template <class Gimpl>
class Transporter: public Gimpl {
/**
 * @brief periodic gauge transporter
 * @author Curtis Taylor Peterson
 * @details
 * Represents periodic transport operations needed for building up common gauge 
 * link constructs. 
 */
public: INHERIT_GIMPL_TYPES(Gimpl)

private:
  int depth, mu;
  std::shared_ptr<GeneralLocalStencil> _stencil;
  std::unique_ptr<GaugeLinkField> _link;

public:
  /** @brief default Transporter constructor */
  Transporter() { _stencil.reset(); _link.reset(); }

  /** @brief main Transporter constructor */
  Transporter(
    GridBase* grid, 
    const GaugeLinkField& link, 
    int mu,
    int depth
  ): mu(mu), depth(depth) {
    GridCartesian* cgrid = (GridCartesian*)grid;
    std::vector<Coordinate> shifts;
    Coordinate backward(Nd, 0);
    Coordinate forward(Nd, 0);

    backward[mu] = -depth;
    forward[mu] = depth;

    shifts.push_back(backward);
    shifts.push_back(forward);

    _stencil = std::make_unique<GeneralLocalStencil>(GeneralLocalStencil(cgrid, shifts));
    _link = std::make_unique<GaugeLinkField>(link);
  }

// accessors
public:
  /** @brief return copy of input buffer */
  inline const GaugeLinkField link() { return *_link; }

  /** @brief return copy of general local stencil */
  inline const GeneralLocalStencil stencil() { return *_stencil; }

// covariant shifts
public:
  inline const GaugeLinkField CovShift(
    const GaugeLinkField& u,
    const GaugeLinkField& v, 
    TransportHeading heading
  );

  inline const GaugeLinkField CovShift(
    const GaugeLinkField& v, 
    TransportHeading heading
  );

  inline const GaugeLinkField CovShiftFwd(
    const GaugeLinkField& u, 
    const GaugeLinkField& v
  );

  inline const GaugeLinkField CovShiftFwd(const GaugeLinkField& v);

  inline const GaugeLinkField CovShiftFwd();

  inline const GaugeLinkField CovShiftBck(
    const GaugeLinkField& u, 
    const GaugeLinkField& v
  );

  inline const GaugeLinkField CovShiftBck(const GaugeLinkField& v);

  inline const GaugeLinkField CovShiftBck();

// cartesian shifts
public:
  inline const GaugeLinkField Cshift(
    const GaugeLinkField& v, 
    TransportHeading heading
  );

  inline const GaugeLinkField CovShiftIdent(
    const GaugeLinkField& v, 
    TransportHeading heading
  );

  inline const GaugeLinkField CovShiftIdentFwd(const GaugeLinkField& v);

  inline const GaugeLinkField CovShiftIdentBck(const GaugeLinkField& v);
};

//
// periodic transporter container definition
//

template <class Gimpl>
class Transporters: public Gimpl {
/**
 * @brief Transporter container
 * @author Curtis Taylor Peterson
 * @details
 * Naive container for saving and indexing a collection of Transporters. 
 * Note of caution: all periodic gauge transporters use the same buffer for the 
 * stencil and the outputs.
 */

public: INHERIT_GIMPL_TYPES(Gimpl)

private:
  int depth;

  GridBase* _pgrid;
  PaddedCell* _pcell;
  
  std::shared_ptr<GeneralLocalStencil> _stencil;
  std::vector<std::vector<std::shared_ptr<GeneralLocalStencil>>> _stencils;
  
  Transporter<Gimpl> _transporter[Nd];

public:
  Transporters(PaddedCell& pcell, const GaugeField& Uin): depth(pcell.depth) {
    auto U = pcell.ExchangePeriodic(Uin);
    std::vector<Coordinate> shifts;

    _pcell = &pcell;
    _pgrid = (GridBase*)_pcell->grids.back();
    for (int mu = 0; mu < Nd; ++mu) 
      _transporter[mu] = Transporter<Gimpl>(_pgrid, toLink(U, mu), mu, depth);

    createMuStencil(depth);
    createMuNuStencils(depth);
  }

private:
  void createMuStencil(int depth) {
    GridCartesian* cgrid = (GridCartesian*)_pgrid;

    std::vector<Coordinate> shifts;

    for (int mu = 0; mu < Nd; ++mu)
      for (int d = 0; d < depth; ++d) { 
        Coordinate forward(Nd, 0); 
        forward[mu] = d + 1; 
        shifts.push_back(forward); 
      }
    
    for (int mu = 0; mu < Nd; ++mu)
      for (int d = 0; d < depth; ++d) {
        Coordinate backward(Nd, 0);
        backward[mu] = -(d + 1);
        shifts.push_back(backward);
      }

    _stencil = std::make_shared<GeneralLocalStencil>(GeneralLocalStencil(cgrid, shifts));
  }

  void createMuNuStencils(int depth) {
    GridCartesian* cgrid = (GridCartesian*)_pgrid;

    for (int mu = 0; mu < Nd; ++mu) {
      std::vector<std::shared_ptr<GeneralLocalStencil>> stencils;

      for (int nu = 0; nu < Nd; ++nu) {
        if (mu == nu) {
          stencils.push_back(nullptr);
          continue;
        }

        std::vector<Coordinate> shifts;
        Coordinate shift_pmu_pnu(Nd, 0);
        Coordinate shift_pmu_mnu(Nd, 0);
        Coordinate shift_mmu_pnu(Nd, 0);
        Coordinate shift_mmu_mnu(Nd, 0);

        shift_pmu_pnu[mu] = depth;   shift_pmu_pnu[nu] = depth;
        shift_pmu_mnu[mu] = depth;   shift_pmu_mnu[nu] = -depth;
        shift_mmu_pnu[mu] = -depth;  shift_mmu_pnu[nu] = depth;
        shift_mmu_mnu[mu] = -depth;  shift_mmu_mnu[nu] = -depth;

        shifts.push_back(shift_pmu_pnu);
        shifts.push_back(shift_pmu_mnu);
        shifts.push_back(shift_mmu_pnu);
        shifts.push_back(shift_mmu_mnu);

        stencils.push_back(
          std::make_shared<GeneralLocalStencil>(GeneralLocalStencil(cgrid, shifts))
        );
      }

      _stencils.push_back(stencils);
    }
  }

private:
  GaugeLinkField toLink(const GaugeField& U, int mu)
  { return PeekIndex<LorentzIndex>(U, mu); }

public:
  /** @brief wrapped halo exchange */
  inline const GaugeField toPaddedGrid(const GaugeField& U) 
  { return _pcell->ExchangePeriodic(U); }

  /** @brief wrapped halo exchange */
  inline const GaugeLinkField toPaddedGrid(const GaugeLinkField& U) 
  { return _pcell->ExchangePeriodic(U); }

  /** @brief wrapped halo exchange */
  inline const LatticeComplex toPaddedGrid(const LatticeComplex& U) 
  { return _pcell->ExchangePeriodic(U); }

public:
  /** @brief wrapped extraction from padding */
  inline const GaugeField toTightGrid(const GaugeField& U) 
  { return _pcell->Extract(U); }

  /** @brief wrapped extraction from padding */
  inline const GaugeLinkField toTightGrid(const GaugeLinkField& U) 
  { return _pcell->Extract(U); }

  /** @brief wrapped extraction from padding */
  inline const LatticeComplex toTightGrid(const LatticeComplex& U) 
  { return _pcell->Extract(U); }

public:
  /** @brief directional halo exchange */
  /*
  inline GaugeLinkField exchange(const GaugeLinkField& u, int mu) { 
    Coordinate processors = _pcell->unpadded_grid->_processors;
    GaugeLinkField tu = toTightGrid(u);
    GridBase* tgrid = tu.Grid();
    GridCartesian* pgrid = _pcell->grids[mu];
    GaugeLinkField pu(pgrid);

    if (processors[mu] == 1) pu = tu;
    else _pcell->Face_exchange(tu, pu, mu, depth);

    return pu;
  }
  */

  /** @brief full halo exchange */
  inline GaugeLinkField exchange(const GaugeLinkField& u) 
  { return toPaddedGrid(toTightGrid(u)); }

  /** @brief update followed by halo exchange */
  void append(GaugeLinkField& u, const GaugeLinkField& du) 
  { u += du; u = exchange(u); }

public:
  /** @brief index transporters -- mutable */
  inline Transporter<Gimpl>& operator[](int mu) { return _transporter[mu]; }

  /** @brief index transporters -- not mutable */
  inline const Transporter<Gimpl>& operator[](int mu) const 
  { return _transporter[mu]; }

public:
  /** @brief return copy of input buffer for transporter */
  inline const GaugeLinkField link(int mu) { return _transporter[mu].link(); }

  /** @brief return copy of general local stencil */
  inline const GeneralLocalStencil stencil() { return *_stencil; }

  /** @breif return copy of general local stencil from transporters */
  inline const GeneralLocalStencil stencil(int mu) 
  { return _transporter[mu].stencil(); }

  /** @brief return copy of general local stencil for mu-nu pairs */
  inline const GeneralLocalStencil stencil(int mu, int nu) 
  { return *(_stencils[mu][nu]); }

public:
  inline const GaugeLinkField CovShiftFwd(
    int mu, 
    const GaugeLinkField& u, 
    const GaugeLinkField& v,
    bool correct_boundaries = true
  ) { 
    if (correct_boundaries) return exchange(_transporter[mu].CovShiftFwd(u, v)); 
    else return _transporter[mu].CovShiftFwd(u, v);
  }

  inline const GaugeLinkField CovShiftFwd(
    int mu, 
    const GaugeLinkField& v,
    bool correct_boundaries = true
  ) { 
    if (correct_boundaries) return exchange(_transporter[mu].CovShiftFwd(v)); 
    else return _transporter[mu].CovShiftFwd(v);
  }

  inline const GaugeLinkField CovShiftFwd(
    int mu, 
    int nu,
    bool correct_boundaries = true
  ) { 
    if (correct_boundaries) return exchange(_transporter[mu].CovShiftFwd(link(nu))); 
    else return _transporter[mu].CovShiftFwd(link(nu));
  }

  inline const GaugeLinkField CovShiftFwd(int mu, bool correct_boundaries = true) { 
    if (correct_boundaries) return exchange(_transporter[mu].CovShiftFwd()); 
    else return _transporter[mu].CovShiftFwd();
  }

  inline const GaugeLinkField CovShiftBck(
    int mu,
    const GaugeLinkField& u, 
    const GaugeLinkField& v,
    bool correct_boundaries = true
  ) { 
    if (correct_boundaries) return exchange(_transporter[mu].CovShiftBck(u, v)); 
    else return _transporter[mu].CovShiftBck(u, v);
  }

  inline const GaugeLinkField CovShiftBck(
    int mu, 
    const GaugeLinkField& v, 
    bool correct_boundaries = true
  ) { 
    if (correct_boundaries) return exchange(_transporter[mu].CovShiftBck(v)); 
    else return _transporter[mu].CovShiftBck(v);
  }

  inline const GaugeLinkField CovShiftBck(
    int mu, 
    int nu, 
    bool correct_boundaries = true
  ) { 
    if (correct_boundaries) return exchange(_transporter[mu].CovShiftBck(link(nu))); 
    else return _transporter[mu].CovShiftBck(link(nu));
  }

  inline const GaugeLinkField CovShiftBck(int mu, bool correct_boundaries = true) { 
    if (correct_boundaries) return exchange(_transporter[mu].CovShiftBck()); 
    else return _transporter[mu].CovShiftBck();
  }

public:
  inline const GaugeLinkField CovShiftIdentFwd(
    int mu, 
    const GaugeLinkField& v,
    bool correct_boundaries = true
  ) { 
    if (correct_boundaries) return exchange(_transporter[mu].CovShiftIdentFwd(v)); 
    else return _transporter[mu].CovShiftIdentFwd(v);
  }

  inline const GaugeLinkField CovShiftIdentBck(
    int mu, 
    const GaugeLinkField& v,
    bool correct_boundaries = true
  ) { 
    if (correct_boundaries) return exchange(_transporter[mu].CovShiftIdentBck(v)); 
    else return _transporter[mu].CovShiftIdentBck(v);
  }

  inline const GaugeLinkField CovShiftIdentFwd(
    int mu, 
    int nu,
    bool correct_boundaries = true
  ) { 
    if (correct_boundaries) 
      return exchange(_transporter[mu].CovShiftIdentFwd(link(nu))); 
    else return _transporter[mu].CovShiftIdentFwd(link(nu));
  }

  inline const GaugeLinkField CovShiftIdentBck(
    int mu, 
    int nu,
    bool correct_boundaries = true
  ) { 
    if (correct_boundaries) 
      return exchange(_transporter[mu].CovShiftIdentBck(link(nu))); 
    else return _transporter[mu].CovShiftIdentBck(link(nu));
  }

public:
  inline const GaugeLinkField ident() 
  { GaugeLinkField id(_pgrid); id = 1.0; return id; }

  inline const GaugeLinkField ident(int mu) { return ident(); } // aesthetic

public:
  inline const GaugeLinkField _staple(
    const GaugeLinkField& u, 
    const GaugeLinkField& v,
    int mu, 
    int nu
  );

public:
  inline const GaugeLinkField staple(
    const GaugeLinkField& u, 
    const GaugeLinkField& v,
    int mu, 
    int nu,
    bool correct_boundaries = true
  );

  inline const GaugeLinkField staple(
    const GaugeLinkField& v, 
    int mu, 
    int nu,
    bool correct_boundaries = true
  );

  inline const GaugeLinkField staple(int mu, int nu, bool correct_boundaries = true);

public:
  inline const GaugeLinkField _stapleDerivative(
    const GaugeLinkField& v,
    const GaugeLinkField& u,
    const GaugeLinkField& c,
    int mu,
    int nu
  );

public:
  inline const GaugeLinkField stapleDerivative(
    const GaugeLinkField& v,
    const GaugeLinkField& u,
    const GaugeLinkField& c,
    int mu,
    int nu,
    bool correct_boundaries = true
  );

  inline const GaugeLinkField stapleDerivative(
    const GaugeLinkField& v,
    const GaugeLinkField& c,
    int mu,
    int nu,
    bool correct_boundaries = true
  );

  inline const GaugeLinkField stapleDerivative(
    const GaugeLinkField& c, 
    int mu, 
    int nu,
    bool correct_boundaries = true
  );

public:
  inline const GaugeLinkField upperStaple(
    const GaugeLinkField& u, // mu link
    const GaugeLinkField& lv, // left nu link
    const GaugeLinkField& rv, // right nu link
    int mu,
    int nu
  );

  inline const GaugeLinkField upperStaple(
    const GaugeLinkField& lv, // left nu link
    const GaugeLinkField& rv, // right nu link
    int mu,
    int nu
  ) { return upperStaple(link(mu), lv, rv, mu, nu); }

  inline const GaugeLinkField upperStaple(
    const GaugeLinkField& u, // mu link
    int mu,
    int nu
  ) { return upperStaple(u, link(nu), link(nu), mu, nu); }

  inline const GaugeLinkField upperStaple(int mu, int nu) 
  { return upperStaple(link(mu), link(nu), link(nu), mu, nu); }

public:
  inline const GaugeLinkField lowerStaple(
    const GaugeLinkField& u, // mu link
    const GaugeLinkField& lv, // left nu link
    const GaugeLinkField& rv, // right nu link
    int mu,
    int nu
  );

  inline const GaugeLinkField lowerStaple(
    const GaugeLinkField& lv, // left nu link
    const GaugeLinkField& rv, // right nu link
    int mu,
    int nu
  ) { return lowerStaple(link(mu), lv, rv, mu, nu); }

  inline const GaugeLinkField lowerStaple(
    const GaugeLinkField& u, // mu link
    int mu,
    int nu
  ) { return lowerStaple(u, link(nu), link(nu), mu, nu); }

  inline const GaugeLinkField lowerStaple(int mu, int nu) 
  { return lowerStaple(link(mu), link(nu), link(nu), mu, nu); }

public:
  inline const GaugeLinkField rightStaple(
    const GaugeLinkField& u, // nu link
    const GaugeLinkField& rv, // top mu link
    const GaugeLinkField& bv, // bottom mu link
    int mu,
    int nu
  );

  inline const GaugeLinkField rightStaple(
    const GaugeLinkField& rv, // top mu link
    const GaugeLinkField& bv, // bottom mu link
    int mu,
    int nu
  ) { return rightStaple(link(nu), rv, bv, mu, nu); }

  inline const GaugeLinkField rightStaple(
    const GaugeLinkField& u, // nu link
    int mu,
    int nu
  ) { return rightStaple(u, link(mu), link(mu), mu, nu); }

  inline const GaugeLinkField rightStaple(int mu, int nu) 
  { return rightStaple(link(nu), link(mu), link(mu), mu, nu); }

public:
  inline const GaugeLinkField leftStaple(
    const GaugeLinkField& u, // nu link
    const GaugeLinkField& rv, // top mu link
    const GaugeLinkField& bv, // bottom mu link
    int mu,
    int nu
  );

  inline const GaugeLinkField leftStaple(
    const GaugeLinkField& rv, // top mu link
    const GaugeLinkField& bv, // bottom mu link
    int mu,
    int nu
  ) { return leftStaple(link(nu), rv, bv, mu, nu); }

  inline const GaugeLinkField leftStaple(
    const GaugeLinkField& u, // nu link
    int mu,
    int nu
  ) { return leftStaple(u, link(mu), link(mu), mu, nu); }

  inline const GaugeLinkField leftStaple(int mu, int nu) 
  { return leftStaple(link(nu), link(mu), link(mu), mu, nu); }
};

//
// periodic transporter implementation
//

//-- covariant shifts --//

/** @brief application of gauge transporter to operand field */
IMPL(Transporter)::CovShift(
  const GaugeLinkField& u,
  const GaugeLinkField& v, 
  TransportHeading heading
) {
  bool forward = heading == FORWARD;
  int direction = forward ? 1 : 0;
  GaugeLinkField f(u.Grid());

  ACCELERATOR_SCOPE(
    GeneralLocalStencilView s_v = stencil().View(AcceleratorRead);
    autoView(v_v, v, AcceleratorRead);
    autoView(u_v, u, AcceleratorRead);
    autoView(f_v, f, AcceleratorWrite);

    // no overhead from branching inside accelerator loop bc heading constant
    accelerator_for(n, f_v.size(), Simd::Nsimd(), {
      STENCIL_ENTRY(se, s_v, direction, n);
      if (forward) ACCWRITE(f_v[n], u_v[n]*ACCREAD(v_v, se));
      else ACCWRITE(f_v[n], adj(ACCREAD(u_v, se))*ACCREAD(v_v, se));
    });
  )

  return f;
}

/** @brief application of gauge transporter to operand field */
IMPL(Transporter)::CovShift(const GaugeLinkField& v, TransportHeading heading) 
{ return CovShift(link(), v, heading); }

/** @brief application of gauge transporter to operand field */
IMPL(Transporter)::CovShiftFwd(const GaugeLinkField& u, const GaugeLinkField& v) 
{ return CovShift(u, v, FORWARD); }

/** @brief application of gauge transporter to operand field */
IMPL(Transporter)::CovShiftFwd(const GaugeLinkField& v) 
{ return CovShift(link(), v, FORWARD); }

/** @brief application of gauge transporter to operand field */
IMPL(Transporter)::CovShiftFwd() 
{ return CovShift(link(), link(), FORWARD); }

/** @brief application of gauge transporter to operand field */
IMPL(Transporter)::CovShiftBck(const GaugeLinkField& u, const GaugeLinkField& v) 
{ return CovShift(u, v, BACKWARD); }

/** @brief application of gauge transporter to operand field */
IMPL(Transporter)::CovShiftBck(const GaugeLinkField& v) 
{ return CovShift(link(), v, BACKWARD); }

/** @brief application of gauge transporter to operand field */
IMPL(Transporter)::CovShiftBck() 
{ return CovShift(link(), link(), BACKWARD); }

//-- cartesian shifts --//

/** @brief application of shift operation to operand field */
IMPL(Transporter)::Cshift(const GaugeLinkField& u, TransportHeading heading) {
  bool forward = heading == FORWARD;
  int direction = forward ? 1 : 0;
  GaugeLinkField f(u.Grid());

  ACCELERATOR_SCOPE(
    GeneralLocalStencilView s_v = stencil().View(AcceleratorRead);
    autoView(u_v, u, AcceleratorRead);
    autoView(f_v, f, AcceleratorWrite);

    accelerator_for(n, f_v.size(), Simd::Nsimd(), {
      STENCIL_ENTRY(se, s_v, direction, n); 
      ACCWRITE(f_v[n], ACCREAD(u_v, se));
    });
  )

  return f;
}

/** @brief slightly more satisfying name for Cartesian shift */
IMPL(Transporter)::CovShiftIdent(const GaugeLinkField& v, TransportHeading heading) 
{ return Cshift(v, heading); }

/** @brief slightly more satisfying name for Cartesian shift */
IMPL(Transporter)::CovShiftIdentFwd(const GaugeLinkField& v) 
{ return Cshift(v, FORWARD); }

/** @brief slightly more satisfying name for Cartesian shift */
IMPL(Transporter)::CovShiftIdentBck(const GaugeLinkField& v) 
{ return Cshift(v, BACKWARD); }

//
// periodic transporters implementation
//

//-- symmetric staple --//

/** @brief calculate symmetric staple */
IMPL(Transporters)::_staple(
  const GaugeLinkField& u, // mu link
  const GaugeLinkField& v, // nu link
  int mu, 
  int nu
) {
  /**
   * @brief Calculates "symmetric" staple (i.e., staple + its reflection)
   * @author Curtis Taylor Peterson
   * @details
   * Calculates closed/symmetric staple with orientation
   *         ---🠢
   *         🠡   🠣
   * ν       x---x
   * 🠡       🠣   🠡
   * -🠢 μ    ---🠢 
   * What is meant by "symmetric" is that the staple and its reflection about
   * the plane bisecting the mu-link are added together.
   */ 
  GaugeLinkField rs(_pgrid);

  ACCELERATOR_SCOPE(
    GeneralLocalStencilView smu_v = stencil(mu).View(AcceleratorRead);
    GeneralLocalStencilView snu_v = stencil(nu).View(AcceleratorRead);
    GeneralLocalStencilView smunu_v = stencil(mu, nu).View(AcceleratorRead);
    
    autoView(u_v, u, AcceleratorRead);
    autoView(v_v, v, AcceleratorRead);
    autoView(rs_v, rs, AcceleratorWrite);

    accelerator_for(n, _pgrid->oSites(), Simd::Nsimd(), {
      STENCIL_ENTRY(se_pmu, smu_v, 1, n);
      STENCIL_ENTRY(se_mnu, snu_v, 0, n);
      STENCIL_ENTRY(se_pnu, snu_v, 1, n);
      STENCIL_ENTRY(se_pmu_mnu, smunu_v, 1, n);

      // upper staple
      auto u_xpnu = ACCREAD(u_v, se_pnu);
      auto v_xpmu = ACCREAD(v_v, se_pmu);
      auto staple = v_v[n]*u_xpnu*adj(v_xpmu);

      // lower staple
      auto v_xmnu = ACCREAD(v_v, se_mnu);
      auto u_xmnu = ACCREAD(u_v, se_mnu);
      auto v_xmnu_pmu = ACCREAD(v_v, se_pmu_mnu);
      staple = staple + adj(v_xmnu)*u_xmnu*v_xmnu_pmu;

      // full result
      ACCWRITE(rs_v[n], staple);
    });
  )

  return rs;
}

IMPL(Transporters)::staple(
  const GaugeLinkField& u, // mu link
  const GaugeLinkField& v, // nu link
  int mu, 
  int nu,
  bool correct_boundaries
) { 
  if (correct_boundaries) return exchange(_staple(u, v, mu, nu)); 
  else return _staple(u, v, mu, nu); 
}

/** @brief calculate symmetric staple */
IMPL(Transporters)::staple(
  const GaugeLinkField& u, 
  int mu, 
  int nu, 
  bool correct_boundaries
) { 
  if (correct_boundaries) return exchange(_staple(u, link(nu), mu, nu)); 
  else return _staple(u, link(nu), mu, nu); 
}

/** @brief calculate symmetric staple using buffer fields **/
IMPL(Transporters)::staple(int mu, int nu, bool correct_boundaries) { 
  if (correct_boundaries) return exchange(_staple(link(mu), link(nu), mu, nu)); 
  else return _staple(link(mu), link(nu), mu, nu); 
}

//-- upper staple --//

IMPL(Transporters)::upperStaple(
  const GaugeLinkField& u, // mu link
  const GaugeLinkField& lv, // left nu link
  const GaugeLinkField& rv, // right nu link
  int mu,
  int nu
) {
  /**
   * @brief Calculates upper staple only
   * @author Curtis Taylor Peterson
   * @details
   * Calculates upper staple only with orientation
   * ν      ---🠢
   * 🠡      🠡   🠣
   * -🠢 μ  x
   */ 
  GaugeLinkField us(_pgrid);

  ACCELERATOR_SCOPE(
    GeneralLocalStencilView s_v = stencil().View(AcceleratorRead);
    autoView(u_v, u, AcceleratorRead);
    autoView(lv_v, lv, AcceleratorRead);
    autoView(rv_v, rv, AcceleratorRead);
    autoView(us_v, us, AcceleratorWrite);

    accelerator_for(n, us_v.size(), Simd::Nsimd(), {
      STENCIL_ENTRY(se_mu, s_v, mu, n);
      STENCIL_ENTRY(se_nu, s_v, nu, n);
      ACCWRITE(us_v[n], lv_v[n]*ACCREAD(u_v, se_nu)*adj(ACCREAD(rv_v, se_mu)));
    });
  )

  return exchange(us);
}

//-- lower staple --//

IMPL(Transporters)::lowerStaple(
  const GaugeLinkField& u, // mu link
  const GaugeLinkField& lv, // left nu link
  const GaugeLinkField& rv, // right nu link
  int mu,
  int nu
) {
  /**
   * @brief Calculates lower staple only
   * @author Curtis Taylor Peterson
   * @details
   * Calculates lower staple only with orientation
   * ν      x
   * 🠡      🠣   🠡
   * -🠢 μ   ---🠢 
   */ 
  GaugeLinkField ls(_pgrid);

  ACCELERATOR_SCOPE(
    GeneralLocalStencilView s_v = stencil().View(AcceleratorRead);
    autoView(u_v, u, AcceleratorRead);
    autoView(lv_v, lv, AcceleratorRead);
    autoView(rv_v, rv, AcceleratorRead);
    autoView(ls_v, ls, AcceleratorWrite);

    accelerator_for(n, ls_v.size(), Simd::Nsimd(), {
      STENCIL_ENTRY(se, s_v, mu, n);
      ACCWRITE(ls_v[n], adj(lv_v[n])*u_v[n]*ACCREAD(rv_v, se));
    });
  )

  return CovShiftIdentBck(nu, exchange(ls));
}

//-- right staple --//

IMPL(Transporters)::rightStaple(
  const GaugeLinkField& u, // nu link
  const GaugeLinkField& rv, // top mu link
  const GaugeLinkField& bv, // bottom mu link
  int mu,
  int nu
) {
  /**
   * @brief Calculates right staple only
   * @author Curtis Taylor Peterson
   * @details
   * Calculates right staple only with orientation
   *       🠤----      
   * ν          🠡
   * 🠡   x ----🠢      
   * -🠢 μ    
   */ 
  GaugeLinkField rs(_pgrid);

  ACCELERATOR_SCOPE(
    GeneralLocalStencilView s_v = stencil().View(AcceleratorRead);
    autoView(u_v, u, AcceleratorRead);
    autoView(rv_v, rv, AcceleratorRead);
    autoView(bv_v, bv, AcceleratorRead);
    autoView(rs_v, rs, AcceleratorWrite);

    accelerator_for(n, rs_v.size(), Simd::Nsimd(), {
      STENCIL_ENTRY(se_mu, s_v, mu, n);
      STENCIL_ENTRY(se_nu, s_v, nu, n);
      ACCWRITE(rs_v[n], bv_v[n]*ACCREAD(u_v, se_mu)*adj(ACCREAD(rv_v, se_nu)));
    });
  )

  return exchange(rs);
}

//-- left staple --//

IMPL(Transporters)::leftStaple(
  const GaugeLinkField& u, // nu link
  const GaugeLinkField& rv, // top mu link
  const GaugeLinkField& bv, // bottom mu link
  int mu,
  int nu
) {
  /**
   * @brief Calculates left staple only
   * @author Curtis Taylor Peterson
   * @details
   * Calculates left staple only with orientation
   *      ----🠢      
   * ν    🠡
   * 🠡    🠤---- x     
   * -🠢 μ    
   */ 
  GaugeLinkField ls(_pgrid);

  ACCELERATOR_SCOPE(
    GeneralLocalStencilView s_v = stencil().View(AcceleratorRead);
    autoView(u_v, u, AcceleratorRead);
    autoView(rv_v, rv, AcceleratorRead);
    autoView(bv_v, bv, AcceleratorRead);
    autoView(ls_v, ls, AcceleratorWrite);

    accelerator_for(n, ls_v.size(), Simd::Nsimd(), {
      STENCIL_ENTRY(se, s_v, nu, n);
      ACCWRITE(ls_v[n], adj(bv_v[n])*u_v[n]*ACCREAD(rv_v, se));
    });
  )

  return CovShiftIdentBck(mu, exchange(ls));
}

//-- symmetric staple derivative --//

IMPL(Transporters)::_stapleDerivative(
  const GaugeLinkField& u, // mu link
  const GaugeLinkField& v, // nu link
  const GaugeLinkField& c, // chain
  int mu,
  int nu
) {
  /**
   * @brief Calculates nu-oriented derivative of symmetric staple
   * @author Curtis Taylor Peterson
   * @details
   * Calculates nu-oriented derivative of closed/symmetric staple. See diagram in
   * symmetric staple method for orientation of symmetric staple. Derivative "D"
   * of just nu links (what this method calculates) is
   *            ----🠢      ----🠢 
   *            ⮾    🠣     🠡   ⮾ 
   * ν     D =  x----x  +  x----x
   * 🠡          ⮾    🠡     🠣   ⮾
   * -🠢 μ       ----🠢      ----🠢 
   *             [1a]       [1b] 
   * where replacement of "🠣" or "🠡" with ⮾ indicates replacement of link with 
   * contribution from chain rule (i.e., action of derivative).
   */
  GaugeLinkField rds(_pgrid);

  ACCELERATOR_SCOPE(
    GeneralLocalStencilView smu_v = stencil(mu).View(AcceleratorRead);
    GeneralLocalStencilView snu_v = stencil(nu).View(AcceleratorRead);
    GeneralLocalStencilView smunu_v = stencil(mu, nu).View(AcceleratorRead);
    
    autoView(u_v, u, AcceleratorRead);
    autoView(v_v, v, AcceleratorRead);
    autoView(c_v, c, AcceleratorRead);
    autoView(rds_v, rds, AcceleratorWrite);

    accelerator_for(n, _pgrid->oSites(), Simd::Nsimd(), {
      STENCIL_ENTRY(se_pmu, smu_v, 1, n);
      STENCIL_ENTRY(se_mnu, snu_v, 0, n);
      STENCIL_ENTRY(se_pnu, snu_v, 1, n);
      STENCIL_ENTRY(se_pmu_mnu, smunu_v, 1, n);

      // left: upper staple
      auto u_xpnu = ACCREAD(u_v, se_pnu);
      auto v_xpmu = ACCREAD(v_v, se_pmu);
      auto staple = c_v[n]*u_xpnu*adj(v_xpmu);

      // left: lower staple
      auto c_xmnu = ACCREAD(c_v, se_mnu);
      auto u_xmnu = ACCREAD(u_v, se_mnu);
      auto v_xmnu_pmu = ACCREAD(v_v, se_pmu_mnu);
      staple = staple + adj(c_xmnu)*u_xmnu*v_xmnu_pmu;

      // right: upper staple
      auto c_xpmu = ACCREAD(c_v, se_pmu);
      staple = staple + v_v[n]*u_xpnu*adj(c_xpmu);

      // right: lower staple
      auto v_xmnu = ACCREAD(v_v, se_mnu);
      auto c_xmnu_pmu = ACCREAD(c_v, se_pmu_mnu);
      staple = staple + adj(v_xmnu)*u_xmnu*c_xmnu_pmu;

      // full result
      ACCWRITE(rds_v[n], staple);
    });
  )
  
  return rds;
}

//-- symmetric staple derivative --//

IMPL(Transporters)::stapleDerivative(
  const GaugeLinkField& u, // mu link
  const GaugeLinkField& v, // nu link
  const GaugeLinkField& c, // chain
  int mu,
  int nu,
  bool correct_boundaries
) { 
  if (correct_boundaries) return exchange(_stapleDerivative(u, v, c, mu, nu));
  else return _stapleDerivative(u, v, c, mu, nu); 
}

/** @brief symmetric staple derivative w/o passing of middle link */
IMPL(Transporters)::stapleDerivative(
  const GaugeLinkField& v,
  const GaugeLinkField& c,
  int mu,
  int nu,
  bool correct_boundaries
) { return stapleDerivative(link(mu), v, c, mu, nu, correct_boundaries); }

/** @brief symmetric staple derivative w/o explicit middle/side links */
IMPL(Transporters)::stapleDerivative(
  const GaugeLinkField& c, 
  int mu, 
  int nu,
  bool correct_boundaries
) { return stapleDerivative(link(mu), link(nu), c, mu, nu, correct_boundaries); }

// undefine macros to prevent conflicts
#undef ACCELERATOR_SCOPE
#undef STENCIL_ENTRY
#undef ACCREAD
#undef ACCWRITE
#undef GAUGELOOP1
#undef GAUGELOOP2
#undef IMPL

} // namespace PeriodicBC

NAMESPACE_END(Grid);

#endif // QCD_UTILS_PERIODIC_TRANSPORTERS_H
